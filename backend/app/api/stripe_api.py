from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session
from typing import List

from ..core.database import get_db
from ..core.auth import get_current_user
from ..models.user import User
from ..schemas.stripe_schemas import (
    PaymentIntentCreate,
    PaymentIntentResponse,
    PaymentStatus,
    CreditBundleInfo,
    CustomerInfo,
)
from ..services.stripe_service import StripeService
from ..services.credit_service import CreditService
from ..core.config import settings

router = APIRouter(prefix="/api/stripe", tags=["stripe"])


@router.get("/config")
async def get_stripe_config():
    """Get Stripe configuration for frontend"""
    return {"publishable_key": settings.stripe_publishable_key}


# Credit bundle configurations
CREDIT_BUNDLES = {
    "10_credits": {
        "name": "Starter Pack",
        "credits": 10,
        "price": 999,  # $9.99
        "description": "Perfect for trying out VoiceFlow Studio",
        "popular": False,
    },
    "25_credits": {
        "name": "Creator Pack",
        "credits": 25,
        "price": 1999,  # $19.99
        "description": "Great for regular podcast creation",
        "popular": True,
    },
    "50_credits": {
        "name": "Pro Pack",
        "credits": 50,
        "price": 3499,  # $34.99
        "description": "Best value for serious creators",
        "popular": False,
    },
    "100_credits": {
        "name": "Studio Pack",
        "credits": 100,
        "price": 5999,  # $59.99
        "description": "Maximum credits for power users",
        "popular": False,
    },
}


@router.get("/bundles", response_model=List[CreditBundleInfo])
async def get_credit_bundles():
    """Get available credit bundles"""
    bundles = []
    for bundle_id, info in CREDIT_BUNDLES.items():
        bundles.append(
            CreditBundleInfo(
                id=bundle_id,
                name=info["name"],
                credits=info["credits"],
                price=info["price"],
                price_formatted=f"${info['price'] / 100:.2f}",
                description=info["description"],
                popular=info["popular"],
            )
        )
    return bundles


@router.post("/create-payment-intent", response_model=PaymentIntentResponse)
async def create_payment_intent(
    payment_data: PaymentIntentCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a payment intent for purchasing credits"""

    # Get bundle information
    bundle_info = CREDIT_BUNDLES.get(payment_data.bundle.value)
    if not bundle_info:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid credit bundle"
        )

    # Ensure user has a Stripe customer ID
    if not current_user.stripe_customer_id:
        customer_id = StripeService.create_customer(current_user)
        if not customer_id:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to create customer",
            )

        # Update user with Stripe customer ID
        current_user.stripe_customer_id = customer_id
        db.commit()
    else:
        customer_id = current_user.stripe_customer_id

    # Create payment intent
    payment_intent = StripeService.create_payment_intent(
        amount=bundle_info["price"],
        customer_id=customer_id,
        metadata={
            "user_id": str(current_user.id),
            "credits": str(bundle_info["credits"]),
            "bundle": payment_data.bundle.value,
            "bundle_name": bundle_info["name"],
        },
    )

    if not payment_intent:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create payment intent",
        )

    return PaymentIntentResponse(
        client_secret=payment_intent["client_secret"],
        payment_intent_id=payment_intent["payment_intent_id"],
        amount=payment_intent["amount"],
        currency=payment_intent["currency"],
        status=payment_intent["status"],
        credits=bundle_info["credits"],
        bundle_name=bundle_info["name"],
    )


@router.get("/payment-status/{payment_intent_id}", response_model=PaymentStatus)
async def get_payment_status(
    payment_intent_id: str, current_user: User = Depends(get_current_user)
):
    """Get the status of a payment intent"""

    payment_intent = StripeService.get_payment_intent(payment_intent_id)
    if not payment_intent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Payment intent not found"
        )

    # Verify the payment intent belongs to the current user
    if payment_intent.get("metadata", {}).get("user_id") != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Access denied"
        )

    credits = int(payment_intent.get("metadata", {}).get("credits", 0))

    return PaymentStatus(
        payment_intent_id=payment_intent["id"],
        status=payment_intent["status"],
        amount=payment_intent["amount"],
        currency=payment_intent["currency"],
        credits=credits,
    )


@router.get("/customer", response_model=CustomerInfo)
async def get_customer_info(current_user: User = Depends(get_current_user)):
    """Get customer information and payment methods"""

    if not current_user.stripe_customer_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="No customer record found"
        )

    payment_methods = StripeService.get_customer_payment_methods(
        current_user.stripe_customer_id
    )

    return CustomerInfo(
        customer_id=current_user.stripe_customer_id,
        email=current_user.email,
        payment_methods=payment_methods,
    )


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    """Handle Stripe webhook events"""

    payload = await request.body()
    sig_header = request.headers.get("stripe-signature")

    if not sig_header:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing stripe-signature header",
        )

    # Construct and verify the event
    event = StripeService.construct_webhook_event(payload, sig_header)
    if not event:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid webhook signature"
        )

    # Handle the event
    if event["type"] == "payment_intent.succeeded":
        payment_intent = event["data"]["object"]

        # Extract metadata
        user_id = payment_intent.get("metadata", {}).get("user_id")
        credits = payment_intent.get("metadata", {}).get("credits")
        bundle_name = payment_intent.get("metadata", {}).get("bundle_name")

        if user_id and credits:
            # Find the user
            user = db.query(User).filter(User.id == int(user_id)).first()
            if user:
                # Add credits to user account
                credit_service = CreditService(db)
                credit_service.add_credits(
                    user_id=user.id,
                    amount=int(credits),
                    description=f"Purchase: {bundle_name}",
                    transaction_type="purchase",
                )

                print(f"Added {credits} credits to user {user_id}")

    elif event["type"] == "payment_intent.payment_failed":
        payment_intent = event["data"]["object"]
        print(f"Payment failed for payment intent: {payment_intent['id']}")

    return {"status": "success"}
