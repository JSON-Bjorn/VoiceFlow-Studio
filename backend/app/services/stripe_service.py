import stripe
from typing import Dict, List, Optional
from ..core.config import settings
from ..models.user import User
from sqlalchemy.orm import Session

# Initialize Stripe
stripe.api_key = settings.stripe_api_key


class StripeService:
    """Service for handling Stripe payment operations"""

    @staticmethod
    def create_customer(user: User) -> Optional[str]:
        """Create a Stripe customer for a user"""
        try:
            customer = stripe.Customer.create(
                email=user.email,
                name=f"{user.email}",  # Using email as name for now
                metadata={"user_id": str(user.id), "app": "voiceflow_studio"},
            )
            return customer.id
        except stripe.error.StripeError as e:
            print(f"Error creating Stripe customer: {e}")
            return None

    @staticmethod
    def create_payment_intent(
        amount: int,  # Amount in cents
        currency: str = "usd",
        customer_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> Optional[Dict]:
        """Create a payment intent for processing payment"""
        try:
            intent_data = {
                "amount": amount,
                "currency": currency,
                "automatic_payment_methods": {"enabled": True},
            }

            if customer_id:
                intent_data["customer"] = customer_id

            if metadata:
                intent_data["metadata"] = metadata

            payment_intent = stripe.PaymentIntent.create(**intent_data)

            return {
                "client_secret": payment_intent.client_secret,
                "payment_intent_id": payment_intent.id,
                "amount": payment_intent.amount,
                "currency": payment_intent.currency,
                "status": payment_intent.status,
            }
        except stripe.error.StripeError as e:
            print(f"Error creating payment intent: {e}")
            return None

    @staticmethod
    def get_payment_intent(payment_intent_id: str) -> Optional[Dict]:
        """Retrieve a payment intent by ID"""
        try:
            payment_intent = stripe.PaymentIntent.retrieve(payment_intent_id)
            return {
                "id": payment_intent.id,
                "amount": payment_intent.amount,
                "currency": payment_intent.currency,
                "status": payment_intent.status,
                "customer": payment_intent.customer,
                "metadata": payment_intent.metadata,
            }
        except stripe.error.StripeError as e:
            print(f"Error retrieving payment intent: {e}")
            return None

    @staticmethod
    def create_product_and_prices() -> Dict[str, str]:
        """Create credit bundle products and prices in Stripe"""
        try:
            # Create the main product
            product = stripe.Product.create(
                name="VoiceFlow Studio Credits",
                description="Credits for generating AI podcasts",
                metadata={"app": "voiceflow_studio"},
            )

            # Define credit bundles
            bundles = [
                {"credits": 10, "price": 999, "name": "Starter Pack"},  # $9.99
                {"credits": 25, "price": 1999, "name": "Creator Pack"},  # $19.99
                {"credits": 50, "price": 3499, "name": "Pro Pack"},  # $34.99
                {"credits": 100, "price": 5999, "name": "Studio Pack"},  # $59.99
            ]

            price_ids = {}

            for bundle in bundles:
                price = stripe.Price.create(
                    product=product.id,
                    unit_amount=bundle["price"],
                    currency="usd",
                    metadata={
                        "credits": str(bundle["credits"]),
                        "bundle_name": bundle["name"],
                    },
                )
                price_ids[f"{bundle['credits']}_credits"] = price.id

            return price_ids

        except stripe.error.StripeError as e:
            print(f"Error creating products and prices: {e}")
            return {}

    @staticmethod
    def construct_webhook_event(payload: bytes, sig_header: str):
        """Construct and verify webhook event"""
        try:
            event = stripe.Webhook.construct_event(
                payload, sig_header, settings.stripe_webhook_secret
            )
            return event
        except ValueError as e:
            print(f"Invalid payload: {e}")
            return None
        except stripe.error.SignatureVerificationError as e:
            print(f"Invalid signature: {e}")
            return None

    @staticmethod
    def get_customer_payment_methods(customer_id: str) -> List[Dict]:
        """Get all payment methods for a customer"""
        try:
            payment_methods = stripe.PaymentMethod.list(
                customer=customer_id, type="card"
            )
            return [
                {
                    "id": pm.id,
                    "brand": pm.card.brand,
                    "last4": pm.card.last4,
                    "exp_month": pm.card.exp_month,
                    "exp_year": pm.card.exp_year,
                }
                for pm in payment_methods.data
            ]
        except stripe.error.StripeError as e:
            print(f"Error retrieving payment methods: {e}")
            return []
