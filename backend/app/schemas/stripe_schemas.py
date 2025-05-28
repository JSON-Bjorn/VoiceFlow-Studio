from pydantic import BaseModel
from typing import Optional, Dict, List
from enum import Enum


class CreditBundle(str, Enum):
    """Available credit bundles"""

    STARTER = "10_credits"  # 10 credits - $9.99
    CREATOR = "25_credits"  # 25 credits - $19.99
    PRO = "50_credits"  # 50 credits - $34.99
    STUDIO = "100_credits"  # 100 credits - $59.99


class PaymentIntentCreate(BaseModel):
    """Request to create a payment intent"""

    bundle: CreditBundle


class PaymentIntentResponse(BaseModel):
    """Response from creating a payment intent"""

    client_secret: str
    payment_intent_id: str
    amount: int
    currency: str
    status: str
    credits: int
    bundle_name: str


class PaymentStatus(BaseModel):
    """Payment status response"""

    payment_intent_id: str
    status: str
    amount: int
    currency: str
    credits: int


class WebhookEvent(BaseModel):
    """Stripe webhook event"""

    type: str
    data: Dict


class CreditBundleInfo(BaseModel):
    """Information about a credit bundle"""

    id: str
    name: str
    credits: int
    price: int  # in cents
    price_formatted: str  # e.g., "$9.99"
    description: str
    popular: bool = False


class PaymentMethodInfo(BaseModel):
    """Payment method information"""

    id: str
    brand: str
    last4: str
    exp_month: int
    exp_year: int


class CustomerInfo(BaseModel):
    """Stripe customer information"""

    customer_id: str
    email: str
    payment_methods: List[PaymentMethodInfo]
