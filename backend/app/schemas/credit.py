from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from ..models.credit_transaction import TransactionType


class CreditTransactionBase(BaseModel):
    amount: int
    transaction_type: TransactionType
    description: Optional[str] = None
    reference_id: Optional[str] = None


class CreditTransactionCreate(CreditTransactionBase):
    user_id: int


class CreditTransactionResponse(CreditTransactionBase):
    id: int
    user_id: int
    created_at: datetime

    class Config:
        from_attributes = True


class CreditUsage(BaseModel):
    description: str
    amount: int = 1  # Default to 1 credit


class CreditPurchase(BaseModel):
    amount: int
    payment_reference: Optional[str] = None


class CreditSummary(BaseModel):
    current_balance: int
    total_purchased: int
    total_used: int
    total_bonus: int
    recent_transactions: List[CreditTransactionResponse]
