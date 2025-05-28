from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from typing import List, Optional
from ..models.user import User
from ..models.credit_transaction import CreditTransaction, TransactionType
from ..schemas.credit import (
    CreditTransactionCreate,
    CreditUsage,
    CreditPurchase,
    CreditSummary,
)


class CreditService:
    def __init__(self, db: Session):
        self.db = db

    def add_credits(
        self,
        user_id: int,
        amount: int,
        transaction_type: TransactionType,
        description: str = None,
        reference_id: str = None,
    ) -> CreditTransaction:
        """Add credits to user account and create transaction record"""
        # Get user
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        # Update user credits
        user.credits += amount

        # Create transaction record
        transaction = CreditTransaction(
            user_id=user_id,
            amount=amount,
            transaction_type=transaction_type,
            description=description,
            reference_id=reference_id,
        )

        self.db.add(transaction)
        self.db.commit()
        self.db.refresh(transaction)
        self.db.refresh(user)

        return transaction

    def use_credits(
        self,
        user_id: int,
        amount: int = 1,
        description: str = "Podcast generation",
        reference_id: str = None,
    ) -> CreditTransaction:
        """Use credits from user account and create transaction record"""
        # Get user
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        # Check if user has enough credits
        if user.credits < amount:
            raise ValueError("Insufficient credits")

        # Update user credits
        user.credits -= amount

        # Create transaction record (negative amount for usage)
        transaction = CreditTransaction(
            user_id=user_id,
            amount=-amount,
            transaction_type=TransactionType.USAGE,
            description=description,
            reference_id=reference_id,
        )

        self.db.add(transaction)
        self.db.commit()
        self.db.refresh(transaction)
        self.db.refresh(user)

        return transaction

    def purchase_credits(
        self, user_id: int, amount: int, payment_reference: str = None
    ) -> CreditTransaction:
        """Purchase credits (wrapper for add_credits with purchase type)"""
        return self.add_credits(
            user_id=user_id,
            amount=amount,
            transaction_type=TransactionType.PURCHASE,
            description=f"Purchased {amount} credits",
            reference_id=payment_reference,
        )

    def add_bonus_credits(
        self, user_id: int, amount: int, description: str = "Bonus credits"
    ) -> CreditTransaction:
        """Add bonus credits (wrapper for add_credits with bonus type)"""
        return self.add_credits(
            user_id=user_id,
            amount=amount,
            transaction_type=TransactionType.BONUS,
            description=description,
        )

    def get_user_transactions(
        self, user_id: int, limit: int = 10
    ) -> List[CreditTransaction]:
        """Get user's credit transaction history"""
        return (
            self.db.query(CreditTransaction)
            .filter(CreditTransaction.user_id == user_id)
            .order_by(desc(CreditTransaction.created_at))
            .limit(limit)
            .all()
        )

    def get_credit_summary(self, user_id: int) -> CreditSummary:
        """Get comprehensive credit summary for user"""
        user = self.db.query(User).filter(User.id == user_id).first()
        if not user:
            raise ValueError("User not found")

        # Calculate totals by transaction type
        totals = (
            self.db.query(
                CreditTransaction.transaction_type,
                func.sum(CreditTransaction.amount).label("total"),
            )
            .filter(CreditTransaction.user_id == user_id)
            .group_by(CreditTransaction.transaction_type)
            .all()
        )

        total_purchased = 0
        total_used = 0
        total_bonus = 0

        for transaction_type, total in totals:
            if transaction_type == TransactionType.PURCHASE:
                total_purchased = total or 0
            elif transaction_type == TransactionType.USAGE:
                total_used = abs(total or 0)  # Usage is stored as negative
            elif transaction_type == TransactionType.BONUS:
                total_bonus = total or 0

        # Get recent transactions
        recent_transactions = self.get_user_transactions(user_id, limit=5)

        return CreditSummary(
            current_balance=user.credits,
            total_purchased=total_purchased,
            total_used=total_used,
            total_bonus=total_bonus,
            recent_transactions=recent_transactions,
        )

    def can_afford(self, user_id: int, amount: int = 1) -> bool:
        """Check if user can afford the specified amount of credits"""
        user = self.db.query(User).filter(User.id == user_id).first()
        return user and user.credits >= amount
