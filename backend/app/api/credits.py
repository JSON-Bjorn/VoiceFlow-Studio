from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List

from ..core.database import get_db
from ..core.auth import get_current_user
from ..schemas.credit import (
    CreditTransactionResponse,
    CreditSummary,
    CreditUsage,
    CreditPurchase,
)
from ..models.user import User
from ..services.credit_service import CreditService

router = APIRouter(prefix="/credits", tags=["credits"])


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@router.get("/summary", response_model=CreditSummary)
async def get_credit_summary(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get comprehensive credit summary for current user"""
    credit_service = CreditService(db)

    try:
        return credit_service.get_credit_summary(current_user.id)
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/transactions", response_model=List[CreditTransactionResponse])
async def get_credit_transactions(
    limit: int = 10,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Get credit transaction history for current user"""
    credit_service = CreditService(db)

    return credit_service.get_user_transactions(current_user.id, limit=limit)


@router.post("/use", response_model=CreditTransactionResponse)
async def use_credits(
    usage_data: CreditUsage,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Use credits for a service (e.g., podcast generation)"""
    credit_service = CreditService(db)

    try:
        transaction = credit_service.use_credits(
            user_id=current_user.id,
            amount=usage_data.amount,
            description=usage_data.description,
        )
        return transaction
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/purchase", response_model=CreditTransactionResponse)
async def purchase_credits(
    purchase_data: CreditPurchase,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Purchase credits (for testing - in production this would be handled by payment webhook)"""
    credit_service = CreditService(db)

    try:
        transaction = credit_service.purchase_credits(
            user_id=current_user.id,
            amount=purchase_data.amount,
            payment_reference=purchase_data.payment_reference,
        )
        return transaction
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/can-afford/{amount}")
async def can_afford_credits(
    amount: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Check if user can afford specified amount of credits"""
    credit_service = CreditService(db)

    return {
        "can_afford": credit_service.can_afford(current_user.id, amount),
        "current_balance": current_user.credits,
        "required": amount,
    }
