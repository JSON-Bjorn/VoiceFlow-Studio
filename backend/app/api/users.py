from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.auth import get_current_user
from ..schemas.user import UserResponse, UserUpdateEmail, UserUpdatePassword
from ..models.user import User
from ..services.user_service import UserService

router = APIRouter(prefix="/users", tags=["users"])


def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Get current active user"""
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user profile"""
    return current_user


@router.put("/me/email", response_model=UserResponse)
async def update_user_email(
    email_data: UserUpdateEmail,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update current user's email"""
    user_service = UserService(db)

    try:
        updated_user = user_service.update_user_email(current_user.id, email_data.email)
        return updated_user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.put("/me/password", response_model=UserResponse)
async def update_user_password(
    password_data: UserUpdatePassword,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    """Update current user's password"""
    user_service = UserService(db)

    try:
        updated_user = user_service.update_user_password(
            current_user.id, password_data.current_password, password_data.new_password
        )
        return updated_user
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID (public endpoint)"""
    user_service = UserService(db)
    user = user_service.get_user_by_id(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
