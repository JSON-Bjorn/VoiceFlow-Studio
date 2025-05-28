from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..core.dependencies import get_current_active_user
from ..schemas.user import UserResponse
from ..models.user import User

router = APIRouter(prefix="/users", tags=["users"])


@router.get("/me", response_model=UserResponse)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """Get current user profile"""
    return current_user


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, db: Session = Depends(get_db)):
    """Get user by ID (public endpoint)"""
    from ..services.user_service import UserService

    user_service = UserService(db)
    user = user_service.get_user_by_id(user_id)
    if not user:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail="User not found")
    return user
