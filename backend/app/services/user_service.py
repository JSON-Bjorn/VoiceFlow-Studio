from sqlalchemy.orm import Session
from typing import Optional
from ..models.user import User
from ..models.credit_transaction import CreditTransaction, TransactionType
from ..schemas.user import UserCreate
from ..core.auth import get_password_hash, verify_password


class UserService:
    def __init__(self, db: Session):
        self.db = db

    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        return self.db.query(User).filter(User.email == email).first()

    def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID"""
        return self.db.query(User).filter(User.id == user_id).first()

    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        # Check if user already exists
        existing_user = self.get_user_by_email(user_data.email)
        if existing_user:
            raise ValueError("User with this email already exists")

        # Hash password and create user
        hashed_password = get_password_hash(user_data.password)
        db_user = User(
            email=user_data.email,
            hashed_password=hashed_password,
            credits=1,  # Give 1 free credit on signup
        )

        self.db.add(db_user)
        self.db.commit()
        self.db.refresh(db_user)

        # Create bonus credit transaction record
        bonus_transaction = CreditTransaction(
            user_id=db_user.id,
            amount=1,
            transaction_type=TransactionType.BONUS,
            description="Welcome bonus - 1 free credit",
            reference_id="signup_bonus",
        )
        self.db.add(bonus_transaction)
        self.db.commit()

        return db_user

    def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = self.get_user_by_email(email)
        if not user:
            return None
        if not verify_password(password, user.hashed_password):
            return None
        return user

    def update_user_credits(self, user_id: int, credits: int) -> Optional[User]:
        """Update user credits"""
        user = self.get_user_by_id(user_id)
        if user:
            user.credits = credits
            self.db.commit()
            self.db.refresh(user)
        return user

    def update_user_email(self, user_id: int, new_email: str) -> User:
        """Update user email"""
        # Check if email is already taken by another user
        existing_user = self.get_user_by_email(new_email)
        if existing_user and existing_user.id != user_id:
            raise ValueError("Email is already taken by another user")

        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        user.email = new_email
        self.db.commit()
        self.db.refresh(user)
        return user

    def update_user_password(
        self, user_id: int, current_password: str, new_password: str
    ) -> User:
        """Update user password"""
        user = self.get_user_by_id(user_id)
        if not user:
            raise ValueError("User not found")

        # Verify current password
        if not verify_password(current_password, user.hashed_password):
            raise ValueError("Current password is incorrect")

        # Hash and update new password
        user.hashed_password = get_password_hash(new_password)
        self.db.commit()
        self.db.refresh(user)
        return user
