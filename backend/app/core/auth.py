from datetime import datetime, timedelta
from typing import Optional, TYPE_CHECKING
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from .config import settings
from .database import get_db

if TYPE_CHECKING:
    from ..models.user import User

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Security scheme
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against its hash"""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Create a JWT access token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.access_token_expire_minutes
        )

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, settings.secret_key, algorithm=settings.algorithm
    )
    return encoded_jwt


def verify_token(token: str) -> Optional[dict]:
    """Verify and decode a JWT token"""
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
        return payload
    except JWTError:
        return None


async def get_current_user_from_token(
    token: str, db: Session = None
) -> Optional["User"]:
    """
    Get user from JWT token (for WebSocket connections)

    Args:
        token: JWT token string
        db: Optional database session (will create new one if not provided)

    Returns:
        User object if token is valid, None otherwise
    """
    from ..models.user import User  # Import here to avoid circular imports

    try:
        payload = verify_token(token)
        if payload is None:
            return None

        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            return None

        # Convert user ID from string to int
        user_id = int(user_id_str)

        # Get database session if not provided
        if db is None:
            db_gen = get_db()
            db = next(db_gen)
            try:
                user = db.query(User).filter(User.id == user_id).first()
                return user
            finally:
                # Close the database session
                try:
                    next(db_gen)
                except StopIteration:
                    pass
        else:
            user = db.query(User).filter(User.id == user_id).first()
            return user

    except (JWTError, ValueError, TypeError, Exception):
        return None


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_db),
):
    """Get the current authenticated user"""
    from ..models.user import User  # Import here to avoid circular imports

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Extract token from credentials
        token = credentials.credentials
        payload = verify_token(token)

        if payload is None:
            raise credentials_exception

        user_id_str: str = payload.get("sub")
        if user_id_str is None:
            raise credentials_exception

        # Convert user ID from string to int
        user_id = int(user_id_str)

    except (JWTError, ValueError, TypeError):
        raise credentials_exception

    # Get user from database
    user = db.query(User).filter(User.id == user_id).first()
    if user is None:
        raise credentials_exception

    return user
