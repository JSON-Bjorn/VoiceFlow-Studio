from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import settings

engine = create_engine(settings.database_url)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def init_db():
    """Initialize the database by creating all tables"""
    # Import all models to ensure they are registered with Base
    from app.models import user, podcast, credit_transaction

    # Create all tables
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
