from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import Session
from typing import Optional, List

from src.models import Base_sqlite


class User(Base_sqlite):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    phone = Column(String, nullable=False, unique=True)
    email = Column(String, nullable=False, unique=True)
    name = Column(String)
    role = Column(String, nullable=False)  # 'user' o 'admin'

    @staticmethod
    def get_by_phone(session: Session, phone: str) -> Optional["User"]:
        return session.query(User).filter(User.phone == phone).first()

    @staticmethod
    def user_exists(session: Session, phone: str) -> bool:
        return session.query(User.phone).filter(User.phone == phone).first() is not None

    @staticmethod
    def get_admins(session: Session) -> List["User"]:
        return session.query(User).filter(User.role == "admin").all()
