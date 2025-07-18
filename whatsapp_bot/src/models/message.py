from sqlalchemy import ForeignKey, DateTime
from sqlalchemy import Column, Integer, String
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from typing import Optional

from src.models import Base_sqlite


class Message(Base_sqlite):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True)
    client_id = Column(Integer, nullable=False)
    client_phone = Column(String, nullable=False)
    user_id = Column(
        Integer, ForeignKey("users.id", onupdate="SET NULL"), nullable=True
    )
    user_phone = Column(String, nullable=False)
    direction = Column(String, nullable=False)  # 'sended' o 'received'
    type = Column(String, nullable=False)  # 'text', 'image', etc.
    content = Column(String)
    timestamp = Column(DateTime, nullable=False)

    @staticmethod
    def create(
        session: Session,
        client_id: int,
        client_phone: str,
        direction: str,
        type_: str,
        user_id: int,
        user_phone: str,
        content: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> "Message":
        msg = Message(
            client_id=client_id,
            client_phone=client_phone,
            user_id=user_id,
            user_phone=user_phone,
            direction=direction,
            type=type_,
            content=content,
            timestamp=timestamp or datetime.now(timezone.utc),
        )
        session.add(msg)
        session.commit()
        session.refresh(msg)
        return msg
