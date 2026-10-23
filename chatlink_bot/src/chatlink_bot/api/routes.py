# chatlink_bot/src/chatlink_bot/api/routes.py
"""The admin API: salesmen and health. Nothing else.

Reached only through an SSH tunnel (compose binds it to 127.0.0.1):
    ssh -L 8000:localhost:8000 <server>   ->   curl localhost:8000/api/healthz
"""
from typing import List

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from ..database import get_db, pg_engine
from ..handlers import login_user, logout_user
from ..models import User, UserCreate, UserResponse
from ..transport.email import email_transport

router = APIRouter()


class GmailPassword(BaseModel):
    password: str


async def _user(db: AsyncSession, user_id: int) -> User:
    user = (await db.execute(select(User).where(User.id == user_id))).scalars().first()
    if not user:
        raise HTTPException(404, "User not found")
    return user


@router.get("/healthz")
async def healthz():
    """200 when Postgres answers; the deploy script and the loop's heartbeat rely on it."""
    try:
        async with pg_engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception as e:
        raise HTTPException(503, f"postgres: {e}")
    return {"status": "ok"}


@router.post("/users", response_model=UserResponse, status_code=201)
async def create_user(data: UserCreate, db: AsyncSession = Depends(get_db)):
    if (await db.execute(select(User).where(User.email == data.email))).scalars().first():
        raise HTTPException(400, "Email already registered")
    user = User(**data.model_dump())
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user


@router.get("/users", response_model=List[UserResponse])
async def list_users(db: AsyncSession = Depends(get_db)):
    return (await db.execute(select(User))).scalars().all()


@router.post("/users/{user_id}/login")
async def api_login(user_id: int, db: AsyncSession = Depends(get_db)):
    out = await login_user(await _user(db, user_id))
    if not out.get("success"):
        raise HTTPException(400, out.get("error", "login_failed"))
    return out


@router.post("/users/{user_id}/logout")
async def api_logout(user_id: int, db: AsyncSession = Depends(get_db)):
    return await logout_user(await _user(db, user_id))


@router.post("/users/{user_id}/gmail-password")
async def api_gmail_password(user_id: int, req: GmailPassword, db: AsyncSession = Depends(get_db)):
    user = await _user(db, user_id)
    if not email_transport.set_app_password(user.email, req.password):
        raise HTTPException(400, "could not store the password")
    return {"success": True}


@router.delete("/users/{user_id}")
async def delete_user(user_id: int, db: AsyncSession = Depends(get_db)):
    user = await _user(db, user_id)
    await logout_user(user)
    await db.delete(user)
    await db.commit()
    return {"success": True}
