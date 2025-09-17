# ===================== app.py =====================
import asyncio
import os
import random
import string
import io
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Tuple, Optional, Set, Dict


from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
Message, CallbackQuery, InlineKeyboardMarkup,
)
from aiogram.types.input_file import BufferedInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder


from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String as SAString, DateTime, ForeignKey, Integer, Boolean, UniqueConstraint, select, func, JSON


# ---------- Config ----------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./santa.db")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL") # e.g. https://your-app.onrender.com
PORT = int(os.environ.get("PORT", "8000"))
REMINDER_HOUR = int(os.environ.get("REMINDER_HOUR", "10"))
MAX_ROOMS_PER_OWNER = int(os.environ.get("MAX_ROOMS_PER_OWNER", "5")) # корпоративный лимит на владельца
MAX_PARTICIPANTS_PER_ROOM = int(os.environ.get("MAX_PARTICIPANTS_PER_ROOM", "200"))
MAX_HINTS_PER_DAY = int(os.environ.get("MAX_HINTS_PER_DAY", "3"))


if not BOT_TOKEN:
raise RuntimeError("BOT_TOKEN is required")


# ---------- DB ----------
class Base(DeclarativeBase):
pass


class Room(Base):
__tablename__ = "rooms"
id: Mapped[int] = mapped_column(primary_key=True)
code: Mapped[str] = mapped_column(SAString(10), unique=True, index=True)
owner_id: Mapped[int]
title: Mapped[str] = mapped_column(default="Secret Santa")
budget: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
deadline_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
# Challenge rules (для веселья)
rule_letter: Mapped[Optional[str]] = mapped_column(SAString(1), nullable=True)
rule_amount_exact: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
rule_amount_max: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
# Corporate mode
corporate: Mapped[bool] = mapped_column(Boolean, default=False)
org_name: Mapped[Optional[str]] = mapped_column(SAString(128), nullable=True)


drawn: Mapped[bool] = mapped_column(Boolean, default=False)
created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


class Participant(Base):
__tablename__ = "participants"
id: Mapped[int] = mapped_column(primary_key=True)
room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), index=True)
user_id: Mapped[int] = mapped_column(index=True)
name: Mapped[str] = mapped_column(SAString(64))
wishes: Mapped[str] = mapped_column(SAString(512), default="")
joined_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)


__table_args__ = (UniqueConstraint("room_id", "user_id", name="uq_room_user"),)


class Pair(Base):
__tablename__ = "pairs"
id: Mapped[int] = mapped_column(primary_key=True)
room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), index=True)
giver_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
receiver_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)


class ForbiddenPair(Base):
__tablename__ = "forbidden_pairs"
id: Mapped[int] = mapped_column(primary_key=True)
room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), index=True)
giver_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
receiver_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
__table_args__ = (UniqueConstraint("room_id", "giver_id", "receiver_id", name="uq_forbidden"),)


class Hint(Base):
__tablename__ = "hints"
id: Mapped[int] = mapped_column(primary_key=True)
room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), index=True)
pass
