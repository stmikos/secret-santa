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
from sqlalchemy import String as SAString, DateTime, ForeignKey, Integer, Boolean, UniqueConstraint, select


# ---------- Config ----------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./santa.db")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL") # e.g. https://your-app.onrender.com
PORT = int(os.environ.get("PORT", "8000"))
REMINDER_HOUR = int(os.environ.get("REMINDER_HOUR", "10")) # hour for reminders (local approx)


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
drawn: Mapped[bool] = mapped_column(Boolean, default=False)
created_at: Mapped[datetime] = mapped_column(default=datetime.utcnow)
pass
