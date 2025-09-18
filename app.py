# app.py — Secret Santa Bot (минимальный, рабочий, без воды)
# Python 3.11+ / Aiogram 3.7+
import os
import asyncio
import random
import string
import hashlib
import urllib.parse
from typing import Optional, List, Tuple, Dict, Set
from datetime import datetime, timedelta, UTC

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart, StateFilter
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.context import FSMContext
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from sqlalchemy import (
    select, func, Integer, Boolean, ForeignKey, String as SAString, DateTime, UniqueConstraint
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.exc import IntegrityError

# ------------------------------------------------------------
# ENV
# ------------------------------------------------------------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required")

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./santa.db")

# Авто-апгрейд URL на psycopg3 (чтобы не ловить psycopg2)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+psycopg://", 1)
elif DATABASE_URL.startswith("postgresql://") and "+psycopg" not in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # если задан — будет режим webhook, иначе polling
PORT = int(os.environ.get("PORT", "10000"))

MAX_ROOMS_PER_OWNER = int(os.environ.get("MAX_ROOMS_PER_OWNER", "5"))
MAX_PARTICIPANTS_PER_ROOM = int(os.environ.get("MAX_PARTICIPANTS_PER_ROOM", "200"))
MAX_HINTS_PER_DAY = int(os.environ.get("MAX_HINTS_PER_DAY", "3"))

# Аффилиаты: {"wb":"https://www.wildberries.ru/catalog/0/search.aspx?search={q}", ...}
try:
    AFF_TEMPLATES: Dict[str, str] = dict(**( __import__("json").loads(os.environ.get("AFFILIATES_JSON","{}")) ))
except Exception:
    AFF_TEMPLATES = {}
HUMAN_NAMES = {"wb": "Wildberries", "ozon": "Ozon", "ym": "Яндекс.Маркет"}
AFF_PRIMARY = next(iter(AFF_TEMPLATES.keys()), None)

# ------------------------------------------------------------
# DB
# ------------------------------------------------------------
class Base(DeclarativeBase): pass

class Room(Base):
    __tablename__ = "rooms"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(SAString(10), unique=True, index=True)
    owner_id: Mapped[int]
    title: Mapped[str] = mapped_column(SAString(64), default="Secret Santa")
    budget: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    deadline_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    rule_letter: Mapped[Optional[str]] = mapped_column(SAString(1), nullable=True)
    rule_amount_exact: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rule_amount_max: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    corporate: Mapped[bool] = mapped_column(Boolean, default=False)
    drawn: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

class Participant(Base):
    __tablename__ = "participants"
    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id", ondelete="CASCADE"), index=True)
    user_id: Mapped[int] = mapped_column(index=True)
    name: Mapped[str] = mapped_column(SAString(64))
    wishes: Mapped[str] = mapped_column(SAString(512), default="")
    joined_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
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
    sender_participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    receiver_participant_id: Mapped[int] = mapped_column(ForeignKey("participants.id", ondelete="CASCADE"), index=True)
    text: Mapped[str] = mapped_column(SAString(512))
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

class RuntimeLock(Base):
    __tablename__ = "runtime_lock"
    id: Mapped[int] = mapped_column(primary_key=True)
    bot_token_hash: Mapped[str] = mapped_column(SAString(64), unique=True, index=True)
    started_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

CONNECT_ARGS = {}
if DATABASE_URL.startswith("postgresql+psycopg://"):
    CONNECT_ARGS["prepare_threshold"] = 0  # защита от DuplicatePreparedStatement

engine = create_async_engine(DATABASE_URL, echo=False, future=True, connect_args=CONNECT_ARGS)
Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def gen_code(n: int = 6) -> str:
    import secrets
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))

def mk_rules(room: Room) -> str:
    rows = ["Правила комнаты:"]
    if room.rule_letter: rows.append(f"• Подарок на букву: <b>{room.rule_letter}</b>")
    if room.rule_amount_exact: rows.append(f"• Сумма ровно: <b>{room.rule_amount_exact}₽</b>")
    if room.rule_amount_max: rows.append(f"• Сумма максимум: <b>{room.rule_amount_max}₽</b>")
    rows.append(f"Бюджет: <b>{room.budget or '—'}</b>")
    rows.append(f"Дедлайн: <b>{room.deadline_at.date() if room.deadline_at else '—'}</b>")
    return "\n".join(rows)

def wishes_to_query(wishes: str, budget_max: Optional[int], letter: Optional[str]) -> str:
    parts = []
    if wishes: parts.append(wishes)
    if budget_max: parts.append(f"до {budget_max} руб")
    if letter: parts.append(f"на букву {letter}")
    return ", ".join(parts) or "подарок сюрприз"

def mk_aff_url(marketplace: str, query: str) -> Optional[str]:
    tpl = AFF_TEMPLATES.get(marketplace)
    if not tpl: return None
    return tpl.replace("{q}", urllib.parse.quote_plus(query.strip()))

def draw_pairs(ids: List[int], forbidden: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    n = len(ids)
    if n < 2: raise ValueError("Need >=2 participants")
    receivers = ids[:]
    random.shuffle(receivers)
    assigned: Dict[int, int] = {}
    def backtrack(i: int) -> bool:
        if i == n: return True
        giver = ids[i]
        random.shuffle(receivers)
        for r in receivers:
            if r == giver: continue
            if (giver, r) in forbidden: continue
            if r in assigned.values(): continue
            assigned[giver] = r
            if backtrack(i + 1): return True
            assigned.pop(giver, None)
        return False
    if not backtrack(0):
        rot = list(zip(ids, ids[1:] + ids[:1]))
        if any(g == r or (g, r) in forbidden for g, r in rot):
            raise RuntimeError("Constraints too strict")
        return rot
    return list(assigned.items())

async def get_user_active_room(user_id: int) -> Optional[Room]:
    async with Session() as s:
        p = (await s.execute(
            select(Participant).where(Participant.user_id == user_id).order_by(Participant.joined_at.desc()).limit(1)
        )).scalar_one_or_none()
        if not p: return None
        return (await s.execute(select(Room).where(Room.id == p.room_id))).scalar_one_or_none()

# ------------------------------------------------------------
# Runtime lock (polling single-instance)
# ------------------------------------------------------------
def _aware(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None: return None
    return dt if dt.tzinfo is not None else dt.replace(tzinfo=UTC)

async def acquire_runtime_lock(ttl_seconds: int = 600) -> bool:
    h = hashlib.sha256(BOT_TOKEN.encode()).hexdigest()
    now = datetime.now(UTC)
    ttl_ago = now - timedelta(seconds=ttl_seconds)
    async with Session() as s:
        existing = (await s.execute(select(RuntimeLock).where(RuntimeLock.bot_token_hash == h))).scalar_one_or_none()
        if existing:
            started = _aware(existing.started_at)
            if started and started < ttl_ago:
                await s.delete(existing); await s.commit()
            else:
                return False
        s.add(RuntimeLock(bot_token_hash=h, started_at=now))
        try:
            await s.commit(); return True
        except IntegrityError:
            await s.rollback(); return False

async def release_runtime_lock():
    h = hashlib.sha256(BOT_TOKEN.encode()).hexdigest()
    async with Session() as s:
        row = (await s.execute(select(RuntimeLock).where(RuntimeLock.bot_token_hash == h))).scalar_one_or_none()
        if row:
            await s.delete(row); await s.commit()

# ------------------------------------------------------------
# Bot
# ------------------------------------------------------------
bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

class Join(StatesGroup):
    name = State()
    wishes = State()

# Reply keyboards (простые)
def kb_root(in_room: bool) -> ReplyKeyboardMarkup:
    if not in_room:
        return ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="➕ Создать"), KeyboardButton(text="🔗 Присоединиться")],
                [KeyboardButton(text="👤 Профиль"), KeyboardButton(text="ℹ️ Правила")],
            ],
            resize_keyboard=True
        )
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🏠 Меню"), KeyboardButton(text="ℹ️ Правила")],
            [KeyboardButton(text="📝 Хотелки"), KeyboardButton(text="📨 Получатель")],
            [KeyboardButton(text="🎁 Идеи"), KeyboardButton(text="🛒 Купить")],
            [KeyboardButton(text="🚪 Выйти из комнаты")],
        ],
        resize_keyboard=True
    )

# Inline keyboards
def main_kb(code: Optional[str], is_owner: bool) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    if not code:
        b.button(text="➕ Создать комнату", callback_data="room_new")
        b.button(text="🔗 Присоединиться", callback_data="room_join")
    else:
        b.button(text="👥 Участники", callback_data=f"room_participants:{code}")
        b.button(text="✏️ Мои хотелки", callback_data=f"me_edit:{code}")
        b.button(text="📨 Получатель", callback_data=f"me_target:{code}")
        b.button(text="🕵️ Подсказка", callback_data=f"hint_send:{code}")
        b.button(text="🎁 Идеи", callback_data=f"ideas:{code}")
        b.button(text="🛒 Купить", callback_data=f"buy:{code}")
        if is_owner:
            b.button(text="🎲 Жеребьёвка", callback_data=f"room_draw:{code}")
            b.button(text="⚙️ Настройки", callback_data=f"room_settings:{code}")
    b.button(text="🏠 В главное меню", callback_data="to_main")
    b.adjust(1)
    return b.as_markup()

async def send_single(m: Message | CallbackQuery, text: str, kb: InlineKeyboardMarkup | ReplyKeyboardMarkup | None = None):
    if isinstance(m, CallbackQuery):
        sent = await m.message.answer(text, reply_markup=kb)
        with contextlib.suppress(Exception): await m.answer()
        return sent
    else:
        return await m.answer(text, reply_markup=kb)

import contextlib

async def show_main(m: Message | CallbackQuery):
    room = await get_user_active_room(m.from_user.id)
    if isinstance(m, CallbackQuery):
        await send_single(m, "Главное меню", main_kb(room.code if room else None, bool(room and room.owner_id == m.from_user.id)))
    else:
        await send_single(m, "Главное меню", kb_root(bool(room)))

@dp.message(StateFilter("*"), F.text.in_({"🏠 Меню", "⬅️ Назад", "Отмена", "/menu"}))
async def any_to_menu(m: Message, state: FSMContext):
    await state.clear()
    await show_main(m)

@dp.callback_query(StateFilter("*"), F.data == "to_main")
async def cb_to_main(cq: CallbackQuery, state: FSMContext):
    await state.clear()
    await show_main(cq)

# Start
@dp.message(CommandStart())
async def start(m: Message):
    payload = m.text.split(maxsplit=1)[1] if len(m.text.split()) > 1 else ""
    if payload.startswith("room_"):
        await enter_room_menu(m, payload.removeprefix("room_")); return
    await show_main(m)

# Create room
@dp.message(F.text == "➕ Создать")
async def create_room_btn(m: Message):
    async with Session() as s:
        cnt = (await s.execute(select(func.count()).select_from(Room).where(Room.owner_id == m.from_user.id))).scalar()
        if cnt >= MAX_ROOMS_PER_OWNER:
            await m.answer("Лимит комнат исчерпан"); return
        code = gen_code()
        s.add(Room(code=code, owner_id=m.from_user.id))
        await s.commit()
    me = await bot.get_me()
    link = f"https://t.me/{me.username}?start=room_{code}"
    await m.answer(f"Комната создана: <code>{code}</code>\nПриглашение: {link}", reply_markup=kb_root(True))
    await enter_room_menu(m, code)

# Join flow
@dp.message(F.text == "🔗 Присоединиться")
async def join_btn(m: Message, state: FSMContext):
    await state.update_data(wait_code=True)
    await m.answer("Введи код комнаты (например: ABC123)")

@dp.message(F.text.regexp(r"^[A-Za-z0-9]{4,10}$"))
async def join_code(m: Message, state: FSMContext):
    if not (await state.get_data()).get("wait_code"):
        return
    code = m.text.strip().upper()
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room:
            await m.answer("Комната не найдена"); return
    await state.clear()
    await state.update_data(room_code=code)
    await state.set_state(Join.name)
    await m.answer("Как тебя звать для списка?")

@dp.message(Join.name)
async def join_name(m: Message, state: FSMContext):
    await state.update_data(name=(m.text or "").strip()[:64])
    await state.set_state(Join.wishes)
    await m.answer("Напиши хотелки/табу:")

@dp.message(Join.wishes)
async def join_wishes(m: Message, state: FSMContext):
    data = await state.get_data()
    code = data["room_code"]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        count = (await s.execute(select(func.count()).select_from(Participant).where(Participant.room_id == room.id))).scalar()
        if count >= MAX_PARTICIPANTS_PER_ROOM:
            await state.clear(); await m.answer("Лимит участников достигнут"); return
        me = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id))).scalar_one_or_none()
        if me:
            me.name = data["name"]; me.wishes = (m.text or "")[:512]
        else:
            s.add(Participant(room_id=room.id, user_id=m.from_user.id, name=data["name"], wishes=(m.text or "")[:512]))
        await s.commit()
    await state.clear()
    await m.answer("Записал. Но свечку всё равно подарят 😏", reply_markup=kb_root(True))
    await enter_room_menu(m, code)

# Room open / participants
async def enter_room_menu(msg: Message | CallbackQuery, code: str):
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room:
            await send_single(msg, "Комната не найдена", main_kb(None, False)); return
        user_id = msg.from_user.id
        part = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == user_id))).scalar_one_or_none()
    if not part:
        kb = InlineKeyboardBuilder()
        kb.button(text="✅ Присоединиться", callback_data=f"join:{code}")
        kb.button(text="↩️ Назад", callback_data="to_main")
        await send_single(msg,
            f"Комната <b>{room.title}</b> (<code>{room.code}</code>)\nБюджет: {room.budget or '—'} | Дедлайн: {room.deadline_at.date() if room.deadline_at else '—'}",
            kb.as_markup())
        return
    await send_single(msg, f"Комната <b>{room.title}</b> (<code>{room.code}</code>)", main_kb(code, room.owner_id==msg.from_user.id))

@dp.callback_query(F.data == "room_new")
async def cb_room_new(cq: CallbackQuery):
    async with Session() as s:
        cnt = (await s.execute(select(func.count()).select_from(Room).where(Room.owner_id == cq.from_user.id))).scalar()
        if cnt >= MAX_ROOMS_PER_OWNER:
            await cq.answer("Лимит комнат исчерпан", show_alert=True); return
        code = gen_code()
        s.add(Room(code=code, owner_id=cq.from_user.id)); await s.commit()
    me = await bot.get_me(); link = f"https://t.me/{me.username}?start=room_{code}"
    kb = InlineKeyboardBuilder()
    kb.button(text="🔗 Приглашение", url=link)
    kb.button(text="➡️ В комнату", callback_data=f"room_open:{code}")
    await send_single(cq, f"Комната создана: <code>{code}</code>", kb.as_markup())

@dp.callback_query(F.data.startswith("room_open:"))
async def cb_room_open(cq: CallbackQuery):
    await enter_room_menu(cq, cq.data.split(":",1)[1])

@dp.callback_query(F.data == "room_join")
async def cb_room_join(cq: CallbackQuery):
    await cq.message.answer("Отправь код комнаты (ABC123)"); await cq.answer()

@dp.callback_query(F.data.startswith("join:"))
async def cb_join(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":",1)[1]
    await state.update_data(room_code=code); await state.set_state(Join.name)
    await send_single(cq, "Как тебя звать для списка?", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.callback_query(F.data.startswith("room_participants:"))
async def cb_participants(cq: CallbackQuery):
    code = cq.data.split(":",1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await cq.answer("Нет комнаты", show_alert=True); return
        rows = (await s.execute(select(Participant).where(Participant.room_id == room.id).order_by(Participant.joined_at))).scalars().all()
    names = "\n".join(f"{i+1}. {p.name}" for i,p in enumerate(rows)) or "пока пусто"
    await send_single(cq, f"Участники ({len(rows)}):\n{names}", main_kb(code, room.owner_id==cq.from_user.id))

# Rules / wishes / recipient
@dp.message(F.text == "ℹ️ Правила")
async def rules_btn(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if not room: await m.answer("Ты ещё не в комнате", reply_markup=kb_root(False)); return
    await m.answer(mk_rules(room), reply_markup=kb_root(True))

@dp.message(F.text == "📝 Хотелки")
async def wishes_btn(m: Message, state: FSMContext):
    room = await get_user_active_room(m.from_user.id)
    if not room: await m.answer("Сначала присоединись", reply_markup=kb_root(False)); return
    await state.update_data(room_code=room.code)
    await state.set_state(Join.wishes)
    await m.answer("Напиши свои хотелки/табу:")

@dp.message(F.text == "📨 Получатель")
async def target_btn(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if not room: await m.answer("Сначала присоединись", reply_markup=kb_root(False)); return
    async with Session() as s:
        me = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id))).scalar_one_or_none()
        if not me: await m.answer("Ты ещё не в комнате", reply_markup=kb_root(False)); return
        pair = (await s.execute(select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id))).scalar_one_or_none()
        if not pair: await m.answer("Жеребьёвки ещё не было", reply_markup=kb_root(True)); return
        recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()
    await m.answer(f"Ты даришь: <b>{recv.name}</b>\nХотелки: {recv.wishes or 'не указаны'}", reply_markup=kb_root(True))

# Draw
@dp.callback_query(F.data.startswith("room_draw:"))
async def cb_draw(cq: CallbackQuery):
    code = cq.data.split(":",1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await cq.answer("Нет комнаты", show_alert=True); return
        if room.owner_id != cq.from_user.id: await cq.answer("Только владелец", show_alert=True); return
        parts = (await s.execute(select(Participant).where(Participant.room_id == room.id))).scalars().all()
        if len(parts) < 2: await cq.answer("Нужно минимум 2 участника", show_alert=True); return
        forb = {(fp.giver_id, fp.receiver_id) for fp in (await s.execute(select(ForbiddenPair).where(ForbiddenPair.room_id == room.id))).scalars().all()}
        await s.execute(Pair.__table__.delete().where(Pair.room_id == room.id))
        pairs = draw_pairs([p.id for p in parts], forb)
        s.add_all([Pair(room_id=room.id, giver_id=g, receiver_id=r) for g,r in pairs])
        room.drawn = True
        await s.commit()
    await cq.answer("Жеребьёвка готова", show_alert=True)

# Hints
@dp.callback_query(F.data.startswith("hint_send:"))
async def cb_hint_send(cq: CallbackQuery):
    code = cq.data.split(":",1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        me = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == cq.from_user.id))).scalar_one_or_none() if room else None
        if not (room and me): await cq.answer("Нужно присоединиться", show_alert=True); return
        pair = (await s.execute(select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id))).scalar_one_or_none()
        if not pair: await cq.answer("Жеребьёвки ещё не было", show_alert=True); return
        since = datetime.now(UTC) - timedelta(days=1)
        cnt = (await s.execute(select(func.count()).select_from(Hint).where(Hint.room_id==room.id, Hint.sender_participant_id==me.id, Hint.created_at>=since))).scalar()
        if cnt >= MAX_HINTS_PER_DAY:
            await cq.answer("Лимит подсказок на сегодня исчерпан", show_alert=True); return
    await cq.message.answer("Напиши подсказку (анонимно).")
    await cq.answer()

@dp.message(F.text.startswith("🕵️"))  # на всякий случай, но реальный ввод — обычный текст
async def noop(_m: Message): pass

@dp.message(F.text & ~F.via_bot & ~F.media_group_id)
async def catch_hint_or_commands(m: Message):
    # Простой роутер: если юзер в комнате и жеребьёвка была — трактуем как подсказку, иначе пропускаем
    room = await get_user_active_room(m.from_user.id)
    if not room: return
    async with Session() as s:
        me = (await s.execute(select(Participant).where(Participant.room_id==room.id, Participant.user_id==m.from_user.id))).scalar_one_or_none()
        if not me: return
        pair = (await s.execute(select(Pair).where(Pair.room_id==room.id, Pair.giver_id==me.id))).scalar_one_or_none()
        if not pair: return
        recv = (await s.execute(select(Participant).where(Participant.id==pair.receiver_id))).scalar_one()
        text = (m.text or "").strip()
        if not text: return
        since = datetime.now(UTC) - timedelta(days=1)
        cnt = (await s.execute(select(func.count()).select_from(Hint).where(Hint.room_id==room.id, Hint.sender_participant_id==me.id, Hint.created_at>=since))).scalar()
        if cnt >= MAX_HINTS_PER_DAY:
            await m.answer("Лимит подсказок на сегодня исчерпан"); return
        s.add(Hint(room_id=room.id, sender_participant_id=me.id, receiver_participant_id=recv.id, text=text[:512]))
        await s.commit()
    with contextlib.suppress(Exception):
        await bot.send_message(recv.user_id, f"🕵️ Тайная подсказка: {text}")
    await m.answer("Подсказка отправлена ✉️")

# Ideas / Buy (affiliate)
@dp.callback_query(F.data.startswith("ideas:"))
async def cb_ideas(cq: CallbackQuery):
    code = cq.data.split(":",1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        me = (await s.execute(select(Participant).where(Participant.room_id==room.id, Participant.user_id==cq.from_user.id))).scalar_one()
        pair = (await s.execute(select(Pair).where(Pair.room_id==room.id, Pair.giver_id==me.id))).scalar_one_or_none()
        if not pair: await cq.answer("Жеребьёвки ещё не было", show_alert=True); return
        recv = (await s.execute(select(Participant).where(Participant.id==pair.receiver_id))).scalar_one()
    q = wishes_to_query(recv.wishes, room.rule_amount_max or room.rule_amount_exact, room.rule_letter)
    await send_single(cq, f"🎁 Идеи для <b>{recv.name}</b> по запросу: <i>{q}</i>\nЖми «🛒 Купить» — дадим ссылки.", main_kb(code, room.owner_id==cq.from_user.id))

@dp.callback_query(F.data.startswith("buy:"))
async def cb_buy(cq: CallbackQuery):
    code = cq.data.split(":",1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        me = (await s.execute(select(Participant).where(Participant.room_id==room.id, Participant.user_id==cq.from_user.id))).scalar_one()
        pair = (await s.execute(select(Pair).where(Pair.room_id==room.id, Pair.giver_id==me.id))).scalar_one_or_none()
        if not pair: await cq.answer("Жеребьёвки ещё не было", show_alert=True); return
        recv = (await s.execute(select(Participant).where(Participant.id==pair.receiver_id))).scalar_one()
    q = wishes_to_query(recv.wishes, room.rule_amount_max or room.rule_amount_exact, room.rule_letter)
    if not AFF_TEMPLATES:
        await send_single(cq, "Партнёрские магазины не настроены (ENV AFFILIATES_JSON).", main_kb(code, room.owner_id==cq.from_user.id))
        return
    kb = InlineKeyboardBuilder()
    for mk, tpl in list(AFF_TEMPLATES.items())[:6]:
        url = mk_aff_url(mk, q)
        if url: kb.button(text=f"Перейти в {HUMAN_NAMES.get(mk, mk)}", url=url)
    kb.button(text="↩️ Назад", callback_data=f"room_open:{code}")
    await send_single(cq, f"🛒 Поиск: <i>{q}</i>\nВыбери магазин:", kb.as_markup())

# Leave room
@dp.message(F.text == "🚪 Выйти из комнаты")
async def leave_room(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if not room: await m.answer("Комнат нет", reply_markup=kb_root(False)); return
    async with Session() as s:
        me = (await s.execute(select(Participant).where(Participant.room_id==room.id, Participant.user_id==m.from_user.id))).scalar_one_or_none()
        if me: await s.delete(me); await s.commit()
    await m.answer("Вышел из комнаты", reply_markup=kb_root(False))

# ------------------------------------------------------------
# Reminder (часовой тик)
# ------------------------------------------------------------
async def reminder_loop():
    await asyncio.sleep(5)
    while True:
        now = datetime.now(UTC)
        if now.minute == 0:
            async with Session() as s:
                rooms = (await s.execute(select(Room).where(Room.drawn == True))).scalars().all()  # noqa: E712
                for room in rooms:
                    dl = room.deadline_at
                    if dl and (dl - now.replace(tzinfo=None)) <= timedelta(days=7):
                        parts = (await s.execute(select(Participant).where(Participant.room_id==room.id))).scalars().all()
                        for p in parts:
                            with contextlib.suppress(Exception):
                                await bot.send_message(p.user_id, "⏰ Напоминание: скоро обмен подарками!")
        await asyncio.sleep(60)

# ------------------------------------------------------------
# main
# ------------------------------------------------------------
async def main():
    await init_db()
    asyncio.create_task(reminder_loop())

    if WEBHOOK_URL:
        from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
        from aiohttp import web
        app = web.Application()
        SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path="/webhook")
        setup_application(app, dp, bot=bot)
        await bot.set_webhook(WEBHOOK_URL + "/webhook", drop_pending_updates=True)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=PORT); await site.start()
        while True: await asyncio.sleep(3600)
    else:
        from aiohttp import web
        info = await bot.get_webhook_info()
        if info.url: await bot.delete_webhook(drop_pending_updates=True)

        got = await acquire_runtime_lock()
        if not got:
            print("Another instance already holds the polling lock. Exiting.")
            return

        app = web.Application()
        app.router.add_get("/health", lambda r: web.Response(text="ok"))
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=PORT); await site.start()
        print(f"Polling + health on :{PORT}/health")

        try:
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        finally:
            with contextlib.suppress(Exception):
                await release_runtime_lock()

if __name__ == "__main__":
    import sys
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(0)
