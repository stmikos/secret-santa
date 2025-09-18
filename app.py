import asyncio
import os
import random
import string
import io
import csv
from datetime import datetime, timedelta, UTC
from typing import List, Tuple, Optional, Set, Dict

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    Message, CallbackQuery, InlineKeyboardMarkup,
    ReplyKeyboardMarkup, KeyboardButton
)
from aiogram.types.input_file import BufferedInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy import String as SAString, DateTime, ForeignKey, Integer, Boolean, UniqueConstraint, select, func

# ---------- Config ----------
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./santa.db")
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # e.g. https://your-app.onrender.com
PORT = int(os.environ.get("PORT", "8000"))
MAX_ROOMS_PER_OWNER = int(os.environ.get("MAX_ROOMS_PER_OWNER", "5"))
MAX_PARTICIPANTS_PER_ROOM = int(os.environ.get("MAX_PARTICIPANTS_PER_ROOM", "200"))
MAX_HINTS_PER_DAY = int(os.environ.get("MAX_HINTS_PER_DAY", "3"))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required")

# ---------- DB ----------
class Base(DeclarativeBase): pass

class Room(Base):
    __tablename__ = "rooms"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(SAString(10), unique=True, index=True)
    owner_id: Mapped[int]
    title: Mapped[str] = mapped_column(default="Secret Santa")
    budget: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    deadline_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    # правила-челленджи
    rule_letter: Mapped[Optional[str]] = mapped_column(SAString(1), nullable=True)
    rule_amount_exact: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rule_amount_max: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # корпоративный режим
    corporate: Mapped[bool] = mapped_column(Boolean, default=False)
    org_name: Mapped[Optional[str]] = mapped_column(SAString(128), nullable=True)
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

class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    room_code: Mapped[Optional[str]] = mapped_column(SAString(10), nullable=True)
    event: Mapped[str] = mapped_column(SAString(64))
    data_json: Mapped[Optional[str]] = mapped_column(SAString(2000), nullable=True)

# ---------- Engine ----------
CONNECT_ARGS = {}
if DATABASE_URL.startswith("postgresql+psycopg://"):
    CONNECT_ARGS["prepare_threshold"] = 0  # fix DuplicatePreparedStatement

engine = create_async_engine(DATABASE_URL, echo=False, future=True, connect_args=CONNECT_ARGS)
Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

# ---------- Utils ----------
def gen_code(n: int = 6) -> str:
    import secrets
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))

async def log(event: str, user_id: Optional[int] = None, room_code: Optional[str] = None, data: Optional[str] = None):
    async with Session() as s:
        s.add(AuditLog(user_id=user_id, room_code=room_code, event=event, data_json=data))
        await s.commit()

def draw_pairs(ids: List[int], forbidden: Set[Tuple[int, int]]) -> List[Tuple[int, int]]:
    n = len(ids)
    if n < 2:
        raise ValueError("At least 2 participants")
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
            if r in assigned and assigned.get(r) == giver: continue  # prevent two-cycle
            assigned[giver] = r
            if backtrack(i + 1): return True
            assigned.pop(giver, None)
        return False

    if not backtrack(0):
        rotated = ids[1:] + ids[:1]
        fallback = list(zip(ids, rotated))
        if any((g, r) in forbidden or g == r for g, r in fallback):
            raise RuntimeError("Cannot satisfy constraints; relax blacklist or add participants")
        return fallback
    return list(assigned.items())

# ---------- Single-message UX (no piling) ----------
_last_bot_msg: Dict[int, int] = {}  # chat_id -> message_id

async def send_single(m: Message | CallbackQuery, text: str, reply_markup: Optional[InlineKeyboardMarkup | ReplyKeyboardMarkup] = None):
    """Send one message per chat: delete previous bot message before sending new."""
    chat_id = m.message.chat.id if isinstance(m, CallbackQuery) else m.chat.id
    bot = m.bot if isinstance(m, CallbackQuery) else m.bot
    prev_id = _last_bot_msg.get(chat_id)
    sent = await (m.message.answer if isinstance(m, CallbackQuery) else m.answer)(text, reply_markup=reply_markup)
    _last_bot_msg[chat_id] = sent.message_id
    if prev_id:
        try:
            await bot.delete_message(chat_id, prev_id)
        except Exception:
            pass
    if isinstance(m, CallbackQuery):
        try: await m.answer()
        except Exception: pass
    return sent

async def send_menu(m: Message | CallbackQuery, text: str, kb: InlineKeyboardMarkup):
    """Try edit; if cannot — fallback to send_single (and delete previous)."""
    try:
        if isinstance(m, CallbackQuery):
            await m.message.edit_text(text, reply_markup=kb)
            await m.answer()
        else:
            await m.edit_text(text, reply_markup=kb)
    except Exception:
        await send_single(m, text, kb)

# ---------- FSM ----------
class Join(StatesGroup):
    name = State()
    wishes = State()

class SetBudget(StatesGroup): waiting = State()
class SetDeadline(StatesGroup): waiting = State()
class AddForbidden(StatesGroup): waiting = State()
class SetRuleLetter(StatesGroup): waiting = State()
class SetRuleExact(StatesGroup): waiting = State()
class SetRuleMax(StatesGroup): waiting = State()
class SendHint(StatesGroup): waiting_text = State()

# ---------- Bot setup ----------
bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

# ---------- Keyboards ----------
def main_kb(room_code: Optional[str] = None, is_owner: bool = False) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    if not room_code:
        kb.button(text="➕ Создать комнату", callback_data="room_new")
        kb.button(text="🔗 Присоединиться", callback_data="room_join")
    else:
        kb.button(text="👥 Участники", callback_data=f"room_participants:{room_code}")
        kb.button(text="✏️ Мои хотелки", callback_data=f"me_edit:{room_code}")
        kb.button(text="📨 Получатель", callback_data=f"me_target:{room_code}")
        kb.button(text="🕵️ Подсказка получателю", callback_data=f"hint_send:{room_code}")
        if is_owner:
            kb.button(text="🎲 Жеребьёвка", callback_data=f"room_draw:{room_code}")
            kb.button(text="⚙️ Настройки", callback_data=f"room_settings:{room_code}")
            kb.button(text="📤 Экспорт CSV", callback_data=f"export_csv:{room_code}")
    kb.adjust(1)
    return kb.as_markup()

def user_reply_kb(in_room: bool) -> ReplyKeyboardMarkup:
    if not in_room:
        return ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="▶️ Старт")],
                [KeyboardButton(text="🔗 Присоединиться"), KeyboardButton(text="➕ Создать")],
                [KeyboardButton(text="👤 Профиль"), KeyboardButton(text="ℹ️ Помощь")],
            ],
            resize_keyboard=True
        )
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🏠 Меню"), KeyboardButton(text="ℹ️ Правила")],
            [KeyboardButton(text="📝 Хотелки"), KeyboardButton(text="📨 Получатель")],
            [KeyboardButton(text="🕵️ Подсказка"), KeyboardButton(text="🚪 Выйти из комнаты")],
            [KeyboardButton(text="👤 Профиль")],
        ],
        resize_keyboard=True
    )

# ---------- Helpers ----------
async def get_user_active_room(user_id: int) -> Optional[Room]:
    async with Session() as s:
        p = (await s.execute(
            select(Participant).where(Participant.user_id == user_id).order_by(Participant.joined_at.desc()).limit(1)
        )).scalar_one_or_none()
        if not p: return None
        return (await s.execute(select(Room).where(Room.id == p.room_id))).scalar_one_or_none()

# ---------- Handlers ----------
@dp.message(CommandStart())
async def cmd_start(m: Message):
    payload = m.text.split(maxsplit=1)[1] if len(m.text.split()) > 1 else ""
    if payload.startswith("room_"):
        code = payload.removeprefix("room_")
        await enter_room_menu(m, code)
        return
    room = await get_user_active_room(m.from_user.id)
    if room:
        await send_single(m, f"👋 <b>Тайный Санта</b>\nТвоя комната: <code>{room.code}</code>", user_reply_kb(True))
        await enter_room_menu(m, room.code)
    else:
        await send_single(m, "👋 <b>Тайный Санта</b>\nСоздай комнату или присоединись по коду.", user_reply_kb(False))

@dp.message(F.text == "▶️ Старт")
@dp.message(F.text == "🏠 Меню")
async def on_start_btn(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if room:
        await send_single(m, "Главное меню комнаты:", user_reply_kb(True))
        await enter_room_menu(m, room.code)
    else:
        await send_single(m, "Главное меню:", user_reply_kb(False))

@dp.message(F.text == "➕ Создать")
async def on_create_btn(m: Message):
    async with Session() as s:
        active_count = (await s.execute(select(func.count()).select_from(Room).where(Room.owner_id == m.from_user.id))).scalar()
        if active_count >= MAX_ROOMS_PER_OWNER:
            await send_single(m, f"Лимит комнат исчерпан ({MAX_ROOMS_PER_OWNER}).", user_reply_kb(False)); return
    code = gen_code()
    async with Session() as s:
        s.add(Room(code=code, owner_id=m.from_user.id))
        await s.commit()
    me = await bot.get_me()
    link = f"https://t.me/{me.username}?start=room_{code}"
    await log("room_new", user_id=m.from_user.id, room_code=code)
    await send_single(m, f"✅ Комната создана: <code>{code}</code>\nЗови друзей: {link}", user_reply_kb(True))
    await enter_room_menu(m, code)

@dp.message(F.text == "🔗 Присоединиться")
async def on_join_btn(m: Message, state: FSMContext):
    await state.update_data(wait_join_code=True)
    await send_single(m, "Введи код комнаты (например: ABC123):", user_reply_kb(False))

@dp.message(F.text.regexp(r"^[A-Za-z0-9]{4,10}$"))
async def on_join_code(m: Message, state: FSMContext):
    data = await state.get_data()
    if not data.get("wait_join_code"): return
    code = m.text.strip().upper()
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room:
            await send_single(m, "Комната не найдена. Проверь код.", user_reply_kb(False)); return
    await state.clear()
    await state.update_data(room_code=code)
    await state.set_state(Join.name)
    await send_single(m, "Как тебя звать для списка?", user_reply_kb(False))

@dp.message(F.text == "ℹ️ Правила")
async def on_rules(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if not room:
        await send_single(m, "Ты ещё не в комнате.", user_reply_kb(False)); return
    rules = []
    if room.rule_letter: rules.append(f"буква {room.rule_letter}")
    if room.rule_amount_exact: rules.append(f"ровно {room.rule_amount_exact}₽")
    if room.rule_amount_max: rules.append(f"до {room.rule_amount_max}₽")
    text = (
        f"Комната: <b>{room.title}</b> (<code>{room.code}</code>)\n"
        f"Бюджет: {room.budget or '—'}\n"
        f"Дедлайн: {room.deadline_at.date() if room.deadline_at else '—'}\n"
        f"Правила: {', '.join(rules) if rules else '—'}"
    )
    await send_single(m, text, user_reply_kb(True))

@dp.message(F.text == "📝 Хотелки")
async def on_wishes_btn(m: Message, state: FSMContext):
    room = await get_user_active_room(m.from_user.id)
    if not room:
        await send_single(m, "Ты ещё не в комнате.", user_reply_kb(False)); return
    await state.update_data(room_code=room.code)
    await state.set_state(Join.wishes)
    await send_single(m, "Напиши свои хотелки/табу:", user_reply_kb(True))

@dp.message(F.text == "📨 Получатель")
async def on_target_btn(m: Message):
    async with Session() as s:
        room = await get_user_active_room(m.from_user.id)
        if not room:
            await send_single(m, "Ты ещё не в комнате.", user_reply_kb(False)); return
        me = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id))).scalar_one_or_none()
        if not me:
            await send_single(m, "Ты ещё не в комнате.", user_reply_kb(False)); return
        pair = (await s.execute(select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id))).scalar_one_or_none()
        if not pair:
            await send_single(m, "Жеребьёвки ещё не было.", user_reply_kb(True)); return
        recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()
    await send_single(m, f"Ты даришь: <b>{recv.name}</b>\nХотелки: {recv.wishes or 'не указаны'}", user_reply_kb(True))

@dp.message(F.text == "🕵️ Подсказка")
async def on_hint_btn(m: Message, state: FSMContext):
    room = await get_user_active_room(m.from_user.id)
    if not room:
        await send_single(m, "Ты ещё не в комнате.", user_reply_kb(False)); return
    await state.update_data(room_code=room.code)
    await state.set_state(SendHint.waiting_text)
    await send_single(m, "Напиши подсказку (анонимно отправим твоему получателю):", user_reply_kb(True))

@dp.message(F.text == "🚪 Выйти из комнаты")
async def on_leave_room(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if not room:
        await send_single(m, "Комнат не найдено.", user_reply_kb(False)); return
    async with Session() as s:
        me = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id))).scalar_one_or_none()
        if me:
            await s.delete(me); await s.commit()
    await send_single(m, "Ты вышел из комнаты. Можно присоединиться к другой или создать новую.", user_reply_kb(False))

@dp.message(F.text == "👤 Профиль")
async def on_profile(m: Message):
    uid = m.from_user.id
    async with Session() as s:
        owned = (await s.execute(select(func.count()).select_from(Room).where(Room.owner_id == uid))).scalar()
        parts = (await s.execute(select(Participant).where(Participant.user_id == uid))).scalars().all()
        last_room = None
        if parts:
            last_room = (await s.execute(select(Room).where(Room.id == parts[-1].room_id))).scalar_one_or_none()
        # hints stats
        since = datetime.now(UTC) - timedelta(days=1)
        # all my participants ids
        p_ids = [p.id for p in parts]
        hints_today = 0
        hints_total = 0
        if p_ids:
            hints_today = (await s.execute(
                select(func.count()).select_from(Hint).where(Hint.sender_participant_id.in_(p_ids), Hint.created_at >= since)
            )).scalar()
            hints_total = (await s.execute(
                select(func.count()).select_from(Hint).where(Hint.sender_participant_id.in_(p_ids))
            )).scalar()
    role = []
    if owned: role.append(f"владелец {owned} комн.")
    if parts: role.append(f"участник {len(parts)} комн.")
    role_text = ", ".join(role) if role else "новичок"
    lr = f"{last_room.title} (<code>{last_room.code}</code>)" if last_room else "—"
    text = (
        f"👤 <b>Профиль</b>\n"
        f"Роль: {role_text}\n"
        f"Последняя комната: {lr}\n"
        f"Подсказок: сегодня {hints_today}, всего {hints_total}"
    )
    in_room = last_room is not None
    await send_single(m, text, user_reply_kb(in_room))

# --------- Join flow (FSM) ----------
@dp.message(Join.name)
async def on_name(m: Message, state: FSMContext):
    await state.update_data(name=(m.text or "").strip()[:64])
    await state.set_state(Join.wishes)
    await send_single(m, "Что подарить? (хотелки/табу)", user_reply_kb(False))

@dp.message(Join.wishes)
async def on_wishes(m: Message, state: FSMContext):
    data = await state.get_data()
    code = data["room_code"]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        count = (await s.execute(select(func.count()).select_from(Participant).where(Participant.room_id == room.id))).scalar()
        if count >= MAX_PARTICIPANTS_PER_ROOM:
            await state.clear()
            await send_single(m, "Достигнут лимит участников для этой комнаты.", user_reply_kb(False)); return
        p = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id))).scalar_one_or_none()
        if p:
            p.name = data["name"]
            p.wishes = (m.text or "")[:512]
        else:
            s.add(Participant(room_id=room.id, user_id=m.from_user.id, name=data["name"], wishes=(m.text or "")[:512]))
        await s.commit()
    await log("join", user_id=m.from_user.id, room_code=code)
    await state.clear()
    await send_single(m, "Записал. Но свечку всё равно подарят 😏", user_reply_kb(True))
    await enter_room_menu(m, code)

# ---------- Inline UI ----------
@dp.callback_query(F.data == "room_new")
async def cb_room_new(cq: CallbackQuery):
    async with Session() as s:
        active_count = (await s.execute(select(func.count()).select_from(Room).where(Room.owner_id == cq.from_user.id))).scalar()
        if active_count >= MAX_ROOMS_PER_OWNER:
            await cq.answer(f"Лимит комнат исчерпан ({MAX_ROOMS_PER_OWNER}).", show_alert=True); return
    code = gen_code()
    async with Session() as s:
        s.add(Room(code=code, owner_id=cq.from_user.id)); await s.commit()
    await log("room_new", user_id=cq.from_user.id, room_code=code)
    me = await bot.get_me()
    link = f"https://t.me/{me.username}?start=room_{code}"
    kb = InlineKeyboardBuilder()
    kb.button(text="🔗 Ссылка для друзей", url=link)
    kb.button(text="➡️ В комнату", callback_data=f"room_open:{code}")
    await send_menu(cq, f"Комната создана: <code>{code}</code>\nЗови друзей по ссылке.", kb.as_markup())

@dp.callback_query(F.data.startswith("room_open:"))
async def cb_room_open(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    await enter_room_menu(cq, code)

async def enter_room_menu(msg: Message | CallbackQuery, code: str):
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room:
            await send_menu(msg, "Комната не найдена.", main_kb()); return
        user_id = msg.from_user.id
        part = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == user_id))).scalar_one_or_none()
        if not part:
            kb = InlineKeyboardBuilder()
            kb.button(text="✅ Присоединиться", callback_data=f"join:{code}")
            kb.button(text="↩️ Назад", callback_data="home")
            rules = []
            if room.rule_letter: rules.append(f"буква {room.rule_letter}")
            if room.rule_amount_exact: rules.append(f"сумма ровно {room.rule_amount_exact}₽")
            if room.rule_amount_max: rules.append(f"сумма до {room.rule_amount_max}₽")
            info = (
                f"Комната <b>{room.title}</b> (<code>{room.code}</code>)\n"
                f"Бюджет: {room.budget or '—'} | Дедлайн: {room.deadline_at.date() if room.deadline_at else '—'}\n"
                f"Правила: {', '.join(rules) if rules else '—'}\n"
                f"Режим: {'Корпоративный' if room.corporate else 'Обычный'}"
            )
            await send_menu(msg, info, kb.as_markup()); return
        is_owner = (room.owner_id == user_id)
        await send_menu(msg, f"Комната <b>{room.title}</b> (<code>{room.code}</code>)", main_kb(room.code, is_owner))

@dp.callback_query(F.data == "home")
async def cb_home(cq: CallbackQuery):
    await send_menu(cq, "Главное меню", main_kb())

@dp.callback_query(F.data.startswith("join:"))
async def cb_join(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code)
    await state.set_state(Join.name)
    await send_menu(cq, "Как тебя звать для списка?", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.callback_query(F.data.startswith("room_participants:"))
async def cb_participants(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await send_menu(cq, "Комната не найдена.", main_kb()); return
        rows = (await s.execute(select(Participant).where(Participant.room_id == room.id).order_by(Participant.joined_at))).scalars().all()
    lines = [f"{i+1}. {p.name}" for i, p in enumerate(rows)]
    names = "\n".join(lines) or "пока пусто"
    await send_menu(cq, f"Участники ({len(rows)}):\n{names}", main_kb(room.code, cq.from_user.id == room.owner_id))

@dp.callback_query(F.data.startswith("me_edit:"))
async def cb_me_edit(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code)
    await state.set_state(Join.wishes)
    await send_menu(cq, "Обнови хотелки:", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.callback_query(F.data.startswith("me_target:"))
async def cb_me_target(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await send_menu(cq, "Комната не найдена.", main_kb()); return
        me = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == cq.from_user.id))).scalar_one_or_none()
        if not me: await send_menu(cq, "Ты ещё не в комнате. Нажми Присоединиться.", main_kb()); return
        pair = (await s.execute(select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id))).scalar_one_or_none()
        if not pair: await send_menu(cq, "Жеребьёвки ещё не было.", main_kb(room.code, cq.from_user.id == room.owner_id)); return
        recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()
    await send_menu(cq, f"Ты даришь: <b>{recv.name}</b>", main_kb(code, cq.from_user.id == room.owner_id))

# ----- Anonymous hints -----
@dp.callback_query(F.data.startswith("hint_send:"))
async def cb_hint_send(cq: CallbackQuery, state: FSMContext):
    from sqlalchemy import select, func as sfunc
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code)
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        me = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == cq.from_user.id))).scalar_one_or_none()
        if not room or not me: await cq.answer("Нужно присоединиться к комнате", show_alert=True); return
        pair = (await s.execute(select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id))).scalar_one_or_none()
        if not pair: await cq.answer("Жеребьёвки ещё не было.", show_alert=True); return
        since = datetime.now(UTC) - timedelta(days=1)
        cnt = (await s.execute(select(sfunc.count()).select_from(Hint).where(Hint.room_id == room.id, Hint.sender_participant_id == me.id, Hint.created_at >= since))).scalar()
        if cnt >= MAX_HINTS_PER_DAY: await cq.answer("Лимит подсказок на сегодня исчерпан", show_alert=True); return
    await state.set_state(SendHint.waiting_text)
    await send_menu(cq, "Напиши подсказку (анонимно отправим твоему получателю):", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.message(SendHint.waiting_text)
async def on_hint_text(m: Message, state: FSMContext):
    from sqlalchemy import select
    data = await state.get_data(); code = data["room_code"]
    text = (m.text or "").strip()
    if not text: await send_single(m, "Пусто. Напиши текст подсказки.", user_reply_kb(True)); return
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        me = (await s.execute(select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id))).scalar_one()
        pair = (await s.execute(select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id))).scalar_one()
        recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()
        s.add(Hint(room_id=room.id, sender_participant_id=me.id, receiver_participant_id=recv.id, text=text[:512]))
        await s.commit()
    try: await bot.send_message(recv.user_id, f"🕵️ Тайная подсказка: {text}")
    except Exception: pass
    await log("hint", user_id=m.from_user.id, room_code=code)
    await state.clear()
    await send_single(m, "Готово. Отправил анонимно ✉️", user_reply_kb(True))

# ----- Settings & Admin -----
def settings_kb(code: str) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="💸 Бюджет", callback_data=f"set_budget:{code}")
    kb.button(text="📅 Дедлайн (YYYY-MM-DD)", callback_data=f"set_deadline:{code}")
    kb.button(text="🔤 Правило: буква", callback_data=f"rule_letter:{code}")
    kb.button(text="💵 Правило: точная сумма", callback_data=f"rule_exact:{code}")
    kb.button(text="💰 Правило: максимум", callback_data=f"rule_max:{code}")
    kb.button(text="🏢 Корп-режим", callback_data=f"corp_toggle:{code}")
    kb.button(text="🚫 Чёрный список пар", callback_data=f"forbid_ui:{code}")
    kb.button(text="⬅️ Назад", callback_data=f"room_open:{code}")
    kb.adjust(1); return kb.as_markup()

@dp.callback_query(F.data.startswith("room_settings:"))
async def cb_settings(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await cq.answer("Нет комнаты", show_alert=True); return
        if room.owner_id != cq.from_user.id: await cq.answer("Только владелец", show_alert=True); return
    await send_menu(cq, "Настройки комнаты:", settings_kb(code))

@dp.callback_query(F.data.startswith("set_budget:"))
async def cb_set_budget(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code); await state.set_state(SetBudget.waiting)
    await send_menu(cq, "Введи бюджет в ₽ (число) или 0 чтобы очистить:", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.message(SetBudget.waiting)
async def on_budget(m: Message, state: FSMContext):
    from sqlalchemy import select
    data = await state.get_data(); code = data["room_code"]
    try: val = int(m.text.strip())
    except Exception: await send_single(m, "Нужно число. Попробуй ещё раз.", user_reply_kb(True)); return
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        if room.owner_id != m.from_user.id: await send_single(m, "Только владелец может менять бюджет", user_reply_kb(True))
        else: room.budget = None if val <= 0 else val; await s.commit()
    await state.clear(); await enter_room_menu(m, code)

@dp.callback_query(F.data.startswith("set_deadline:"))
async def cb_set_deadline(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code); await state.set_state(SetDeadline.waiting)
    await send_menu(cq, "Введи дедлайн в формате YYYY-MM-DD или 0 чтобы очистить:", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.message(SetDeadline.waiting)
async def on_deadline(m: Message, state: FSMContext):
    from sqlalchemy import select
    data = await state.get_data(); code = data["room_code"]; txt = (m.text or "").strip()
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        if room.owner_id != m.from_user.id:
            await send_single(m, "Только владелец может менять дедлайн", user_reply_kb(True))
        else:
            if txt == "0": room.deadline_at = None
            else:
                try: room.deadline_at = datetime.strptime(txt, "%Y-%m-%d").replace(tzinfo=None)
                except Exception: await send_single(m, "Неверный формат. Пример: 2025-12-20", user_reply_kb(True)); return
            await s.commit()
    await state.clear(); await enter_room_menu(m, code)

@dp.callback_query(F.data.startswith("rule_letter:"))
async def cb_rule_letter(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code); await state.set_state(SetRuleLetter.waiting)
    await send_menu(cq, "Укажи букву (A–Я) или 0 чтобы очистить:", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.message(SetRuleLetter.waiting)
async def on_rule_letter(m: Message, state: FSMContext):
    from sqlalchemy import select
    code = (await state.get_data())["room_code"]; val = (m.text or "").strip()
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        if room.owner_id != m.from_user.id: await send_single(m, "Только владелец", user_reply_kb(True))
        else: room.rule_letter = None if val == "0" else val[:1].upper(); await s.commit()
    await state.clear(); await enter_room_menu(m, code)

@dp.callback_query(F.data.startswith("rule_exact:"))
async def cb_rule_exact(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code); await state.set_state(SetRuleExact.waiting)
    await send_menu(cq, "Укажи точную сумму в ₽ или 0 чтобы очистить:", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.message(SetRuleExact.waiting)
async def on_rule_exact(m: Message, state: FSMContext):
    from sqlalchemy import select
    code = (await state.get_data())["room_code"]
    try: val = int(m.text.strip())
    except Exception: await send_single(m, "Нужно число", user_reply_kb(True)); return
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        if room.owner_id != m.from_user.id: await send_single(m, "Только владелец", user_reply_kb(True))
        else: room.rule_amount_exact = None if val <= 0 else val; await s.commit()
    await state.clear(); await enter_room_menu(m, code)

@dp.callback_query(F.data.startswith("rule_max:"))
async def cb_rule_max(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code); await state.set_state(SetRuleMax.waiting)
    await send_menu(cq, "Укажи максимальную сумму в ₽ или 0 чтобы очистить:", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.message(SetRuleMax.waiting)
async def on_rule_max(m: Message, state: FSMContext):
    from sqlalchemy import select
    code = (await state.get_data())["room_code"]
    try: val = int(m.text.strip())
    except Exception: await send_single(m, "Нужно число", user_reply_kb(True)); return
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        if room.owner_id != m.from_user.id: await send_single(m, "Только владелец", user_reply_kb(True))
        else: room.rule_amount_max = None if val <= 0 else val; await s.commit()
    await state.clear(); await enter_room_menu(m, code)

@dp.callback_query(F.data.startswith("corp_toggle:"))
async def cb_corp_toggle(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await cq.answer("Нет комнаты", show_alert=True); return
        if room.owner_id != cq.from_user.id: await cq.answer("Только владелец", show_alert=True); return
        room.corporate = not room.corporate; await s.commit()
    await log("corp_toggle", user_id=cq.from_user.id, room_code=code, data="on" if room.corporate else "off")
    await enter_room_menu(cq, code)

@dp.callback_query(F.data.startswith("forbid_ui:"))
async def cb_forbid_ui(cq: CallbackQuery, state: FSMContext):
    from sqlalchemy import select
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await cq.answer("Нет комнаты", show_alert=True); return
        if room.owner_id != cq.from_user.id: await cq.answer("Только владелец", show_alert=True); return
        parts = (await s.execute(select(Participant).where(Participant.room_id == room.id))).scalars().all()
        forb = (await s.execute(select(ForbiddenPair).where(ForbiddenPair.room_id == room.id))).scalars().all()
    if not parts:
        await send_menu(cq, "Список пуст.", settings_kb(code)); return
    id_to_idx = {p.id: i+1 for i, p in enumerate(parts)}
    ftext = "\n".join(f"{id_to_idx[fp.giver_id]}→{id_to_idx[fp.receiver_id]}" for fp in forb) or "—"
    txt = (
        "Чёрный список пар (giver→receiver):\n"
        f"Текущие: {ftext}\n\n"
        "Отправь два номера через пробел, напр.: <code>1 3</code> (1 не дарит 3).\n"
        "Отправь 0 0 чтобы очистить всё."
    )
    await state.update_data(room_code=code); await state.set_state(AddForbidden.waiting)
    await send_menu(cq, txt, InlineKeyboardMarkup(inline_keyboard=[]))

@dp.message(AddForbidden.waiting)
async def on_forbidden(m: Message, state: FSMContext):
    from sqlalchemy import select
    data = await state.get_data(); code = data["room_code"]; txt = (m.text or "").strip()
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        if room.owner_id != m.from_user.id: await send_single(m, "Только владелец", user_reply_kb(True)); return
        parts = (await s.execute(select(Participant).where(Participant.room_id == room.id).order_by(Participant.joined_at))).scalars().all()
        if txt == "0 0":
            await s.execute(ForbiddenPair.__table__.delete().where(ForbiddenPair.room_id == room.id)); await s.commit()
            await state.clear(); await enter_room_menu(m, code); return
        try:
            i_str, j_str = txt.split(); i, j = int(i_str) - 1, int(j_str) - 1
            if not (0 <= i < len(parts) and 0 <= j < len(parts)): raise ValueError
        except Exception:
            await send_single(m, "Неверный ввод. Пример: 1 3 или 0 0", user_reply_kb(True)); return
        giver, recv = parts[i], parts[j]
        if giver.id == recv.id:
            await send_single(m, "Нельзя запрещать самого себя (это и так запрещено)", user_reply_kb(True)); return
        exists = (await s.execute(select(ForbiddenPair).where(
            ForbiddenPair.room_id == room.id, ForbiddenPair.giver_id == giver.id, ForbiddenPair.receiver_id == recv.id
        ))).scalar_one_or_none()
        if not exists:
            from json import dumps
            s.add(ForbiddenPair(room_id=room.id, giver_id=giver.id, receiver_id=recv.id)); await s.commit()
            await log("forbid_add", user_id=m.from_user.id, room_code=code, data=dumps({"giver": giver.name, "recv": recv.name}))
    await state.clear(); await enter_room_menu(m, code)

@dp.callback_query(F.data.startswith("room_draw:"))
async def cb_room_draw(cq: CallbackQuery):
    from sqlalchemy import select
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await send_menu(cq, "Комната не найдена.", main_kb()); return
        if room.owner_id != cq.from_user.id: await cq.answer("Только создатель комнаты.", show_alert=True); return
        parts = (await s.execute(select(Participant).where(Participant.room_id == room.id))).scalars().all()
        if len(parts) < 2: await cq.answer("Нужно минимум 2 участника.", show_alert=True); return
        forbidden = {(fp.giver_id, fp.receiver_id) for fp in (await s.execute(select(ForbiddenPair).where(ForbiddenPair.room_id == room.id))).scalars().all()}
        await s.execute(Pair.__table__.delete().where(Pair.room_id == room.id))
        pairs = draw_pairs([p.id for p in parts], forbidden)
        s.add_all([Pair(room_id=room.id, giver_id=g, receiver_id=r) for g, r in pairs])
        room.drawn = True; await s.commit()
    await cq.answer("Жеребьёвка выполнена!", show_alert=True)
    await log("draw", user_id=cq.from_user.id, room_code=code)
    await send_menu(cq, "Жеребьёвка готова. Всем отправлены инструкции в личку.", main_kb(code, True))
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        pairs = (await s.execute(select(Pair).where(Pair.room_id == room.id))).scalars().all()
        rules = []
        if room.rule_letter: rules.append(f"буква {room.rule_letter}")
        if room.rule_amount_exact: rules.append(f"сумма ровно {room.rule_amount_exact}₽")
        if room.rule_amount_max: rules.append(f"сумма до {room.rule_amount_max}₽")
        rules_text = ("\nПравило: " + ", ".join(rules)) if rules else ""
        for pair in pairs:
            giver = (await s.execute(select(Participant).where(Participant.id == pair.giver_id))).scalar_one()
            recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()
            try:
                await bot.send_message(giver.user_id,
                    f"🎄 Твой получатель: <b>{recv.name}</b>{rules_text}\n"
                    f"Хотелки: {recv.wishes or 'не указаны'}")
            except Exception: pass

@dp.callback_query(F.data.startswith("export_csv:"))
async def cb_export_csv(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room: await cq.answer("Нет комнаты", show_alert=True); return
        if room.owner_id != cq.from_user.id: await cq.answer("Только владелец", show_alert=True); return
        parts = (await s.execute(select(Participant).where(Participant.room_id == room.id))).scalars().all()
        pairs = (await s.execute(select(Pair).where(Pair.room_id == room.id))).scalars().all()
    part_by_id = {p.id: p for p in parts}
    buf = io.StringIO(); w = csv.writer(buf)
    w.writerow(["name", "wishes", "target_name"])
    tgt_by_giver = {p.giver_id: p.receiver_id for p in pairs}
    for p in parts:
        recv = part_by_id.get(tgt_by_giver.get(p.id))
        w.writerow([p.name, p.wishes, recv.name if recv else ""])
    data = buf.getvalue().encode("utf-8")
    await bot.send_document(cq.from_user.id, BufferedInputFile(data, filename=f"secret_santa_{code}.csv"))
    await cq.answer("Отправил CSV в личку")

# ---------- Reminders ----------
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
                        parts = (await s.execute(select(Participant).where(Participant.room_id == room.id))).scalars().all()
                        for p in parts:
                            try: await bot.send_message(p.user_id, "⏰ Напоминание: скоро обмен подарками! Зайди в «Получатель».")
                            except Exception: pass
        await asyncio.sleep(60)

# ---------- Entrypoint ----------
async def main():
    await init_db()
    asyncio.create_task(reminder_loop())

    if WEBHOOK_URL:
        from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
        from aiohttp import web
        app = web.Application()
        webhook_path = "/webhook"
        SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path=webhook_path)
        setup_application(app, dp, bot=bot)
        await bot.set_webhook(WEBHOOK_URL + webhook_path, drop_pending_updates=True)
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=PORT); await site.start()
        print(f"Webhook listening on :{PORT}{webhook_path}")
        while True: await asyncio.sleep(3600)
    else:
        # polling + health endpoint для Render
        from aiohttp import web
        info = await bot.get_webhook_info()
        if info.url:  # гасим случайный чужой вебхук
            print("Drop foreign webhook:", info.url)
        await bot.delete_webhook(drop_pending_updates=True)

        app = web.Application()
        app.router.add_get("/health", lambda request: web.Response(text="ok"))
        runner = web.AppRunner(app); await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=PORT); await site.start()
        print(f"Polling + health on :{PORT}/health")

        await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
