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
class Base(DeclarativeBase): ...
class Room(Base):
    __tablename__ = "rooms"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(SAString(10), unique=True, index=True)
    owner_id: Mapped[int]
    title: Mapped[str] = mapped_column(default="Secret Santa")
    budget: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    deadline_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=False), nullable=True)
    rule_letter: Mapped[Optional[str]] = mapped_column(SAString(1), nullable=True)
    rule_amount_exact: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    rule_amount_max: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
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
    CONNECT_ARGS["prepare_threshold"] = 0  # fix DuplicatePreparedStatement on psycopg3

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
        if i == n:
            return True
        giver = ids[i]
        random.shuffle(receivers)
        for r in receivers:
            if r == giver:
                continue
            if (giver, r) in forbidden:
                continue
            if r in assigned.values():
                continue
            if r in assigned and assigned.get(r) == giver:
                continue  # prevent two-cycle
            assigned[giver] = r
            if backtrack(i + 1):
                return True
            assigned.pop(giver, None)
        return False

    if not backtrack(0):
        rotated = ids[1:] + ids[:1]
        fallback = list(zip(ids, rotated))
        if any((g, r) in forbidden or g == r for g, r in fallback):
            raise RuntimeError("Cannot satisfy constraints; relax blacklist or add participants")
        return fallback
    return list(assigned.items())

def make_rules_text(room: Room) -> str:
    rules = []
    if room.rule_letter:
        rules.append(f"• Подарок на букву: <b>{room.rule_letter}</b>")
    if room.rule_amount_exact:
        rules.append(f"• Сумма ровно: <b>{room.rule_amount_exact}₽</b>")
    if room.rule_amount_max:
        rules.append(f"• Сумма максимум: <b>{room.rule_amount_max}₽</b>")
    basics = (
        "Общие правила:\n"
        "• Не раскрывай, кому даришь, до обмена 🎅\n"
        "• Уважай хотелки и табу получателя ✅\n"
        "• Дедлайн — ориентир, не тяни до последнего ⏰\n"
        "• Чеки не присылай, эмоции — присылай 🙂"
    )
    spec = "\n".join(rules) if rules else "• Дополнительных ограничений нет."
    extra = f"\n\nБюджет: <b>{room.budget or '—'}</b>\nДедлайн: <b>{room.deadline_at.date() if room.deadline_at else '—'}</b>"
    return f"{basics}\n\nСпец-правила комнаты:\n{spec}{extra}"

# ---------- Single-message UX (no piling) ----------
_last_bot_msg: Dict[int, int] = {}  # chat_id -> message_id

async def send_single(m: Message | CallbackQuery, text: str, reply_markup: Optional[InlineKeyboardMarkup | ReplyKeyboardMarkup] = None):
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
        try:
            await m.answer()
        except Exception:
            pass
    return sent

async def send_menu(m: Message | CallbackQuery, text: str, kb: InlineKeyboardMarkup):
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
    kb.button(text="🏠 В главное меню", callback_data="to_main")
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
            [KeyboardButton(text="✍️ Продолжить"), KeyboardButton(text="👤 Профиль")],
        ],
        resize_keyboard=True
    )

# ---------- Helpers ----------
async def get_user_active_room(user_id: int) -> Optional[Room]:
    async with Session() as s:
        p = (await s.execute(
            select(Participant).where(Participant.user_id == user_id).order_by(Participant.joined_at.desc()).limit(1)
        )).scalar_one_or_none()
        if not p:
            return None
        return (await s.execute(select(Room).where(Room.id == p.room_id))).scalar_one_or_none()

async def show_main_menu(m: Message | CallbackQuery):
    room = await get_user_active_room(m.from_user.id)
    if room:
        await send_single(m, f"Главное меню комнаты <code>{room.code}</code>:", user_reply_kb(True))
        await enter_room_menu(m, room.code)
    else:
        await send_single(m, "Главное меню:", user_reply_kb(False))

async def re_prompt_for_state(m: Message, state: FSMContext):
    st = await state.get_state()
    if st == Join.name.state:
        await send_single(m, "Как тебя звать для списка?", user_reply_kb(False))
    elif st == Join.wishes.state:
        await send_single(m, "Напиши свои хотелки/табу:", user_reply_kb(True))
    elif st == SetBudget.waiting.state:
        await send_single(m, "Введи бюджет в ₽ (число) или 0 чтобы очистить:", user_reply_kb(True))
    elif st == SetDeadline.waiting.state:
        await send_single(m, "Введи дедлайн YYYY-MM-DD или 0 чтобы очистить:", user_reply_kb(True))
    elif st == SetRuleLetter.waiting.state:
        await send_single(m, "Укажи букву (A–Я) или 0 чтобы очистить:", user_reply_kb(True))
    elif st == SetRuleExact.waiting.state:
        await send_single(m, "Укажи точную сумму в ₽ или 0 чтобы очистить:", user_reply_kb(True))
    elif st == SetRuleMax.waiting.state:
        await send_single(m, "Укажи максимальную сумму в ₽ или 0 чтобы очистить:", user_reply_kb(True))
    elif st == AddForbidden.waiting.state:
        await send_single(m, "Отправь два номера через пробел (напр. 1 3) или 0 0 чтобы очистить всё.", user_reply_kb(True))
    elif st == SendHint.waiting_text.state:
        await send_single(m, "Напиши подсказку (анонимно отправим получателю):", user_reply_kb(True))
    else:
        await show_main_menu(m)

# ---------- Global return (works everywhere, no state clear) ----------
@dp.message(F.text.in_({"🏠 Меню", "⬅️ Назад", "Отмена"}), state="*")
async def go_main_any_state(m: Message, state: FSMContext):
    await show_main_menu(m)

@dp.callback_query(F.data == "to_main", state="*")
async def cb_to_main_any_state(cq: CallbackQuery, state: FSMContext):
    await show_main_menu(cq)
    try:
        await cq.answer()
    except Exception:
        pass

@dp.message(F.text == "✍️ Продолжить", state="*")
async def resume_input(m: Message, state: FSMContext):
    await re_prompt_for_state(m, state)

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
        await send_single(m, f"👋 <b>Тайный Санта</b>\nТвоя комната: <code>{room.code}</code>\n\nНажимай «🏠 Меню» в любой момен_
