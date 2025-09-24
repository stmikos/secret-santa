# app.py — Secret Santa Bot (минимальный, рабочий)
# Python 3.11+ / Aiogram 3.7+

import os
import re
import asyncio
import string
import hashlib
import contextlib
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
    Message, CallbackQuery,
    InlineKeyboardMarkup, ReplyKeyboardMarkup, KeyboardButton
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

from sqlalchemy import (
    select, func, Integer, BigInteger, Boolean, ForeignKey,
    String as SAString, DateTime, UniqueConstraint, text
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import NullPool
from urllib.parse import urlsplit, urlunsplit, parse_qsl, urlencode

# ============================================================
# ENV + URL sanitize
# ============================================================
BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required")

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite+aiosqlite:///./santa.db")

def _sanitize_pg_url(url: str) -> str:
    if not url.startswith(("postgres://", "postgresql://", "postgresql+psycopg://")):
        return url
    if url.startswith("postgres://"):
        url = url.replace("postgres://", "postgresql+psycopg://", 1)
    elif url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+psycopg://", 1)
    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    bad = {
        "prepared_statement_cache_size",
        "statement_cache_size",
        "prepared_statements",
        "server_prepared_statements",
    }
    for k in list(q):
        if k in bad:
            q.pop(k, None)
    q.setdefault("sslmode", "require")
    new_parts = parts._replace(query=urlencode(q))
    return urlunsplit(new_parts)

DATABASE_URL = _sanitize_pg_url(DATABASE_URL)

WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # если задан — режим webhook, иначе polling
PORT = int(os.environ.get("PORT", "10000"))

MAX_ROOMS_PER_OWNER = int(os.environ.get("MAX_ROOMS_PER_OWNER", "5"))
MAX_PARTICIPANTS_PER_ROOM = int(os.environ.get("MAX_PARTICIPANTS_PER_ROOM", "200"))
MAX_HINTS_PER_DAY = int(os.environ.get("MAX_HINTS_PER_DAY", "3"))

# Аффилиаты: JSON в ENV AFFILIATES_JSON, пример:
# {"wb":"https://www.wildberries.ru/catalog/0/search.aspx?search={q}"}
try:
    AFF_TEMPLATES: Dict[str, str] = dict(**(__import__("json").loads(os.environ.get("AFFILIATES_JSON", "{}"))))
except Exception:
    AFF_TEMPLATES = {}
HUMAN_NAMES = {"wb": "Wildberries", "ozon": "Ozon", "ym": "Яндекс.Маркет"}

# ============================================================
# DB models
# ============================================================
class Base(DeclarativeBase): pass

class Room(Base):
    __tablename__ = "rooms"
    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(SAString(10), unique=True, index=True)
    owner_id: Mapped[int] = mapped_column(BigInteger)
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
    user_id: Mapped[int] = mapped_column(BigInteger, index=True)
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

class Referral(Base):
    __tablename__ = "referrals"
    id: Mapped[int] = mapped_column(primary_key=True)
    referrer_user_id: Mapped[int] = mapped_column(BigInteger, index=True)
    referee_user_id: Mapped[int] = mapped_column(BigInteger, index=True, unique=True)  # один раз кем-то приглашён
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(UTC))

# ============================================================
# Engine / Session (расположить ДО функций, где используется Session)
# ============================================================
CONNECT_ARGS: Dict[str, object] = {}
if DATABASE_URL.startswith("postgresql+psycopg://"):
    # Disable server-side prepared statements so PgBouncer in transaction
    # pooling mode doesn't invalidate cached statements between requests.
    CONNECT_ARGS["prepare_threshold"] = None

engine = create_async_engine(
    DATABASE_URL,
    echo=False,
    future=True,
    connect_args=CONNECT_ARGS,
    poolclass=NullPool,  # безопасно за PgBouncer
)
Session = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

async def init_db():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

async def migrate_bigints_if_needed():
    if not DATABASE_URL.startswith("postgresql+psycopg://"):
        return
    async with engine.begin() as conn:
        await conn.execute(text("""
        DO $$
        BEGIN
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='rooms' AND column_name='owner_id' AND data_type IN ('integer','int4')
          ) THEN
            ALTER TABLE rooms ALTER COLUMN owner_id TYPE BIGINT;
          END IF;
          IF EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE table_name='participants' AND column_name='user_id' AND data_type IN ('integer','int4')
          ) THEN
            ALTER TABLE participants ALTER COLUMN user_id TYPE BIGINT;
          END IF;
        END $$;
        """))

# ============================================================
# Helpers
# ============================================================
def gen_code(n: int = 6) -> str:
    import secrets
    alphabet = string.ascii_uppercase + string.digits
    return "".join(secrets.choice(alphabet) for _ in range(n))
   
def mk_rules(room: Room) -> str:
    rows = ["Параметры комнаты:"]
    if room.rule_letter:
        rows.append(f"• Буква подарка: <b>{room.rule_letter}</b>")
    if room.rule_amount_exact:
        rows.append(f"• Сумма ровно: <b>{room.rule_amount_exact}₽</b>")
    if room.rule_amount_max:
        rows.append(f"• Сумма максимум: <b>{room.rule_amount_max}₽</b>")
    if room.budget:
        rows.append(f"• Бюджет: <b>{room.budget}₽</b>")
    if room.deadline_at:
        rows.append(f"• Дедлайн: <b>{room.deadline_at.date()}</b>")
    return "\n".join(rows)

def wishes_to_query(wishes: str, budget_max: Optional[int], letter: Optional[str]) -> str:
    parts: List[str] = []
    if wishes: parts.append(wishes)
    if budget_max: parts.append(f"до {budget_max} руб")
    if letter: parts.append(f"на букву {letter}")
    return ", ".join(parts) or "подарок сюрприз"

def mk_aff_url(marketplace: str, query: str, room_code: str | None = None, user_id: int | None = None) -> Optional[str]:
    tpl = AFF_TEMPLATES.get(marketplace)
    if not tpl:
        return None
    base = tpl.replace("{q}", urllib.parse.quote_plus(query.strip()))
    utm = {
        "utm_source": "santa_bot",
        "utm_medium": "room" if room_code else "menu",
        "utm_campaign": room_code or "none",
        "utm_content": str(user_id or 0),
    }
    sep = "&" if "?" in base else "?"
    return base + sep + urllib.parse.urlencode(utm)

def draw_pairs(ids: List[int], forbidden: Set[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    n = len(ids)
    if n < 2:
        raise ValueError("Need >=2 participants")

    forbidden_set = set(forbidden)

    rotation = list(zip(ids, ids[1:] + ids[:1]))
    if all((giver, receiver) not in forbidden_set for giver, receiver in rotation):
        return rotation
    adjacency: List[List[int]] = []
    for giver in ids:
        options = [r for r in ids if r != giver and (giver, r) not in forbidden_set]
        options.sort()
        if not options:
            return None
        adjacency.append(options)

    match_r: Dict[int, int] = {}
    giver_to_receiver: List[Optional[int]] = [None] * n

    def dfs(i: int, seen: Set[int]) -> bool:
        for receiver in adjacency[i]:
            if receiver in seen:
                continue
            seen.add(receiver)
            current = match_r.get(receiver)
            if current is None or dfs(current, seen):
                match_r[receiver] = i
                giver_to_receiver[i] = receiver
                return True
        return False

    for i in range(n):
        if not dfs(i, set()):
            return None

    result: List[Tuple[int, int]] = []
    for i, giver in enumerate(ids):
        receiver = giver_to_receiver[i]
        if receiver is None:
            return None
        result.append((giver, receiver))
    return result

async def get_user_active_room(user_id: int) -> Optional[Room]:
    async with Session() as s:
        p = (await s.execute(
            select(Participant).where(Participant.user_id == user_id)
            .order_by(Participant.joined_at.desc())
            .limit(1)
        )).scalar_one_or_none()
        if not p: return None
        return (await s.execute(select(Room).where(Room.id == p.room_id))).scalar_one_or_none()

async def record_referral_if_first(ref_payload: str, referee_user_id: int) -> None:
    # ожидаем payload вида 'ref_123456789'
    if not ref_payload.startswith("ref_"):
        return
    try:
        referrer = int(ref_payload[4:])
    except ValueError:
        return
    if referrer <= 0 or referrer == referee_user_id:
        return
    async with Session() as s:
        # если этот referee уже кем-то отмечен — не перезаписываем
        exists = (await s.execute(
            select(Referral).where(Referral.referee_user_id == referee_user_id)
        )).scalar_one_or_none()
        if exists:
            return
        s.add(Referral(referrer_user_id=referrer, referee_user_id=referee_user_id))
        await s.commit()

# ============================================================
# Runtime lock (single-instance polling)
# ============================================================
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
                await s.delete(existing)
                await s.commit()
            else:
                return False
        s.add(RuntimeLock(bot_token_hash=h, started_at=now))
        try:
            await s.commit()
            return True
        except IntegrityError:
            await s.rollback()
            return False

async def release_runtime_lock():
    h = hashlib.sha256(BOT_TOKEN.encode()).hexdigest()
    async with Session() as s:
        row = (await s.execute(select(RuntimeLock).where(RuntimeLock.bot_token_hash == h))).scalar_one_or_none()
        if row:
            await s.delete(row)
            await s.commit()

# ============================================================
# Bot / Keyboards
# ============================================================
bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

class Join(StatesGroup):
    name = State()
    wishes = State()

def kb_root(in_room: bool) -> ReplyKeyboardMarkup:
    if not in_room:
        return ReplyKeyboardMarkup(
            keyboard=[
                [KeyboardButton(text="➕ Создать комнату"), KeyboardButton(text="🔗 Подключиться")],
                [KeyboardButton(text="ℹ️ Правила"), KeyboardButton(text="📰 Новости")],
            ],
            resize_keyboard=True,
            one_time_keyboard=False,
            input_field_placeholder="Создай комнату или подключись по коду"
        )
    return ReplyKeyboardMarkup(
        keyboard=[
            [KeyboardButton(text="🏠 Меню"), KeyboardButton(text="ℹ️ Правила")],
            [KeyboardButton(text="📝 Хотелки"), KeyboardButton(text="📨 Получатель")],
            [KeyboardButton(text="🎁 Идеи"), KeyboardButton(text="🛒 Купить")],
            [KeyboardButton(text="🚪 Выйти из комнаты")],
        ],
        resize_keyboard=True,
        one_time_keyboard=False
    )

def main_kb(code: Optional[str], is_owner: bool) -> InlineKeyboardMarkup:
    b = InlineKeyboardBuilder()
    if code:
        b.button(text="👥 Участники", callback_data=f"room_participants:{code}")
        b.button(text="✏️ Мои хотелки", callback_data=f"me_edit:{code}")
        b.button(text="📨 Получатель", callback_data=f"me_target:{code}")
        b.button(text="🕵️ Подсказка", callback_data=f"hint_send:{code}")
        b.button(text="🎁 Идеи", callback_data=f"ideas:{code}")
        b.button(text="🛒 Купить", callback_data=f"buy:{code}")
        if is_owner:
            b.button(text="🎲 Жеребьёвка", callback_data=f"room_draw:{code}")
    b.button(text="🏠 В главное меню", callback_data="to_main")
    b.adjust(1)
    return b.as_markup()

async def send_single(m: Message | CallbackQuery, text: str, kb: InlineKeyboardMarkup | ReplyKeyboardMarkup | None = None):
    if isinstance(m, CallbackQuery):
        sent = await m.message.answer(text, reply_markup=kb)
        with contextlib.suppress(Exception):
            await m.answer()
        return sent
    return await m.answer(text, reply_markup=kb)

async def show_main(m: Message | CallbackQuery):
    room = await get_user_active_room(m.from_user.id)
    await send_single(m, "Главное меню", kb_root(bool(room)))

# ============================================================
# Handlers
# ============================================================
@dp.message(StateFilter("*"), F.text.in_({"🏠 Меню", "⬅️ Назад", "Отмена", "/menu"}))
async def any_to_menu(m: Message, state: FSMContext):
    await state.clear()
    await show_main(m)

@dp.callback_query(StateFilter("*"), F.data == "to_main")
async def cb_to_main(cq: CallbackQuery, state: FSMContext):
    await state.clear()
    await show_main(cq)

@dp.message(CommandStart())
async def on_cmd_start(m: Message, state: FSMContext):
    await state.clear()
    payload = m.text.split(maxsplit=1)[1] if len(m.text.split()) > 1 else ""

    # deep-link: комната
    if payload.startswith("room_"):
        await enter_room_menu(m, payload.removeprefix("room_"))
        return

    # deep-link: реферал
    if payload.startswith("ref_"):
        await record_referral_if_first(payload, m.from_user.id)

    await show_main(m)

# реферал
@dp.message(F.text == "👤 Профиль")
async def profile(m: Message):
    me = await bot.get_me()
    ref_link = f"https://t.me/{me.username}?start=ref_{m.from_user.id}"
    await m.answer(
        "Твоя реф-ссылка:\n"
        f"{ref_link}\n\n"
        "За приглашения можно получить PRO-фичи и бонусы.",
        reply_markup=kb_root(bool(await get_user_active_room(m.from_user.id)))
    )

# Создать комнату
@dp.message(F.text == "➕ Создать комнату")
async def on_create_btn(m: Message):
    async with Session() as s:
        cnt = (await s.execute(
            select(func.count()).select_from(Room).where(Room.owner_id == m.from_user.id)
        )).scalar()
        if cnt >= MAX_ROOMS_PER_OWNER:
            await m.answer("Лимит комнат исчерпан")
            return
        code = gen_code()
        s.add(Room(code=code, owner_id=m.from_user.id))
        await s.commit()
    me = await bot.get_me()
    link = f"https://t.me/{me.username}?start=room_{code}"
    await m.answer(
        f"Комната создана: <code>{code}</code>\nПриглашение: {link}",
        reply_markup=kb_root(True)
    )
    await enter_room_menu(m, code)

# Присоединиться
@dp.message(F.text == "🔗 Подключиться")
async def on_join_btn(m: Message, state: FSMContext):
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
            await m.answer("Комната не найдена")
            return
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
        me = (await s.execute(
            select(Participant).where(
                Participant.room_id == room.id,
                Participant.user_id == m.from_user.id
            )
        )).scalar_one_or_none()

        if not me:
            count = (await s.execute(
                select(func.count()).select_from(Participant).where(Participant.room_id == room.id)
            )).scalar()
            if count >= MAX_PARTICIPANTS_PER_ROOM:
                await state.clear()
                await m.answer("Лимит участников достигнут")
                return

        name = data.get("name") or (me.name if me else None)
        if not name:
            await state.clear()
            await m.answer("Сначала представься", reply_markup=kb_root(False))
            return

        if me:
            me.name = name
            me.wishes = (m.text or "")[:512]
        else:
            s.add(Participant(
                room_id=room.id,
                user_id=m.from_user.id,
                name=name,
                wishes=(m.text or "")[:512]
            ))
        await s.commit()

    await state.clear()
    await m.answer("Записал. Но свечку всё равно подарят 😏", reply_markup=kb_root(True))
    await enter_room_menu(m, code)

# Открыть комнату / список участников
async def enter_room_menu(msg: Message | CallbackQuery, code: str):
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room:
            await send_single(msg, "Комната не найдена", kb_root(False))
            return
        user_id = msg.from_user.id
        part = (await s.execute(
            select(Participant).where(Participant.room_id == room.id, Participant.user_id == user_id)
        )).scalar_one_or_none()
    if not part:
        kb = InlineKeyboardBuilder()
        kb.button(text="✅ Присоединиться", callback_data=f"join:{code}")
        kb.button(text="↩️ Назад", callback_data="to_main")
        await send_single(
            msg,
            f"Комната <b>{room.title}</b> (<code>{room.code}</code>)\n"
            f"Бюджет: {room.budget or '—'} | Дедлайн: {room.deadline_at.date() if room.deadline_at else '—'}",
            kb.as_markup()
        )
        return
    await send_single(
        msg,
        f"Комната <b>{room.title}</b> (<code>{room.code}</code>)",
        main_kb(code, room.owner_id == msg.from_user.id)
    )

@dp.callback_query(F.data.startswith("join:"))
async def cb_join(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    await state.update_data(room_code=code)
    await state.set_state(Join.name)
    await send_single(cq, "Как тебя звать для списка?", InlineKeyboardMarkup(inline_keyboard=[]))

@dp.callback_query(F.data.startswith("room_participants:"))
async def cb_participants(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room:
            await cq.answer("Нет комнаты", show_alert=True)
            return
        rows = (await s.execute(
            select(Participant).where(Participant.room_id == room.id).order_by(Participant.joined_at)
        )).scalars().all()
    names = "\n".join(f"{i+1}. {p.name}" for i, p in enumerate(rows)) or "пока пусто"
    await send_single(cq, f"Участники ({len(rows)}):\n{names}", main_kb(code, room.owner_id == cq.from_user.id))

# Правила / Новости
GENERAL_RULES = (
    "Правила игры:\n"
    "• Не раскрывай, кому даришь, до обмена 🎅\n"
    "• Уважай хотелки и табу получателя ✅\n"
    "• Соблюдай дедлайн ⏰\n"
    "• Дарим эмоции, а не аргументы из бухгалтерии 🙂"
)

# ✅ КОРОТКИЕ ПРАВИЛА ДЛЯ КОМНАТЫ
SHORT_ROOM_RULES = (
    "Правила комнаты 🎁\n"
    "• Дарим подарок только своему получателю.\n"
    "• Не раскрываем, кто кому дарит, до обмена.\n"
    "• Соблюдаем бюджет и дедлайн.\n"
    "• Учитываем хотелки и табу получателя.\n"
    "• Подсказки — только через бота (анонимно)."
)

@dp.message(F.text == "ℹ️ Правила")
async def rules_btn(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if room:
        # короткие правила + динамические параметры комнаты
        text = SHORT_ROOM_RULES + "\n\n" + mk_rules(room)
        await m.answer(text, reply_markup=kb_root(True))
    else:
        await m.answer(GENERAL_RULES, reply_markup=kb_root(False))

@dp.message(F.text == "📰 Новости")
async def on_news(m: Message):
    await m.answer(
        "Новости бота:\n"
        "• Подсказки «Санта → получателю»\n"
        "• Челлендж-правила (буква/сумма)\n"
        "• Интеграции WB/Ozon/Я.Маркет\n"
        "• Корпоративный режим",
        reply_markup=kb_root(bool(await get_user_active_room(m.from_user.id)))
    )

# Хотелки / Получатель
@dp.message(F.text == "📝 Хотелки")
async def wishes_btn(m: Message, state: FSMContext):
    room = await get_user_active_room(m.from_user.id)
    if not room:
        await m.answer("Сначала присоединись", reply_markup=kb_root(False))
        return
    async with Session() as s:
        me = (await s.execute(
            select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id)
        )).scalar_one_or_none()
    if not me:
        await m.answer("Ты ещё не в комнате", reply_markup=kb_root(False))
        return
    await state.clear()
    await state.update_data(room_code=room.code, name=me.name)
    await state.set_state(Join.wishes)
    await m.answer("Напиши свои хотелки/табу:")

@dp.message(F.text == "📨 Получатель")
async def target_btn(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if not room:
        await m.answer("Сначала присоединись", reply_markup=kb_root(False))
        return
    async with Session() as s:
        me = (await s.execute(
            select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id)
        )).scalar_one_or_none()
        if not me:
            await m.answer("Ты ещё не в комнате", reply_markup=kb_root(False))
            return
        pair = (await s.execute(
            select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id)
        )).scalar_one_or_none()
        if not pair:
            await m.answer("Жеребьёвки ещё не было", reply_markup=kb_root(True))
            return
        recv = (await s.execute(
            select(Participant).where(Participant.id == pair.receiver_id)
        )).scalar_one()
    await m.answer(f"Ты даришь: <b>{recv.name}</b>\nХотелки: {recv.wishes or 'не указаны'}", reply_markup=kb_root(True))

@dp.callback_query(F.data.startswith("me_edit:"))
async def cb_me_edit(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        me = None
        if room:
            me = (await s.execute(
                select(Participant).where(Participant.room_id == room.id, Participant.user_id == cq.from_user.id)
            )).scalar_one_or_none()
    if not room:
        await cq.answer("Комната не найдена", show_alert=True)
        return
    if not me:
        await cq.answer("Нужно присоединиться", show_alert=True)
        return
    await state.clear()
    await state.update_data(room_code=room.code, name=me.name)
    await state.set_state(Join.wishes)
    await cq.message.answer("Напиши свои хотелки/табу:")
    await cq.answer()

@dp.callback_query(F.data.startswith("me_target:"))
async def cb_me_target(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        me = None
        pair = None
        recv = None
        if room:
            me = (await s.execute(
                select(Participant).where(Participant.room_id == room.id, Participant.user_id == cq.from_user.id)
            )).scalar_one_or_none()
            if me:
                pair = (await s.execute(
                    select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id)
                )).scalar_one_or_none()
                if pair:
                    recv = (await s.execute(
                        select(Participant).where(Participant.id == pair.receiver_id)
                    )).scalar_one_or_none()
    if not room:
        await cq.answer("Комната не найдена", show_alert=True)
        return
    if not me:
        await cq.answer("Нужно присоединиться", show_alert=True)
        return
    if not pair or not recv:
        await send_single(cq, "Жеребьёвки ещё не было", main_kb(code, room.owner_id == cq.from_user.id))
        return
    await send_single(cq, f"Ты даришь: <b>{recv.name}</b>\nХотелки: {recv.wishes or 'не указаны'}", main_kb(code, room.owner_id == cq.from_user.id))

# Жеребьёвка
@dp.callback_query(F.data.startswith("room_draw:"))
async def cb_draw(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        if not room:
            await cq.answer("Нет комнаты", show_alert=True)
            return
        if room.owner_id != cq.from_user.id:
            await cq.answer("Только владелец", show_alert=True)
            return
        parts = (await s.execute(select(Participant).where(Participant.room_id == room.id))).scalars().all()
        if len(parts) < 2:
            await cq.answer("Нужно минимум 2 участника", show_alert=True)
            return
        forb = {(fp.giver_id, fp.receiver_id) for fp in (await s.execute(
            select(ForbiddenPair).where(ForbiddenPair.room_id == room.id)
        )).scalars().all()}
        participant_ids = [p.id for p in parts]
        pairs = draw_pairs(participant_ids, forb)
        if not pairs:
            await cq.answer("Не удалось провести жеребьёвку. Проверь ограничения.", show_alert=True)
            return
        await s.execute(Pair.__table__.delete().where(Pair.room_id == room.id))
        s.add_all([Pair(room_id=room.id, giver_id=g, receiver_id=r) for g, r in pairs])
        room.drawn = True
        await s.commit()
    await cq.answer("Жеребьёвка готова", show_alert=True)

# Подсказки
@dp.callback_query(F.data.startswith("hint_send:"))
async def cb_hint_send(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one_or_none()
        me = (await s.execute(
            select(Participant).where(Participant.room_id == room.id, Participant.user_id == cq.from_user.id)
        )).scalar_one_or_none() if room else None
        if not (room and me):
            await cq.answer("Нужно присоединиться", show_alert=True)
            return
        pair = (await s.execute(
            select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id)
        )).scalar_one_or_none()
        if not pair:
            await cq.answer("Жеребьёвки ещё не было", show_alert=True)
            return
        since = datetime.now(UTC) - timedelta(days=1)
        cnt = (await s.execute(
            select(func.count()).select_from(Hint).where(
                Hint.room_id == room.id,
                Hint.sender_participant_id == me.id,
                Hint.created_at >= since
            )
        )).scalar()
        if cnt >= MAX_HINTS_PER_DAY:
            await cq.answer("Лимит подсказок на сегодня исчерпан", show_alert=True)
            return
    await cq.message.answer("Напиши подсказку (анонимно).")
    await cq.answer()

@dp.message(F.text & ~F.via_bot & ~F.media_group_id)
async def catch_hint_or_commands(m: Message):
    # Любой обычный текст в комнате после жеребьёвки — считаем подсказкой
    room = await get_user_active_room(m.from_user.id)
    if not room:
        return
    async with Session() as s:
        me = (await s.execute(
            select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id)
        )).scalar_one_or_none()
        if not me:
            return
        pair = (await s.execute(
            select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id)
        )).scalar_one_or_none()
        if not pair:
            return
        recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()
        text_msg = (m.text or "").strip()
        if not text_msg:
            return
        since = datetime.now(UTC) - timedelta(days=1)
        cnt = (await s.execute(
            select(func.count()).select_from(Hint).where(
                Hint.room_id == room.id,
                Hint.sender_participant_id == me.id,
                Hint.created_at >= since
            )
        )).scalar()
        if cnt >= MAX_HINTS_PER_DAY:
            await m.answer("Лимит подсказок на сегодня исчерпан")
            return
        s.add(Hint(
            room_id=room.id,
            sender_participant_id=me.id,
            receiver_participant_id=recv.id,
            text=text_msg[:512]
        ))
        await s.commit()
    with contextlib.suppress(Exception):
        await bot.send_message(recv.user_id, f"🕵️ Тайная подсказка: {text_msg}")
    await m.answer("Подсказка отправлена ✉️")

# Идеи / Покупка (аффилиаты)
@dp.callback_query(F.data.startswith("ideas:"))
async def cb_ideas(cq: CallbackQuery):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        me = (await s.execute(
            select(Participant).where(Participant.room_id == room.id, Participant.user_id == cq.from_user.id)
        )).scalar_one()
        pair = (await s.execute(
            select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id)
        )).scalar_one_or_none()
        if not pair:
            await cq.answer("Жеребьёвки ещё не было", show_alert=True)
            return
        recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()
    q = wishes_to_query(recv.wishes, room.rule_amount_max or room.rule_amount_exact, room.rule_letter)
    await send_single(cq, f"🎁 Идеи для <b>{recv.name}</b> по запросу: <i>{q}</i>\nЖми «🛒 Купить».", main_kb(code, room.owner_id == cq.from_user.id))

@dp.callback_query(F.data.startswith("buy:"))
async def cb_buy(cq: CallbackQuery, state: FSMContext):
    code = cq.data.split(":", 1)[1]
    async with Session() as s:
        room = (await s.execute(select(Room).where(Room.code == code))).scalar_one()
        me = (await s.execute(
            select(Participant).where(Participant.room_id == room.id, Participant.user_id == cq.from_user.id)
        )).scalar_one()
        pair = (await s.execute(
            select(Pair).where(Pair.room_id == room.id, Pair.giver_id == me.id)
        )).scalar_one_or_none()
        if not pair:
            await cq.answer("Жеребьёвки ещё не было", show_alert=True)
            return
        recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()

    # подставляем пользовательский запрос, если есть (если ты уже добавлял BuyPref — ок; если нет, просто закомментируй 3 строки ниже)
    try:
        bp = await get_or_create_buypref(room.id, cq.from_user.id)  # noqa: F821 (должен быть из твоих прошлых правок)
        base_q = bp.custom_query or wishes_to_query(
            recv.wishes, room.rule_amount_max or room.rule_amount_exact, room.rule_letter
        )
        preferred = getattr(bp, "preferred_market", None)
    except NameError:
        bp = None
        base_q = wishes_to_query(recv.wishes, room.rule_amount_max or room.rule_amount_exact, room.rule_letter)
        preferred = None

    if not AFF_TEMPLATES:
        await send_single(cq, "Партнёрские магазины не настроены (ENV AFFILIATES_JSON).", main_kb(code, room.owner_id == cq.from_user.id))
        return

    kb = InlineKeyboardBuilder()
    # Сначала — «любимый» маркет, если выбран
    ordered = list(AFF_TEMPLATES.items())
    if preferred and preferred in AFF_TEMPLATES:
        ordered = [(preferred, AFF_TEMPLATES[preferred])] + [(k, v) for k, v in AFF_TEMPLATES.items() if k != preferred]

    added = 0
    for mk, tpl in ordered:
        url = mk_aff_url(mk, base_q, room_code=code, user_id=cq.from_user.id)
        if url:
            title = HUMAN_NAMES.get(mk, mk)
            star = " ⭐" if preferred == mk else ""
            kb.button(text=f"{title}{star}", url=url)
            added += 1
        if added >= 6:
            break

    # управление (если есть BuyPref)
    if bp is not None:
        kb.button(text="✏️ Изменить запрос", callback_data=f"buy_editq:{code}")
        if "wb" in AFF_TEMPLATES: kb.button(text="⭐ По умолчанию WB", callback_data=f"buy_pref:wb:{code}")
        if "ozon" in AFF_TEMPLATES: kb.button(text="⭐ По умолчанию Ozon", callback_data=f"buy_pref:ozon:{code}")
        if "ym" in AFF_TEMPLATES: kb.button(text="⭐ По умолчанию Я.Маркет", callback_data=f"buy_pref:ym:{code}")

    kb.button(text="↩️ Назад", callback_data=f"room_open:{code}")
    kb.adjust(1)

    text = (
        f"🛒 Поиск: <i>{base_q}</i>\n"
        f"Выбери магазин или измени запрос.\n\n"
        f"<i>Ссылки могут быть партнёрскими. Покупая, вы поддерживаете бота — для вас цена не меняется.</i>"
    )
    await send_single(cq, text, kb.as_markup())

# Выход
@dp.message(F.text == "🚪 Выйти из комнаты")
async def leave_room(m: Message):
    room = await get_user_active_room(m.from_user.id)
    if not room:
        await m.answer("Комнат нет", reply_markup=kb_root(False))
        return
    async with Session() as s:
        me = (await s.execute(
            select(Participant).where(Participant.room_id == room.id, Participant.user_id == m.from_user.id)
        )).scalar_one_or_none()
        if me:
            await s.delete(me)
            await s.commit()
    await m.answer("Вышел из комнаты", reply_markup=kb_root(False))

# ============================================================
# Reminder (ежечасно, мягкие напоминания)
# ============================================================
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
                            with contextlib.suppress(Exception):
                                await bot.send_message(p.user_id, "⏰ Напоминание: скоро обмен подарками!")
        await asyncio.sleep(60)

# ============================================================
# main
# ============================================================
async def main():
    await init_db()
    await migrate_bigints_if_needed()
    asyncio.create_task(reminder_loop())

    if WEBHOOK_URL:
        # WEBHOOK MODE
        from aiogram.webhook.aiohttp_server import SimpleRequestHandler, setup_application
        from aiohttp import web

        app = web.Application()
        SimpleRequestHandler(dispatcher=dp, bot=bot).register(app, path="/webhook")
        setup_application(app, dp, bot=bot)
        await bot.set_webhook(WEBHOOK_URL + "/webhook", drop_pending_updates=True)

        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
        await site.start()

        try:
            while True:
                await asyncio.sleep(3600)
        finally:
            with contextlib.suppress(Exception):
                await runner.cleanup()
            with contextlib.suppress(Exception):
                await bot.session.close()
    else:
        # POLLING MODE + HEALTH
        from aiohttp import web

        info = await bot.get_webhook_info()
        if info.url:
            await bot.delete_webhook(drop_pending_updates=True)

        got = await acquire_runtime_lock()
        if not got:
            print("Another instance already holds the polling lock. Exiting.")
            with contextlib.suppress(Exception):
                await bot.session.close()
            return

        app = web.Application()
        async def _health(_req): return web.Response(text="ok")
        app.router.add_get("/health", _health)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
        await site.start()
        print(f"Polling + health on :{PORT}/health")

        try:
            await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())
        finally:
            with contextlib.suppress(Exception):
                await release_runtime_lock()
            with contextlib.suppress(Exception):
                await runner.cleanup()
            with contextlib.suppress(Exception):
                await bot.session.close()

# ============================================================
# Entrypoint
# ============================================================
if __name__ == "__main__":
    import sys
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        with contextlib.suppress(Exception):
            asyncio.run(bot.session.close())
        sys.exit(0)
