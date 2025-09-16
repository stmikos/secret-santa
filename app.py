# app.py
import os
import json
import random
from typing import List, Tuple, Optional
from datetime import datetime, timezone

import asyncpg
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel

from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandObject
from aiogram.fsm.state import StatesGroup, State
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.types import (
    Update, Message, CallbackQuery,
    ReplyKeyboardMarkup, KeyboardButton,
    InlineKeyboardMarkup, InlineKeyboardButton
)

# -------- ENV --------
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
BASE_URL = os.getenv("BASE_URL", "")
WEBHOOK_SECRET = os.getenv("WEBHOOK_SECRET", "secret123")
WEBHOOK_PATH = f"/telegram/webhook/{WEBHOOK_SECRET}"
WEBHOOK_URL = (BASE_URL + WEBHOOK_PATH) if BASE_URL else None

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")

# -------- Globals --------
bot = Bot(BOT_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(storage=MemoryStorage())
app = FastAPI()
DB_POOL: asyncpg.Pool = None

# -------- UI Templates --------
MSG_WELCOME = (
    "🎅 Привет! Я — бот Тайный Санта.\n"
    "Создавай комнаты, собирай друзей, делай жеребьёвку и дарите радость.\n\n"
    "Выбери действие ниже 👇"
)

MSG_ROOM_CREATED = (
    "✨ Комната создана!\n"
    "<b>{title}</b>\n"
    "Бюджет: {budget}\n"
    "Регистрация до: {join_until}\n\n"
    "Пригласительная ссылка:\n<code>{join_link}</code>\n\n"
    "Или команда: <code>/join {room_id}</code>"
)

MSG_JOIN_OK = "Готово! Ты вступил в <b>{title}</b>. Хотелки записал: {wcount}."
MSG_DRAW_DONE = "🎲 Жеребьёвка проведена! Всем отправил, проверь ЛС."
MSG_UNDO_OK = "↩️ Последняя жеребьёвка отменена. Предыдущая активна (если была)."
MSG_YOU_GIFT_TO = (
    "🎁 Ты — Санта для <b>{name}</b>!\n"
    "Хотелки: {wishlist}\n"
    "Анти: {anti}\n"
    "{addr}"
)
MSG_ANON_SENT = "🕵️ Анонимное сообщение отправлено."
MSG_NOT_ORG = "Только организатор комнаты может это сделать."
MSG_NO_ROOM = "Комната не найдена."

MAIN_KB = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="➕ Создать игру"), KeyboardButton(text="🎁 Мои игры")],
        [KeyboardButton(text="📝 Мои хотелки"), KeyboardButton(text="🕵️ Анонимное сообщение")],
        [KeyboardButton(text="👑 Премиум"), KeyboardButton(text="❓ Помощь")],
    ],
    resize_keyboard=True
)

# -------- States --------
class CreateRoomState(StatesGroup):
    title = State()
    budget = State()
    join_until = State()

class JoinState(StatesGroup):
    room_id = State()
    name = State()
    wishlist = State()
    anti = State()

class AnonMessageState(StatesGroup):
    room_id = State()
    text = State()

# -------- Helpers --------
def format_ts(dt: Optional[datetime]) -> str:
    if not dt:
        return "—"
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

def room_inline_kb(room_id: str, owner: bool, has_prev_draw: bool) -> InlineKeyboardMarkup:
    btns = [
        [InlineKeyboardButton(text="👥 Участники", callback_data=f"room_users:{room_id}")],
        [InlineKeyboardButton(text="✍️ Редактировать хотелки", callback_data=f"room_editwl:{room_id}")],
        [InlineKeyboardButton(text="🚚 Отметить отправку", callback_data=f"room_ship:{room_id}")],
    ]
    if owner:
        owner_row = [InlineKeyboardButton(text="🎲 Жеребьёвка", callback_data=f"room_draw:{room_id}")]
        if has_prev_draw:
            owner_row.append(InlineKeyboardButton(text="↩️ Отменить", callback_data=f"room_undo:{room_id}"))
        btns.insert(0, owner_row)
        btns.append([InlineKeyboardButton(text="📊 Прогресс", callback_data=f"room_progress:{room_id}")])
        btns.append([InlineKeyboardButton(text="⚙️ Настройки", callback_data=f"room_settings:{room_id}")])
        btns.append([InlineKeyboardButton(text="💳 Апгрейд до PRO", callback_data=f"room_pro:{room_id}")])
    return InlineKeyboardMarkup(inline_keyboard=btns)

# Track bot messages in FSM to delete later (reduce clutter)
async def add_cleanup_msg(state, msg: Message):
    data = await state.get_data() if state else {}
    ids = data.get("cleanup_ids", [])
    ids.append(msg.message_id)
    if state:
        await state.update_data(cleanup_ids=ids)

async def do_cleanup(m: Message, state):
    if not state:
        return
    data = await state.get_data()
    ids = data.get("cleanup_ids", [])
    for mid in ids:
        try:
            await bot.delete_message(m.chat.id, mid)
        except:
            pass
    await state.update_data(cleanup_ids=[])

# -------- DB helpers --------
async def db() -> asyncpg.Connection:
    return await DB_POOL.acquire()

async def fetchrow(q, *a):
    con = await db()
    try:
        return await con.fetchrow(q, *a)
    finally:
        await DB_POOL.release(con)

async def fetch(q, *a):
    con = await db()
    try:
        return await con.fetch(q, *a)
    finally:
        await DB_POOL.release(con)

async def execute(q, *a):
    con = await db()
    try:
        return await con.execute(q, *a)
    finally:
        await DB_POOL.release(con)

# -------- Matching --------
def draw_pairs(user_ids: List[int], forbidden: List[Tuple[int, int]]) -> Optional[List[Tuple[int, int]]]:
    n = len(user_ids)
    if n < 2:
        return None
    forb = set(forbidden)
    for _ in range(200):
        receivers = user_ids[:]
        random.shuffle(receivers)
        # avoid self
        for i in range(n):
            if receivers[i] == user_ids[i]:
                j = (i + 1) % n
                receivers[i], receivers[j] = receivers[j], receivers[i]
        pairs = list(zip(user_ids, receivers))
        # check forb
        if any((g, r) in forb for g, r in pairs):
            continue
        # break 2-cycles
        pos = {u: i for i, u in enumerate(user_ids)}
        bad = []
        for i, (g, r) in enumerate(pairs):
            ri = pos.get(r)
            if ri is not None and receivers[ri] == g:
                bad.append(i)
        if bad:
            tmp = [pairs[i][1] for i in bad]
            random.shuffle(tmp)
            for k, i in enumerate(bad):
                receivers[i] = tmp[k]
            pairs = list(zip(user_ids, receivers))
        # final checks
        if any(g == r for g, r in pairs):
            continue
        if any((g, r) in forb for g, r in pairs):
            continue
        # re-check 2-cycles
        ok = True
        pos = {u: i for i, u in enumerate(user_ids)}
        for g, r in pairs:
            ri = pos.get(r)
            if ri is not None and receivers[ri] == g:
                ok = False; break
        if ok:
            return pairs
    return None

# -------- Handlers --------
@dp.message(Command("start"))
async def cmd_start(m: Message):
    sent = await m.answer(MSG_WELCOME, reply_markup=MAIN_KB)
    # We don't track cleanup for /start to keep one anchor message

@dp.message(F.text == "❓ Помощь")
async def help_msg(m: Message):
    await m.answer("Команды: /create — создать игру, /join <id> — вступить. Меню — кнопки ниже.", reply_markup=MAIN_KB)

@dp.message(F.text == "➕ Создать игру")
@dp.message(Command("create"))
async def create_room_start(m: Message, state):
    await state.set_state(CreateRoomState.title)
    q = await m.answer("Как назовём комнату? (например, «Santa 2025 — Маркетинг»)")
    await add_cleanup_msg(state, q)

@dp.message(CreateRoomState.title)
async def create_room_title(m: Message, state):
    await state.update_data(title=m.text.strip()[:120])
    await state.set_state(CreateRoomState.budget)
    q = await m.answer("Бюджет? Введи диапазон: 10-20 (в у.е.) или одно число.")
    await add_cleanup_msg(state, q)

@dp.message(CreateRoomState.budget)
async def create_room_budget(m: Message, state):
    txt = m.text.strip()
    minb, maxb = None, None
    try:
        if "-" in txt:
            a, b = txt.split("-", 1)
            minb, maxb = int(a), int(b)
        else:
            minb = int(txt)
            maxb = int(txt)
    except:
        q = await m.answer("Не понял бюджет. Пример: 10-20")
        await add_cleanup_msg(state, q)
        return
    await state.update_data(budget_min=minb, budget_max=maxb)
    await state.set_state(CreateRoomState.join_until)
    q = await m.answer("Дедлайн регистрации (UTC), формат: 2025-12-10 18:00 (или «нет»).")
    await add_cleanup_msg(state, q)

@dp.message(CreateRoomState.join_until)
async def create_room_join_until(m: Message, state):
    data = await state.get_data()
    join_until = None
    if m.text.strip().lower() != "нет":
        try:
            join_until = datetime.fromisoformat(m.text.strip()).replace(tzinfo=timezone.utc)
        except:
            q = await m.answer("Формат даты неверен. Пример: 2025-12-10 18:00")
            await add_cleanup_msg(state, q)
            return
    row = await fetchrow(
        """insert into rooms(owner_tg_id, title, budget_min, budget_max, join_until)
           values($1,$2,$3,$4,$5)
           returning id, title, budget_min, budget_max, join_until""",
        m.from_user.id, data["title"], data["budget_min"], data["budget_max"], join_until
    )
    me = await bot.me()
    join_link = f"https://t.me/{me.username}?start=join_{row['id']}"
    budget = f"{row['budget_min']}-{row['budget_max']}" if row['budget_min'] != row['budget_max'] else f"{row['budget_min']}"
    has_prev_draw = False
    await m.answer(
        MSG_ROOM_CREATED.format(
            title=row["title"],
            budget=budget,
            join_until=format_ts(row["join_until"]),
            join_link=join_link,
            room_id=row["id"]
        ),
        reply_markup=room_inline_kb(str(row["id"]), owner=True, has_prev_draw=has_prev_draw)
    )
    await do_cleanup(m, state)
    await state.clear()

@dp.message(F.text.startswith("🎁 Мои игры"))
async def my_rooms(m: Message):
    rows = await fetch("select id, title, status, created_at from rooms where owner_tg_id=$1 order by created_at desc limit 10", m.from_user.id)
    if not rows:
        await m.answer("У тебя пока нет созданных комнат. Нажми «Создать игру».")
        return
    text = "Твои комнаты:\n" + "\n".join([f"• <code>{r['id']}</code> — {r['title']} ({r['status']})" for r in rows])
    await m.answer(text)

@dp.message(Command("join"))
async def join_cmd(m: Message, command: CommandObject, state):
    if not command.args:
        await m.answer("Укажи ID комнаты: /join <id>")
        return
    await state.set_state(JoinState.room_id)
    await state.update_data(room_id=command.args.strip())
    await state.set_state(JoinState.name)
    q = await m.answer("Как тебя подписать в этой игре? (Имя/ник)")
    await add_cleanup_msg(state, q)

@dp.message(F.text.startswith("/start join_"))
async def deep_join(m: Message, state):
    room_id = m.text.split("join_", 1)[1].strip()
    await state.set_state(JoinState.room_id)
    await state.update_data(room_id=room_id)
    await state.set_state(JoinState.name)
    q = await m.answer("Как тебя подписать в этой игре? (Имя/ник)")
    await add_cleanup_msg(state, q)

@dp.message(JoinState.name)
async def join_name(m: Message, state):
    await state.update_data(name=m.text.strip()[:60])
    await state.set_state(JoinState.wishlist)
    q = await m.answer("Супер. Напиши хотелки через запятую (до 5). Пример: кофе, настолки, тёплые носки")
    await add_cleanup_msg(state, q)

@dp.message(JoinState.wishlist)
async def join_wishlist(m: Message, state):
    wl = [x.strip()[:40] for x in m.text.split(",") if x.strip()]
    wl = wl[:5]
    await state.update_data(wishlist=wl)
    await state.set_state(JoinState.anti)
    q = await m.answer("Антыхотелки? (что НЕ дарить) — тоже через запятую. Или «нет».")
    await add_cleanup_msg(state, q)

@dp.message(JoinState.anti)
async def join_anti(m: Message, state):
    anti = []
    if m.text.strip().lower() != "нет":
        anti = [x.strip()[:40] for x in m.text.split(",") if x.strip()]
        anti = anti[:5]
    data = await state.get_data()
    room = await fetchrow("select id, title from rooms where id=$1", data["room_id"])
    if not room:
        await m.answer(MSG_NO_ROOM)
        await do_cleanup(m, state)
        await state.clear()
        return
    await execute("""
        insert into participants(room_id, user_tg_id, name, wishlist, anti)
        values($1,$2,$3,$4,$5)
        on conflict (room_id, user_tg_id) do update
        set name=excluded.name, wishlist=excluded.wishlist, anti=excluded.anti
    """, data["room_id"], m.from_user.id, data["name"], data["wishlist"], anti)
    await m.answer(MSG_JOIN_OK.format(title=room["title"], wcount=len(data["wishlist"])))
    await do_cleanup(m, state)
    await state.clear()

@dp.message(F.text == "🕵️ Анонимное сообщение")
async def anon_start(m: Message, state):
    await state.set_state(AnonMessageState.room_id)
    q = await m.answer("Введи ID комнаты (из /join или /create).")
    await add_cleanup_msg(state, q)

@dp.message(AnonMessageState.room_id)
async def anon_room(m: Message, state):
    rid = m.text.strip()
    pair = await fetchrow("select receiver_tg_id from pairs where room_id=$1 and giver_tg_id=$2 order by created_at desc limit 1", rid, m.from_user.id)
    if not pair:
        await m.answer("Не нашёл связку для этой комнаты. Либо нет жеребьёвки, либо ты не участник.")
        await do_cleanup(m, state)
        await state.clear()
        return
    await state.update_data(room_id=rid, to_tg_id=pair["receiver_tg_id"])
    await state.set_state(AnonMessageState.text)
    q = await m.answer("Напиши текст анонимного сообщения получателю:")
    await add_cleanup_msg(state, q)

@dp.message(AnonMessageState.text)
async def anon_text(m: Message, state):
    data = await state.get_data()
    text = m.text.strip()
    await execute("""
        insert into messages_anon(room_id, from_tg_id, to_tg_id, text)
        values($1,$2,$3,$4)
    """, data["room_id"], m.from_user.id, data["to_tg_id"], text)
    try:
        await bot.send_message(data["to_tg_id"], f"🕵️ Тебе анонимное сообщение:\n\n{text}")
    except Exception:
        pass
    await m.answer(MSG_ANON_SENT)
    await do_cleanup(m, state)
    await state.clear()

# -------- Callbacks --------
def owner_has_prev_draw(draw_count: int) -> bool:
    return draw_count > 0

@dp.callback_query(F.data.startswith("room_draw:"))
async def cb_room_draw(c: CallbackQuery):
    room_id = c.data.split(":", 1)[1]
    room = await fetchrow("select id, owner_tg_id, title from rooms where id=$1", room_id)
    if not room:
        await c.answer("Комната не найдена", show_alert=True); return
    if room["owner_tg_id"] != c.from_user.id:
        await c.answer(MSG_NOT_ORG, show_alert=True); return
    users = await fetch("select user_tg_id from participants where room_id=$1", room_id)
    user_ids = [r["user_tg_id"] for r in users]
    if len(user_ids) < 2:
        await c.answer("Нужно минимум 2 участника", show_alert=True); return
    ex = await fetch("select giver_tg_id, receiver_tg_id from exclusions where room_id=$1", room_id)
    forbidden = [(r["giver_tg_id"], r["receiver_tg_id"]) for r in ex]
    pairs = draw_pairs(user_ids, forbidden)
    if not pairs:
        await c.answer("Не удалось построить пары. Добавь участников/измени исключения.", show_alert=True); return
    # create new draw
    draw_row = await fetchrow("insert into draws(room_id) values($1) returning id, created_at", room_id)
    draw_id = draw_row["id"]
    for g, rcv in pairs:
        await execute("insert into pairs(room_id, draw_id, giver_tg_id, receiver_tg_id) values($1,$2,$3,$4)", room_id, draw_id, g, rcv)
    # notify participants
    for g, rcv in pairs:
        rec = await fetchrow("""
            select name, wishlist, anti, address_json from participants
            where room_id=$1 and user_tg_id=$2
        """, room_id, rcv)
        wl = ", ".join(rec["wishlist"]) if rec and rec["wishlist"] else "—"
        anti = ", ".join(rec["anti"]) if rec and rec["anti"] else "—"
        addr = ""
        if rec and rec["address_json"] and rec["address_json"] != {}:
            addr = "📦 Доставка: " + json.dumps(rec["address_json"], ensure_ascii=False)
        try:
            await bot.send_message(g, MSG_YOU_GIFT_TO.format(name=rec["name"], wishlist=wl, anti=anti, addr=addr))
        except Exception:
            pass
    # count previous draws
    cnt = await fetchrow("select count(*) as c from draws where room_id=$1", room_id)
    has_prev = owner_has_prev_draw(cnt["c"] > 1)
    await c.message.edit_text(MSG_DRAW_DONE, reply_markup=room_inline_kb(room_id, owner=True, has_prev_draw=has_prev))
    await c.answer()

@dp.callback_query(F.data.startswith("room_undo:"))
async def cb_room_undo(c: CallbackQuery):
    room_id = c.data.split(":", 1)[1]
    room = await fetchrow("select id, owner_tg_id from rooms where id=$1", room_id)
    if not room:
        await c.answer("Комната не найдена", show_alert=True); return
    if room["owner_tg_id"] != c.from_user.id:
        await c.answer(MSG_NOT_ORG, show_alert=True); return
    # find latest draw
    last_draw = await fetchrow("select id from draws where room_id=$1 order by created_at desc limit 1", room_id)
    if not last_draw:
        await c.answer("Пока нет жеребьёвок", show_alert=True); return
    # delete pairs of latest draw and the draw record
    await execute("delete from pairs where room_id=$1 and draw_id=$2", room_id, last_draw["id"])
    await execute("delete from draws where id=$1", last_draw["id"])
    # check if any previous remains
    prev_exists = await fetchrow("select id from draws where room_id=$1 limit 1", room_id)
    has_prev = owner_has_prev_draw(prev_exists is not None)
    await c.message.edit_text(MSG_UNDO_OK, reply_markup=room_inline_kb(room_id, owner=True, has_prev_draw=has_prev))
    await c.answer()

@dp.callback_query(F.data.startswith("room_users:"))
async def cb_room_users(c: CallbackQuery):
    room_id = c.data.split(":", 1)[1]
    rows = await fetch("select name from participants where room_id=$1 order by joined_at", room_id)
    if not rows:
        await c.answer("Пока пусто", show_alert=True); return
    names = ", ".join([r["name"] for r in rows])
    await c.message.answer(f"👥 Участники: {names}")
    await c.answer()

@dp.callback_query(F.data.startswith("room_progress:"))
async def cb_room_progress(c: CallbackQuery):
    await c.answer("Скоро тут будет дешборд 📊", show_alert=True)

@dp.callback_query(F.data.startswith("room_settings:"))
async def cb_room_settings(c: CallbackQuery):
    await c.answer("Настройки в разработке ⚙️", show_alert=True)

@dp.callback_query(F.data.startswith("room_pro:"))
async def cb_room_pro(c: CallbackQuery):
    await c.answer("PRO доступ скоро. Цена: €4.99/комната 👑", show_alert=True)

# -------- FastAPI Webhook --------
class TelegramUpdate(BaseModel):
    update_id: int

@app.get("/")
async def root():
    return {"ok": True, "app": "secret-santa-bot"}

@app.post(WEBHOOK_PATH)
async def telegram_webhook(request: Request):
    if WEBHOOK_URL is None:
        raise HTTPException(status_code=400, detail="Webhook URL not configured")
    body = await request.json()
    update = Update.model_validate(body)
    await dp.feed_update(bot, update)
    return {"ok": True}

# -------- Startup / Shutdown --------
@app.on_event("startup")
async def on_startup():
    global DB_POOL
    DB_POOL = await asyncpg.create_pool(DATABASE_URL, max_size=10)
    if WEBHOOK_URL:
        try:
            await bot.set_webhook(WEBHOOK_URL, secret_token=WEBHOOK_SECRET, drop_pending_updates=True)
        except Exception as e:
            print("Webhook set failed:", e)

@app.on_event("shutdown")
async def on_shutdown():
    global DB_POOL
    if DB_POOL:
        await DB_POOL.close()
    try:
        await bot.delete_webhook(drop_pending_updates=False)
    except:
        pass
