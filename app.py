import os
import asyncio
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties
from aiogram.filters import CommandStart
from aiogram.types import Message, CallbackQuery, InlineKeyboardMarkup
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiohttp import web

BOT_TOKEN = os.environ.get("BOT_TOKEN", "")
PORT = int(os.environ.get("PORT", "8000"))
WEBHOOK_URL = os.environ.get("WEBHOOK_URL")  # оставим на будущее

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is required")

bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()

def main_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="➕ Создать комнату", callback_data="room_new")
    kb.button(text="🔗 Присоединиться", callback_data="room_join")
    kb.adjust(1)
    return kb.as_markup()

@dp.message(CommandStart())
async def cmd_start(m: Message):
    await m.answer("👋 <b>Тайный Санта</b>\nМеню ниже:", reply_markup=main_kb())

@dp.callback_query(F.data == "room_new")
async def cb_room_new(cq: CallbackQuery):
    me = await bot.get_me()
    invite = "https://t.me/{username}?start=room_DEMO".format(username=me.username)
    kb = InlineKeyboardBuilder()
    kb.button(text="🔗 Ссылка для друзей", url=invite)
    kb.button(text="⬅️ Назад", callback_data="home")
    await cq.message.edit_text("✅ Комната (демо) создана.\nЗови друзей по ссылке:", reply_markup=kb.as_markup())
    await cq.answer()

@dp.callback_query(F.data == "room_join")
async def cb_room_join(cq: CallbackQuery):
    kb = InlineKeyboardBuilder()
    kb.button(text="⬅️ Назад", callback_data="home")
    await cq.message.edit_text("Введите код комнаты (демо: room_DEMO) — функционал позже.", reply_markup=kb.as_markup())
    await cq.answer()

@dp.callback_query(F.data == "home")
async def cb_home(cq: CallbackQuery):
    await cq.message.edit_text("Меню:", reply_markup=main_kb())
    await cq.answer()

async def start_http_and_polling():
    # Лёгкий HTTP для Render, чтобы был открытый порт
    app = web.Application()
    app.router.add_get("/health", lambda r: web.Response(text="ok"))
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
    await site.start()
    print(f"Health server on :{PORT}/health")

    # Polling бота
    await bot.delete_webhook(drop_pending_updates=True)
    await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())

if __name__ == "__main__":
    try:
        asyncio.run(start_http_and_polling())
    except (KeyboardInterrupt, SystemExit):
        pass
