import asyncio
if room.rule_amount_max:
rules.append(f"сумма до {room.rule_amount_max}₽")
rules_text = ("
Правило: " + ", ".join(rules)) if rules else ""
for pair in pairs:
giver = (await s.execute(select(Participant).where(Participant.id == pair.giver_id))).scalar_one()
recv = (await s.execute(select(Participant).where(Participant.id == pair.receiver_id))).scalar_one()
try:
await bot.send_message(
giver.user_id,
f"🎄 Твой получатель: <b>{recv.name}</b>{rules_text}
"
f"Хотелки: {recv.wishes or 'не указаны'}"
)
except Exception:
pass


@dp.callback_query(F.data.startswith("export_csv:"))
async def cb_export_csv(cq: CallbackQuery):
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
pairs = (await s.execute(select(Pair).where(Pair.room_id == room.id))).scalars().all()
part_by_id = {p.id: p for p in parts}
buf = io.StringIO()
w = csv.writer(buf)
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
now = datetime.utcnow()
if now.minute == 0: # hourly
async with Session() as s:
rooms = (await s.execute(select(Room).where(Room.drawn == True))).scalars().all() # noqa: E712
for room in rooms:
if room.deadline_at and (room.deadline_at - now) <= timedelta(days=7):
parts = (await s.execute(select(Participant).where(Participant.room_id == room.id))).scalars().all()
for p in parts:
try:
await bot.send_message(p.user_id, "⏰ Напоминание: скоро обмен подарками! Зайди в ‘Получатель’.")
except Exception:
pass
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
runner = web.AppRunner(app)
await runner.setup()
site = web.TCPSite(runner, host="0.0.0.0", port=PORT)
await site.start()
print(f"Webhook listening on :{PORT}{webhook_path}")
while True:
await asyncio.sleep(3600)
else:
await bot.delete_webhook(drop_pending_updates=True)
await dp.start_polling(bot, allowed_updates=dp.resolve_used_update_types())


if __name__ == "__main__":
try:
asyncio.run(main())
except (KeyboardInterrupt, SystemExit):
pass
