import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from aiogram.utils.keyboard import InlineKeyboardBuilder

from api_handler import qa
from gsheet import *


from dotenv import load_dotenv
load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
ADMIN_ID = os.environ.get('ADMIN_ID')

bot = Bot(token=BOT_TOKEN)
storage = MemoryStorage()
dp = Dispatcher(storage=storage)

kb = InlineKeyboardBuilder()
kb.add(InlineKeyboardButton(text="Сообщить о проблеме", callback_data="cmd_problem"))
kb.add(InlineKeyboardButton(text="Связаться с администратором", url=f"tg://user?id={ADMIN_ID}"))


class ProblemState(StatesGroup):
    waiting_for_problem = State()

@dp.message(Command("start"))
async def cmd_start(message: types.Message):

    
    await message.answer(
        "Добро пожаловать в стоматологический бот!\n\nВот список доступных команд:\n"
        "/problem - Сообщить о проблеме\n"
        "Связаться с администратором - кнопка ниже",
        reply_markup=kb.as_markup()
    )

# Хендлер для команды /problem
@dp.message(Command("problem"))
async def cmd_problem(message: types.Message, state: FSMContext):
    await message.answer("Опишите вашу проблему. Я передам её специалисту.")
    await state.set_state(ProblemState.waiting_for_problem)

# Хендлер для получения сообщений после /problem
@dp.message(ProblemState.waiting_for_problem)
async def handle_problem(message: types.Message, state: FSMContext):
    user_message = message.text
    
    # Обработка сообщения функцией (здесь будет ваш алгоритм)
    user_id = message.from_user.id
    bot_message, status, summary, recs = process_user_message(user_message, user_id)



    if status == "1":
        await message.answer(recs)
        random_days = random.randint(1, 30)
        random_hour = random.randint(9, 18)  # Рабочие часы (с 9 до 18)
        random_minute = random.randint(0, 59)  # Минуты
        appointment_datetime = datetime.now() + timedelta(days=random_days, hours=random_hour, minutes=random_minute)
        appointment_date = appointment_datetime.strftime("%Y-%m-%d %H:%M")  # Дата и время записи
        contact_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # Дата обращения

        # Логирование в Google Sheets
        log_to_google_sheets(message.from_user.username, contact_date, appointment_date, summary)

        await bot.send_message(
            ADMIN_ID, 
            f"⚠️ Внимание! Триггерное сообщение от клиента @{message.from_user.username}:\n\n{summary}"
        )
    elif status == "-1":
        print('ВСЕ НОРМ')
        await message.answer("Я считаю, что с вами все в порядке, но если это не так, вы также можете обратиться в нашу клинику для консультации!\n\nВот список доступных команд:\n"
        "/problem - Сообщить о проблеме\n"
        "Связаться с администратором - кнопка ниже",
        reply_markup=kb.as_markup())
        await state.clear()

    else:
        await message.answer(bot_message)

# Функция обработки сообщения клиента
def process_user_message(user_message, user_id):
    response = qa(user_query=user_message, user_id=user_id)
    status = response['trigger']
    bot_message = response['bot_message']
    summary = response['summary']
    recs = response['recs']
    return bot_message, status, summary, recs

    # return f"Мы работаем над вашей проблемой: {user_message}"

# Хендлер для кнопки "Сообщить о проблеме"
@dp.callback_query(lambda c: c.data == "cmd_problem")
async def callback_problem(callback_query: types.CallbackQuery, state: FSMContext):
    await callback_query.message.answer("Опишите вашу проблему. Я передам её специалисту.")
    await state.set_state(ProblemState.waiting_for_problem)
    await callback_query.answer()

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
