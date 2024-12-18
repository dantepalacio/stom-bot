import asyncio
import os
import uuid
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

    # Генерация session_id
    session_id = str(uuid.uuid4())
    
    # Устанавливаем состояние
    await state.set_state(ProblemState.waiting_for_problem)

    # Сохраняем session_id в FSMContext
    await state.update_data(session_id=session_id)

    # Проверяем сохранение session_id
    data = await state.get_data()
    print(f"Сохраненный session_id: {data.get('session_id')}")

    await message.answer("Опишите вашу проблему. Я передам её специалисту.")

# Хендлер для получения сообщений после /problem
@dp.message(ProblemState.waiting_for_problem)
async def handle_problem(message: types.Message, state: FSMContext):
    user_message = message.text
    
    # Получаем session_id из состояния
    state_data = await state.get_data()
    session_id = state_data.get("session_id")
    user_id = message.from_user.id

    # Обработка сообщения функцией (здесь будет ваш алгоритм)
    bot_message, status, summary, recs = process_user_message(user_message, user_id, session_id)

    if status == "1":
        await message.answer(recs, reply_markup=kb.as_markup())
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
        await state.clear()
    elif status == "-1":
        await message.answer("Я считаю, что с вами все в порядке, но если это не так, вы также можете обратиться в нашу клинику для консультации!\n\nВот список доступных команд:\n"
                             "/problem - Сообщить о проблеме\n"
                             "Связаться с администратором - кнопка ниже",
                             reply_markup=kb.as_markup())
        await state.clear()  # Сбрасываем состояние, удаляя session_id

    else:
        await message.answer(bot_message)

    # Очистка состояния после завершения
    

# Функция обработки сообщения клиента
def process_user_message(user_message, user_id, session_id):
    response = qa(user_query=user_message, user_id=user_id, session_id=session_id)
    status = response['trigger']
    bot_message = response['bot_message']
    summary = response['summary']
    recs = response['recs']
    return bot_message, status, summary, recs

    # return f"Мы работаем над вашей проблемой: {user_message}"

# Хендлер для кнопки "Сообщить о проблеме"
@dp.callback_query(lambda c: c.data == "cmd_problem")
async def callback_problem(callback_query: types.CallbackQuery, state: FSMContext):
    # Генерация session_id
    session_id = str(uuid.uuid4())
    
    # Устанавливаем состояние
    await state.set_state(ProblemState.waiting_for_problem)
    
    # Сохраняем session_id в FSMContext
    await state.update_data(session_id=session_id)
    
    # Проверяем сохранение session_id
    data = await state.get_data()
    print(f"Сохраненный session_id (через кнопку): {data.get('session_id')}")
    
    await callback_query.message.answer("Опишите вашу проблему. Я передам её специалисту.")
    await callback_query.answer()

# Запуск бота
async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
