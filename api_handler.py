import os
import json
import openai

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import MessagesPlaceholder

from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain.prompts.chat import ChatPromptTemplate

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.globals import set_verbose
from langchain.schema import ChatMessage, BaseChatMessageHistory


set_verbose(True)

from dotenv import load_dotenv

from prompts import DENTIST_PROMPT, CLASSIFICATION_PROMPT, SUMMARY_PROMPT, RECOMMENDATION_PROMPT

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
REDIS_URL=os.environ.get('REDIS_URL')


openai.api_key = OPENAI_API_KEY
client = openai.OpenAI()


# Путь к файлу для хранения истории
CHAT_HISTORY_FILE = "chat_history.json"

# Функция для загрузки истории чата из JSON
def load_chat_history() -> dict:
    if os.path.exists(CHAT_HISTORY_FILE):
        print('store exists')
        with open(CHAT_HISTORY_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    return {}

def get_session_messages(session_id: str) -> list:
    """
    Возвращает историю сообщений для session_id в виде списка словарей.
    """
    session_id = str(session_id)
    store = load_chat_history()

    # Если сессия отсутствует, возвращаем пустой список
    if session_id not in store:
        print('FUUUUUCK NOT IN STORE')
        return []

    # Возвращаем историю сообщений
    return store[session_id]
# Функция для сохранения истории чата в JSON
def save_chat_history(store: dict):
    with open(CHAT_HISTORY_FILE, "w", encoding="utf-8") as file:
        json.dump(store, file, ensure_ascii=False, indent=4)

# Универсальная функция для управления историей чата
def get_session_history(session_id: str, new_message=None) -> BaseChatMessageHistory:
    """
    Возвращает объект ChatMessageHistory для session_id.
    Если передан параметр new_message, обновляет историю в JSON.
    """
    session_id = str(session_id)
    # Загружаем историю из JSON
    store = load_chat_history()

    # Если сессия отсутствует, создаём новую
    if session_id not in store:
        store[session_id] = []

    # Добавляем новое сообщение, если передано
    if new_message:
        store[session_id].append({"role": new_message.role, "content": new_message.content})
        save_chat_history(store)  # Сохраняем изменения

    # Возвращаем объект ChatMessageHistory
    history = ChatMessageHistory(messages=[
        ChatMessage(role=message["role"], content=message["content"])
        for message in store[session_id]
    ])
    
    return history

def format_chat_history(messages: list) -> str:
    """
    Форматирует историю чата в виде строки с ролями 'Клиент' и 'Стоматолог'.

    :param messages: Список сообщений с ключами 'role' и 'content'.
    :return: История в виде отформатированной строки.
    """
    role_map = {
        "user": "Клиент",
        "assistant": "Стоматолог"
    }
    formatted_history = []
    for message in messages:
        role = role_map.get(message["role"], "Неизвестный")
        content = message["content"]
        formatted_history.append(f"{role}: {content}")
    return "\n\n".join(formatted_history)
##

def classify_patient_answer(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": DENTIST_PROMPT},
            {"role": "user", "content": f'Ответ пациента: {text}'}
        ],
        temperature=0.0,
        max_tokens=512,
    )
    
    answer = response.choices[0].message.content
    return answer

def summary_history(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f'История разговора: {text}'}
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    
    answer = response.choices[0].message.content
    return answer

def give_recomendations(text):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": RECOMMENDATION_PROMPT},
            {"role": "user", "content": f'История разговора: {text}'}
        ],
        temperature=0.7,
        max_tokens=2048,
    )
    
    answer = response.choices[0].message.content
    return answer

def binary_classify(text:str) -> json:
    '''Функция для генерации уточняющих вопрос пациенту
        text: Результат векторного поиска по самочувствию пациента
    '''
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": CLASSIFICATION_PROMPT},
            {"role": "user", "content": f'История разговора: {text}'}
        ],
        temperature=0.0,
        max_tokens=2048,
    )
    
    answer = response.choices[0].message.content
    return answer


def get_chain(retriever):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4
    )

    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", DENTIST_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,  # Используем единую функцию
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain

def get_retrievers(pages):
    embeddings = OpenAIEmbeddings()

    bm25_retriever = BM25Retriever.from_texts(pages)
    bm25_retriever.k = 5

    faiss_vectorstore = FAISS.from_texts(pages, embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever],weights=[0.6, 0.4])

    return ensemble_retriever

def qa(user_query, user_id):
    embeddings = OpenAIEmbeddings()
    db_vector = FAISS.load_local(r"faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db_vector.as_retriever()
    chat_chain = get_chain(retriever)

    # Добавляем запрос пользователя в историю
    get_session_history(user_id, new_message=ChatMessage(role="user", content=user_query))

    # Выполнение цепочки
    response = chat_chain.invoke(
        {"input": user_query},
        config={
            "configurable": {"session_id": user_id}
        },
    )

    # Добавляем ответ бота в историю
    bot_message = response["answer"]
    get_session_history(user_id, new_message=ChatMessage(role="assistant", content=bot_message))

    # Логируем обновленную историю чата
    chat_history = get_session_messages(user_id)
    print(f'{type(chat_history)} {chat_history}')
    updated_chat_history = format_chat_history(chat_history[-4:])
    print("\n\n\nФорматированная история", updated_chat_history)

    # Классификация ответа
    status_answer = binary_classify(updated_chat_history)

    if status_answer == '1':
        summary = summary_history(updated_chat_history)
        recs = give_recomendations(updated_chat_history)
    else:
        summary = None
        recs = None

    result = {'bot_message': bot_message, 'trigger': status_answer, 'summary':summary, 'recs':recs}
    return result

if __name__=='__main__':
    while True:
        user = input('text here: ')
        print(qa(user))