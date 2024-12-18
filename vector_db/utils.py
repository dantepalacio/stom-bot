from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

def create_vector_db(file_path: str) -> FAISS:
    loader = TextLoader(file_path, encoding='utf-8')
    documents = loader.load()
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0, separator="***")
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    db.save_local("../faiss_index")
    
    return db


if __name__=='__main__':
    create_vector_db('symptoms.txt')