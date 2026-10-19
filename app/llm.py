from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY, MODEL_NAME

llm = ChatGroq(
    model=MODEL_NAME,
    groq_api_key=GROQ_API_KEY,
    temperature=0,
)
