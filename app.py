from fastapi import FastAPI, HTTPException, Body
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langdetect import detect, LangDetectException  # Import LangDetectException to handle detection errors

# Load environment variables from .env file
load_dotenv()

# Set OpenAI API Key from environment variables
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI
app = FastAPI()

# Load FAISS vector stores
english_vectorstore = FAISS.load_local("faiss_db/english_faiss.index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
bangla_vectorstore = FAISS.load_local("faiss_db/bangla_faiss.index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Load ConversationalRetrievalChain for both languages
english_qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever=english_vectorstore.as_retriever())
bangla_qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever=bangla_vectorstore.as_retriever())

@app.post("/chat/")
async def chat(query: str = Body(..., embed=True)):  # query is required and passed via the body
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    # Detect language using langdetect
    detected_language = detect_language(query)

    if detected_language == "en":
        answer = english_qa({"question": query, "chat_history": []})['answer']
    elif detected_language == "bn":
        answer = bangla_qa({"question": query, "chat_history": []})['answer']
    else:
        # Fallback or ask the user to specify the language
        raise HTTPException(status_code=400, detail="Unable to detect language properly, please ensure your query is in either English or Bangla.")

    return {"answer": answer}

# Improved language detection function using langdetect library with fallback
def detect_language(query: str):
    try:
        lang = detect(query)
        if lang == 'en':
            return "en"  # English
        elif lang == 'bn':
            return "bn"  # Bangla
        else:
            return "unknown"  # In case we can't detect the language accurately
    except LangDetectException:
        return "unknown"  # If there's an issue with detection, treat it as unknown
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")
