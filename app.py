from fastapi import FastAPI, HTTPException, Body
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langdetect import detect, LangDetectException  

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

app = FastAPI()

english_vectorstore = FAISS.load_local("faiss_db/english_faiss.index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
bangla_vectorstore = FAISS.load_local("faiss_db/bangla_faiss.index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

english_qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever=english_vectorstore.as_retriever())
bangla_qa = ConversationalRetrievalChain.from_llm(OpenAI(), retriever=bangla_vectorstore.as_retriever())

@app.post("/chat/")
async def chat(query: str = Body(..., embed=True)):  
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    detected_language = detect_language(query)

    if detected_language == "en":
        answer = english_qa({"question": query, "chat_history": []})['answer']
    elif detected_language == "bn":
        answer = bangla_qa({"question": query, "chat_history": []})['answer']
    else:
        raise HTTPException(status_code=400, detail="Unable to detect language properly, please ensure your query is in either English or Bangla.")

    return {"answer": answer}

def detect_language(query: str):
    try:
        lang = detect(query)
        if lang == 'en':
            return "en"  
        elif lang == 'bn':
            return "bn"  
        else:
            return "unknown"  
    except LangDetectException:
        return "unknown"  
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")
