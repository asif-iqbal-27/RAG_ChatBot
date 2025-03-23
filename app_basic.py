from fastapi import FastAPI, HTTPException, Body
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langdetect import detect, LangDetectException  

# Load environment variables
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI app
app = FastAPI()

# Load FAISS vector stores
english_vectorstore = FAISS.load_local("faiss_db/english_faiss.index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
bangla_vectorstore = FAISS.load_local("faiss_db/bangla_faiss.index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Define LLM with constraints to keep responses short
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=100)  # Limit tokens to prevent long answers

# Create conversational retrieval chains
english_qa = ConversationalRetrievalChain.from_llm(llm, retriever=english_vectorstore.as_retriever())
bangla_qa = ConversationalRetrievalChain.from_llm(llm, retriever=bangla_vectorstore.as_retriever())

@app.post("/chat/")
async def chat(query: str = Body(..., embed=True)):  
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    detected_language = detect_language(query)

    # Modify the prompt to ensure the model only answers from the knowledge base and limits the response to three lines
    if detected_language == "en":
        prompt = f"Please answer the following question concisely in no more than three lines. Only answer if the information is in the knowledge base or the book, and refrain from providing external information. If the information is not available, reply with 'Sorry, I don't have information about that topic in my knowledge base.':\n{query}"
        result = english_qa({"question": prompt, "chat_history": []})  # Get result
    elif detected_language == "bn":
        prompt = f"প্রশ্নের উত্তর তিন লাইনের মধ্যে সংক্ষিপ্তভাবে দিন। শুধুমাত্র যদি প্রশ্নের তথ্য জ্ঞানের মধ্যে থাকে, তবেই উত্তর দিন। যদি না থাকে, তবে 'দুঃখিত, আমি এই বিষয়ে কোনও তথ্য জানি না।' এরূপ উত্তর দিন:\n{query}"
        result = bangla_qa({"question": prompt, "chat_history": []})  # Get result
    else:
        raise HTTPException(status_code=400, detail="Unable to detect language properly, please ensure your query is in either English or Bangla.")

    answer = result.get('answer', '').strip()

    # Check if the answer is empty, irrelevant, or indicates a lack of knowledge
    if not answer or "Sorry" in answer or "দুঃখিত" in answer:
        return {"answer": "Sorry, I don't have information about that topic in my knowledge base."}

    # Return the answer from the model
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
