from fastapi import FastAPI, HTTPException, Body
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os
from langdetect import detect, LangDetectException

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# Initialize FastAPI app
app = FastAPI()

# Load FAISS vector stores for English and Bangla
english_vectorstore = FAISS.load_local("faiss_db/english_faiss.index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
bangla_vectorstore = FAISS.load_local("faiss_db/bangla_faiss.index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)

# Define LLM with token constraints and conversational memory
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7, max_tokens=100)

# Define memory for tracking past user-bot conversations
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Define standalone question prompt template (for rewriting incomplete questions)
standalone_question_prompt = PromptTemplate(
    input_variables=["question", "chat_history"],
    template="""Based on the following conversation history, convert the latest question into a complete standalone question:

    Chat History: {chat_history}
    Latest Question: {question}

    Standalone Question:"""
)

# Create conversational retrieval chains with memory
english_qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=english_vectorstore.as_retriever(), memory=memory
)
bangla_qa = ConversationalRetrievalChain.from_llm(
    llm, retriever=bangla_vectorstore.as_retriever(), memory=memory
)

@app.post("/chat/")
async def chat(query: str = Body(..., embed=True)):
    if not query:
        raise HTTPException(status_code=400, detail="Query is required")

    detected_language = detect_language(query)

    # Retrieve chat history from memory (this allows history-awareness)
    past_chat_history = memory.load_memory_variables({}).get("chat_history", "")

    # Rewrite incomplete question as a standalone question
    standalone_question = llm.predict(standalone_question_prompt.format(
        question=query, chat_history=past_chat_history
    )).strip()

    # Generate language-based prompts and pass chat history explicitly
    if detected_language == "en":
        prompt = f"Please answer the following question concisely in no more than three lines. Only answer if the information is in the knowledge base or the book, and refrain from providing external information. If the information is not available, reply with 'Sorry, I don't have information about that topic in my knowledge base.':\n{standalone_question}"
        result = english_qa({"question": prompt, "chat_history": past_chat_history})  # Pass history here
    elif detected_language == "bn":
        prompt = f"প্রশ্নের উত্তর তিন লাইনের মধ্যে সংক্ষিপ্তভাবে দিন। শুধুমাত্র যদি প্রশ্নের তথ্য জ্ঞানের মধ্যে থাকে, তবেই উত্তর দিন। যদি না থাকে, তবে 'দুঃখিত, আমি এই বিষয়ে কোনও তথ্য জানি না।' এরূপ উত্তর দিন:\n{standalone_question}"
        result = bangla_qa({"question": prompt, "chat_history": past_chat_history})  # Pass history here
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
        raise HTTPException(status_code=500, detail=f"Error detecting language: {str(e)}")
