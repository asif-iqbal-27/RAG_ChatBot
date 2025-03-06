# RAG Chatbot

This is a **Retrieval-Augmented Generation (RAG) Chatbot** that supports both **English and Bangla** languages. It utilizes **FAISS for vector search**, **Langchain for retrieval-augmented generation**, **FastAPI for backend**, and **Streamlit for frontend**. The chatbot detects the language of the query and retrieves answers accordingly.

## Features
- **Multilingual Support**: Detects and responds in English and Bangla.
- **Retrieval-Augmented Generation (RAG)**: Enhances responses using document retrieval.
- **FAISS Vector Database**: Efficient similarity search for document embeddings.
- **FastAPI Backend**: Handles API requests and processes user queries.
- **Streamlit Frontend**: Simple UI for users to interact with the chatbot.
- **PDF Processing**: Extracts information from books and teacher's guides.
- **Language Detection**: Uses `langdetect` to identify the query language.

## Installation

### 1. Clone the Repository
```sh
git clone https://github.com/your-username/RAG_ChatBot.git
cd RAG_ChatBot
```

### 2. Install Dependencies
```sh
pip install -r requirements.txt
```

### 3. Set Up Environment Variables
Create a `.env` file and add your OpenAI API key:
```
OPENAI_API_KEY=your_openai_api_key
```

### 4. Process PDFs and Generate FAISS Index
```sh
python process_pdfs.py
```

### 5. Start the Backend (FastAPI)
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Start the Frontend (Streamlit)
```sh
streamlit run app.py
```

## Usage
1. Enter a question in either English or Bangla.
2. The chatbot detects the language and retrieves relevant answers.
3. Answers are generated using RAG (Retrieval-Augmented Generation).



## Technologies Used
- FAISS
- FastAPI
- Streamlit
- Langchain
- OpenAI Embeddings
- Sentence-Transformers
- LangDetect
- PyPDF & PDFMiner
- NumPy & Pandas



## Author
Mohammad Asif Iqbal

