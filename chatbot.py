import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS

load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def process_pdf(pdf_paths, language="english"):
    all_docs = []
    
    for pdf_path in pdf_paths:
        loader = PyPDFLoader(file_path=pdf_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""])
        split_docs = text_splitter.split_documents(docs)
        
        all_docs.extend(split_docs)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(all_docs, embeddings)
    
    vector_file = f"faiss_db/{language}_faiss.index"
    vectorstore.save_local(vector_file)
    print(f"FAISS vectors saved for {language} at {vector_file}")

def process_pdfs():
    english_pdfs = ['pdfs/english_book.pdf', 'pdfs/english_teachers_guide.pdf']
    bangla_pdfs = ['pdfs/bangla_book.pdf', 'pdfs/bangla_teachers_guide.pdf']

    print("Processing English PDFs...")
    process_pdf(english_pdfs, language="english")

    print("Processing Bangla PDFs...")
    process_pdf(bangla_pdfs, language="bangla")
process_pdfs()