import streamlit as st
import requests

st.header("Language-Specific Chatbot")

# Input field for user query
prompt = st.text_input("Ask a question:")

# Define backend URL
backend_url = "http://localhost:8000/chat/"

# Process input query
if prompt:
    response = requests.post(backend_url, json={"query": prompt})
    if response.status_code == 200:
        answer = response.json().get("answer")
        st.write("Answer:", answer)
    else:
        st.write("Error:", response.text)
