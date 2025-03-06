import streamlit as st
import requests

st.header("Language-Specific Chatbot")

prompt = st.text_input("Ask a question:")

backend_url = "http://localhost:8000/chat/"

if prompt:
    response = requests.post(backend_url, json={"query": prompt})
    if response.status_code == 200:
        answer = response.json().get("answer")
        st.write("Answer:", answer)
    else:
        st.write("Error:", response.text)
