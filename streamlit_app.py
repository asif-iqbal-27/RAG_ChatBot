import streamlit as st
import requests

# Page Configuration
st.set_page_config(page_title="AI Chatbot", page_icon="🤖", layout="centered")

st.title("🗣️ RAG Chatbot")
st.write("Ask questions in **English** or **Bangla**, and get concise, accurate answers!")

# Initialize chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Backend URL
backend_url = "http://localhost:8000/chat/"

# Chat Display Section (Scrollable)
chat_container = st.container()
with chat_container:
    st.write("**Chat History**:")
    for chat in st.session_state.chat_history:
        st.chat_message("user").markdown(f"**You:** {chat['question']}")
        st.chat_message("assistant").markdown(f"**Bot:** {chat['answer']}")

# Fixed Input Box at Bottom with Aligned Button
with st.container():
    with st.form(key="chat_form"):
        col1, col2 = st.columns([5, 1])  # Adjust column ratio for better alignment
        with col1:
            prompt = st.text_input("Enter your question:", placeholder="Type your question here...", label_visibility="collapsed")
        with col2:
            ask_button = st.form_submit_button("Ask")

# Handle User Input
if ask_button and prompt:
    # Send query to backend
    payload = {"query": prompt}
    response = requests.post(backend_url, json=payload)

    if response.status_code == 200:
        answer = response.json().get("answer", "No answer received.")
    else:
        answer = "Error connecting to backend!"

    # Display the latest user question and bot response
    with chat_container:
        st.chat_message("user").markdown(f"**You:** {prompt}")
        st.chat_message("assistant").markdown(f"**Bot:** {answer}")

    # Update chat history
    st.session_state.chat_history.append({"question": prompt, "answer": answer})
