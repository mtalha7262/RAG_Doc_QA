import os
import streamlit as st
from multipart import file_path

from doc_model import get_answer

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Chat With Document",
    page_icon="📄",
    layout="centered"

)

st.title("Document Q&A  -  Llama 3 - Ollama ")

upload_file = st.file_uploader(label="Upload your file", type=["pdf"])

user_query = st.text_input("Ask your Question")

if st.button("Run"):
    bytes_data = upload_file.read()
    file_name = upload_file.name
    # Save file at working directory
    file_path = os.path.join(working_dir, file_name)

    with open(file_path, "wb") as f:
        f.write(bytes_data)

    # Pass file_path to get_answer
    answer = get_answer(file_path, user_query)

    st.success(answer)