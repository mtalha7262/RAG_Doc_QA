import os

from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.document_loaders import UnstructuredFileLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
# from huggingface_hub import hf_hub_download
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import retrieval_qa
from langchain.document_loaders import PyMuPDFLoader


working_dir = os.path.dirname(os.path.abspath(r"C:\Users\Muhammad_Talha\PycharmProjects\RAG_doc_QA\venv\src"))

llm = Ollama(
    model = "llama3:instruct",
    temperature=0
)


embeddings = HuggingFaceEmbeddings()

def get_answer(file_path, query):
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    text_splitter = CharacterTextSplitter(separator="\n",  # Fix typo: use "\n" instead of "/n"
                                          chunk_size=1000,
                                          chunk_overlap=200)
    text_chunks = text_splitter.split_documents(documents)

    knowledge_base = FAISS.from_documents(text_chunks, embeddings)

    qa_chain = RetrievalQA.from_chain_type(
        llm,
        retriever=knowledge_base.as_retriever()  # Fix typo: use retriever instead of retrival
    )

    response = qa_chain.invoke({"query": query})

    return response["result"]
