import os
import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Load .env file with OPENAI_API_KEY
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="📄 GenAI PDF Assistant", layout="wide")
st.title("📄 GenAI PDF Assistant (LangChain + Streamlit)")

# PDF text extraction
def extract_text_from_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Vectorstore + QA chain setup
def process_pdf_and_create_chain(text):
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", openai_api_key=openai_key)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

# Upload and process
uploaded_file = st.file_uploader("📤 Upload your PDF", type=["pdf"])

if uploaded_file:
    with st.spinner("⏳ Reading and indexing your PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_file)
        qa_chain = process_pdf_and_create_chain(pdf_text)
        st.success("✅ PDF processed successfully!")

        # Ask a question
        query = st.text_input("💬 Ask a question based on this PDF:")
        if query:
            with st.spinner("🤖 Generating answer..."):
                answer = qa_chain.run(query)
                st.markdown("### 📢 Answer:")
                st.write(answer)
