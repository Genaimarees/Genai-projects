import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import requests
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader

# ✅ Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
serper_api_key = os.getenv("SERPER_API_KEY")

if not openai_api_key or not serper_api_key:
    st.error("❌ Please set both OPENAI_API_KEY and SERPER_API_KEY in your .env file.")
    st.stop()

# ✅ Initialize LLM and Embeddings
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo", api_key=openai_api_key)
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# ✅ Web search function using Serper.dev
def fetch_web_result(query):
    st.info("🌐 Searching the web with Serper...")
    headers = {
        "X-API-KEY": serper_api_key,
        "Content-Type": "application/json"
    }
    payload = {"q": query}

    try:
        response = requests.post("https://google.serper.dev/search", headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            return f"❌ Serper API error: {response.status_code}"

        data = response.json()
        results = data.get("organic", [])
        if not results:
            return "❌ No relevant web results found."

        # Format top 3 results
        answer = "\n\n".join([f"🔹 **{res['title']}**\n{res['link']}" for res in results[:3]])
        return answer

    except Exception as e:
        return f"❌ Error fetching web results: {str(e)}"

# ✅ Streamlit UI
st.set_page_config(page_title="GenAI Multi-PDF with Web Fallback", layout="wide")
st.title("📄 GenAI App - Multi-PDF Q&A with Web Fallback")

uploaded_files = st.file_uploader("📁 Upload one or more PDF files", type=["pdf"], accept_multiple_files=True)
query = st.text_input("❓ Ask a question")

# ✅ PDF-based Q&A if files uploaded
if uploaded_files and query:
    all_docs = []

    for file in uploaded_files:
        # Save uploaded PDF temporarily
        temp_path = f"temp_{file.name}"
        with open(temp_path, "wb") as f:
            f.write(file.read())

        # Load and extract documents
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        all_docs.extend(docs)

    # Split documents into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(all_docs)

    # Create vectorstore from chunks
    vectorstore = Chroma.from_documents(chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()

    # Setup RAG-based QA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    result = qa_chain.invoke({"query": query})
    answer = result["result"].strip()

    # Fallback if weak answer
    fallback_phrases = ["i don't know", "not found", "not mentioned", "uncertain", "unable to find", "no relevant", "cannot", "don't have"]
    if any(phrase in answer.lower() for phrase in fallback_phrases) or len(answer) < 30:
        st.warning("❔ Insufficient answer from PDFs. Using web search...")
        answer = fetch_web_result(query)

    st.subheader("📌 Answer:")
    st.write(answer)

# ✅ No PDF uploaded – use web directly
elif query:
    answer = fetch_web_result(query)
    st.subheader("📌 Answer:")
    st.write(answer)
