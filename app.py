# app.py
import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from typing import List

# Load environment variables from .env
load_dotenv()

# --------- Helpers ---------
def extract_text_from_pdf(uploaded_file) -> str:
    reader = PdfReader(uploaded_file)
    full_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            full_text += page_text + "\n"
    return full_text

@st.cache_data(show_spinner=False)
def split_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

@st.cache_resource(show_spinner=False)
def build_faiss_index(chunks: List[str]):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    index = FAISS.from_texts(chunks, embeddings)
    return index

# --------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Ask your PDF ", layout="wide")

    # Custom CSS for red & black theme
    st.markdown("""
        <style>
        /* App background & font */
        .stApp {
            background-color: #111111;
            color: #00ffff;
            font-family:  'Gadget', sans-serif;
        }
        /* Title */
        .title {
            color: #00ffff;
            font-family: 'Gadget', sans-serif;
            font-size: 48px;
            text-align: center;
            text-shadow: 2px 2px #000000;
        }
        /* Buttons */
        .stButton>button {
            background-color: #ff3131;
            color: #00ffff;
            font-weight: bold;
            border-radius: 8px;
        }
        /* Input boxes */
        .stTextInput>div>div>input {
            background-color: #222222;
            color: #ffffff;
            border: 1px solid #ff3131;
        }
        /* Expanders */
        .stExpander {
            background-color: #1a1a1a;
            color: #ff4d4d;
        }
        </style>
    """, unsafe_allow_html=True)

    # Styled title
    st.markdown('<h1 class="title">📄 Ask your PDF — </h1>', unsafe_allow_html=True)
    st.markdown("<h3 style='color:#ff4d4d; text-align:center;'>AI-powered PDF Question Answering</h3>", unsafe_allow_html=True)
    st.markdown("<hr style='border: 1px solid #ff0000;'>", unsafe_allow_html=True)

    # Load Google API key
    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        st.warning("No GOOGLE_API_KEY found. Set it in your .env file.")
        return

    # Upload PDF & question input
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    user_question = st.text_input("Ask a question about the PDF:")

    if uploaded_file is None:
        st.info("Upload a PDF to enable question answering.")
        return

    # Extract text
    with st.spinner("Extracting text from PDF..."):
        text = extract_text_from_pdf(uploaded_file)
    if not text.strip():
        st.error("Could not extract any text from the PDF.")
        return

    # Split text & build FAISS index
    chunks = split_text(text)
    st.success(f"PDF loaded — split into {len(chunks)} chunks.")
    with st.spinner("Building FAISS index..."):
        knowledge_base = build_faiss_index(chunks)

    if not user_question:
        st.write("Preview of first chunk:")
        st.write(chunks[0][:1000] + ("..." if len(chunks[0]) > 1000 else ""))
        st.info("Type a question above to get an answer.")
        return

    # Initialize Gemini 2.5 Pro
    try:
        llm = ChatGoogleGenerativeAI(api_key=google_api_key, model="gemini-2.5-pro", temperature=0)
    except Exception as e:
        st.error(f"Failed to initialize AI Studio LLM: {e}")
        return

    # Build RetrievalQA
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=knowledge_base.as_retriever(),
        return_source_documents=True,
        chain_type="stuff"
    )

    # Generate answer
    with st.spinner("Retrieving documents and generating answer..."):
        try:
            result = qa_chain(user_question)
        except Exception as e:
            st.error(f"Generation failed: {e}")
            return

    # Display answer
    answer = result.get("result") or result.get("answer") or result
    st.subheader("📌 Answer")
    st.write(answer)

    # Display source chunks
    src_docs = result.get("source_documents")
    if src_docs:
        with st.expander("📚 Source chunks"):
            for i, doc in enumerate(src_docs, start=1):
                st.markdown(f"**Source {i}:**")
                st.write(doc.page_content[:1000] + ("..." if len(doc.page_content) > 1000 else ""))

if __name__ == "__main__":
    main()
