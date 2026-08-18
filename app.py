import os
import shutil
import streamlit as st

from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not found in .env file")
    st.stop()


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Level 2 RAG Dashboard",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Level 2 — Multi-PDF RAG Dashboard")

st.caption(
    "Multiple PDF Retrieval + Configurable Chunking + Top-K Search"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Level 2 Settings")


# -------------------------
# CHUNK SIZE
# -------------------------

chunk_size = st.sidebar.slider(
    "📏 Chunk Size",
    min_value=200,
    max_value=2000,
    value=500,
    step=100
)


# -------------------------
# CHUNK OVERLAP
# -------------------------

max_overlap = max(0, chunk_size - 1)

default_overlap = min(100, max_overlap)

chunk_overlap = st.sidebar.slider(
    "🔄 Chunk Overlap",
    min_value=0,
    max_value=max_overlap,
    value=default_overlap,
    step=50 if max_overlap >= 50 else 1
)


# -------------------------
# TOP K
# -------------------------

top_k = st.sidebar.slider(
    "🔎 Top-K Retrieved Documents",
    min_value=1,
    max_value=10,
    value=3,
    step=1
)


st.sidebar.divider()


# ============================================================
# CURRENT SETTINGS
# ============================================================

st.sidebar.subheader("📊 Current Settings")

st.sidebar.info(
    f"""
**Chunk Size:** {chunk_size}

**Chunk Overlap:** {chunk_overlap}

**Top-K:** {top_k}
"""
)


# ============================================================
# PDF UPLOAD
# ============================================================

st.header("📄 Upload Multiple PDFs")

uploaded_files = st.file_uploader(
    "Upload up to 5 PDF files",
    type=["pdf"],
    accept_multiple_files=True
)


# ============================================================
# DISPLAY UPLOADED FILES
# ============================================================

if uploaded_files:

    if len(uploaded_files) > 5:

        st.error("❌ Please upload a maximum of 5 PDF files.")
        st.stop()

    st.success(
        f"✅ {len(uploaded_files)} PDF file(s) uploaded"
    )

    st.subheader("📁 Uploaded Files")

    for file in uploaded_files:
        st.write(f"📄 {file.name}")


# ============================================================
# PROCESS DOCUMENTS
# ============================================================

if uploaded_files:

    if st.button(
        "🚀 Process Documents",
        use_container_width=True
    ):

        all_documents = []

        progress = st.progress(0)

        # --------------------------------------------
        # LOAD EACH PDF
        # --------------------------------------------

        for index, uploaded_file in enumerate(uploaded_files):

            temp_path = os.path.join(
                os.getcwd(),
                uploaded_file.name
            )

            try:

                # Save temporary PDF
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                # Load PDF
                loader = PyPDFLoader(temp_path)

                documents = loader.load()

                # Add filename to metadata
                for doc in documents:

                    doc.metadata["source"] = uploaded_file.name

                    # Page number
                    if "page" in doc.metadata:
                        doc.metadata["page"] = (
                            doc.metadata["page"] + 1
                        )

                all_documents.extend(documents)

            finally:

                # Delete temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)

            progress.progress(
                (index + 1) / len(uploaded_files)
            )


        # ====================================================
        # TOTAL PAGES
        # ====================================================

        st.info(
            f"📄 Total pages loaded: {len(all_documents)}"
        )


        # ====================================================
        # CHUNKING
        # ====================================================

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

        chunks = text_splitter.split_documents(
            all_documents
        )

        st.success(
            f"✅ Created {len(chunks)} chunks"
        )


        # ====================================================
        # EMBEDDINGS
        # ====================================================

        st.info("🔄 Loading embedding model...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",

            # Important for your meta-tensor problem
            model_kwargs={
                "device": "cpu"
            },

            encode_kwargs={
                "normalize_embeddings": True
            }
        )


        # ====================================================
        # VECTOR DATABASE
        # ====================================================

        # Create a new collection
        collection_name = "level2_rag"

        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=collection_name
        )


        # ====================================================
        # SAVE TO SESSION
        # ====================================================

        st.session_state["vectorstore"] = vectorstore

        st.session_state["processed"] = True

        st.session_state["file_names"] = [
            file.name for file in uploaded_files
        ]

        st.session_state["chunk_count"] = len(chunks)

        st.session_state["page_count"] = len(all_documents)


        st.success(
            "🎉 Documents processed successfully!"
        )


# ============================================================
# QUESTION ANSWERING
# ============================================================

if st.session_state.get("processed", False):

    st.divider()

    st.header("🔎 Ask Questions")


    # ========================================================
    # QUESTION INPUT
    # ========================================================

    question = st.text_input(
        "Enter your question",
        placeholder=(
            "Example: What is machine learning?"
        )
    )


    # ========================================================
    # GET ANSWER
    # ========================================================

    if st.button(
        "🤖 Get Answer",
        use_container_width=True
    ):

        if not question.strip():

            st.warning(
                "⚠️ Please enter a question."
            )

            st.stop()


        # ====================================================
        # VECTORSTORE
        # ====================================================

        vectorstore = st.session_state["vectorstore"]


        # ====================================================
        # TOP-K RETRIEVAL
        # ====================================================

        retrieved_docs = vectorstore.similarity_search(
            question,
            k=top_k
        )


        # ====================================================
        # FILE NAMES
        # ====================================================

        file_names = st.session_state.get(
            "file_names",
            []
        )

        file_list = "\n".join(
            f"- {name}"
            for name in file_names
        )


        # ====================================================
        # CONTEXT WITH METADATA
        # ====================================================

        context_parts = []

        for i, doc in enumerate(retrieved_docs):

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )

            context_parts.append(
                f"""
RETRIEVED DOCUMENT {i + 1}

SOURCE FILE:
{source}

PAGE:
{page}

CONTENT:
{doc.page_content}
"""
            )


        context = "\n\n".join(
            context_parts
        )


        # ====================================================
        # LLM
        # ====================================================

        llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model="llama-3.1-8b-instant",
            temperature=0
        )


        # ====================================================
        # PROMPT
        # ====================================================

        prompt = f"""
You are a helpful document question-answering assistant.

You are working with multiple uploaded PDF documents.

UPLOADED FILES:
{file_list}


RETRIEVED DOCUMENTS:
{context}


USER QUESTION:
{question}


INSTRUCTIONS:

1. Answer the user's question clearly and directly.

2. If the user asks:
   "What files are there?"
   "What PDFs are uploaded?"
   "Which documents are available?"
   
   List the uploaded filenames from the UPLOADED FILES section.

3. If the user asks about information inside a PDF,
   use the retrieved document content.

4. If possible, mention the source filename and page number.

5. Do not invent information.

6. If the requested information is not available,
   say:
   "I could not find the answer in the uploaded documents."


ANSWER:
"""


        # ====================================================
        # GENERATE ANSWER
        # ====================================================

        with st.spinner("🤖 Generating answer..."):

            response = llm.invoke(prompt)


        # ====================================================
        # CURRENT RETRIEVED ANSWER
        # ====================================================

        st.subheader(
            "💡 Current Retrieved Answer"
        )

        st.success(
            response.content
        )


        # ====================================================
        # RETRIEVED DOCUMENTS
        # ====================================================

        st.divider()

        st.subheader(
            f"📚 Top-{top_k} Retrieved Documents"
        )


        for i, doc in enumerate(
            retrieved_docs
        ):

            source = doc.metadata.get(
                "source",
                "Unknown"
            )

            page = doc.metadata.get(
                "page",
                "Unknown"
            )


            with st.expander(
                f"📄 {i + 1}. {source} — Page {page}"
            ):

                st.write(
                    f"**Source File:** {source}"
                )

                st.write(
                    f"**Page:** {page}"
                )

                st.write(
                    "### Retrieved Content"
                )

                st.write(
                    doc.page_content
                )


# ============================================================
# LEVEL 2 INFORMATION
# ============================================================

st.divider()

st.header(
    "📊 Level 2 RAG Configuration"
)


col1, col2, col3 = st.columns(3)


with col1:

    st.metric(
        "📏 Chunk Size",
        chunk_size
    )


with col2:

    st.metric(
        "🔄 Chunk Overlap",
        chunk_overlap
    )


with col3:

    st.metric(
        "🔎 Top-K Documents",
        top_k
    )


# ============================================================
# PROCESSING STATISTICS
# ============================================================

if st.session_state.get(
    "processed",
    False
):

    st.subheader(
        "📈 Processing Statistics"
    )

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "📁 PDF Files",
            len(
                st.session_state.get(
                    "file_names",
                    []
                )
            )
        )


    with col2:

        st.metric(
            "📄 Total Pages",
            st.session_state.get(
                "page_count",
                0
            )
        )


    with col3:

        st.metric(
            "🧩 Total Chunks",
            st.session_state.get(
                "chunk_count",
                0
            )
        )