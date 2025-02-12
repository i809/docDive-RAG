import os
import asyncio
import time
import gzip
import tempfile
import streamlit as st
import dill  # using dill for serialization
import concurrent.futures
from typing import Any, List
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain.vectorstores.faiss import FAISS
from functools import lru_cache
from langchain.docstore.in_memory import InMemoryDocstore

# Check for nest_asyncio and apply it.
try:
    import nest_asyncio
except ModuleNotFoundError:
    st.error("Required dependency 'nest_asyncio' is missing. Install it with 'pip install nest_asyncio'.")
    raise
nest_asyncio.apply()

try:
    import docx2txt  # for DOCX files
except ImportError:
    docx2txt = None

# Basic CSS styling.
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    .stChatInput input { background-color: #1E1E1E !important; color: #FFFFFF !important; border: 1px solid #3A3A3A !important; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) { background-color: #1E1E1E !important; border: 1px solid #3A3A3A; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) { background-color: #2A2A2A !important; border: 1px solid #404040; border-radius: 10px; padding: 15px; margin: 10px 0; }
    .stChatMessage .avatar { background-color: #00FFAA !important; color: #000000 !important; }
    .stChatMessage p, .stChatMessage div { color: #FFFFFF !important; }
    .stFileUploader { background-color: #1E1E1E; border: 1px solid #3A3A3A; border-radius: 5px; padding: 15px; }
    h1, h2, h3 { color: #00FFAA !important; }
    </style>
""", unsafe_allow_html=True)

# Custom prompt template (for Pydantic v2 compatibility)
class CustomChatPromptTemplate(ChatPromptTemplate):
    model_config = {"arbitrary_types_allowed": True}

PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context from the uploaded document(s) to answer the query.
Answer solely based on the document information. If the context is insufficient or not provided, state that you don't have relevant information.
Keep your answer concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""

PDF_STORAGE_PATH = 'document_store/pdfs/'
FAISS_INDEX_FILE = 'document_store/faiss_index.pkl'
# Adjust chunking: chunk size is set to 1000 and overlap 300.
CHUNK_SIZE = 1000  
CHUNK_OVERLAP = 300
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")

os.makedirs(PDF_STORAGE_PATH, exist_ok=True)
os.makedirs(os.path.dirname(FAISS_INDEX_FILE), exist_ok=True)

# Cache for duplicate processing.
_document_loading_cache = {}

def create_empty_faiss_index() -> FAISS:
    dummy_embedding = EMBEDDING_MODEL.embed_query("dummy")
    dim = len(dummy_embedding)
    import faiss
    index = faiss.IndexFlatL2(dim)
    docstore = InMemoryDocstore({})
    return FAISS(
        embedding_function=EMBEDDING_MODEL,
        index=index,
        docstore=docstore,
        index_to_docstore_id={}
    )

def load_faiss_index() -> FAISS:
    if os.path.exists(FAISS_INDEX_FILE) and os.path.getsize(FAISS_INDEX_FILE) > 0:
        try:
            with open(FAISS_INDEX_FILE, "rb") as f:
                state = dill.load(f)
            faiss_index = create_empty_faiss_index()
            faiss_index.__dict__.update(state)
            faiss_index.embedding_function = EMBEDDING_MODEL
            return faiss_index
        except Exception as e:
            st.error(f"Error loading FAISS index; initializing new index: {e}")
    return create_empty_faiss_index()

DOCUMENT_VECTOR_DB = load_faiss_index()

def persist_faiss_index(index: FAISS) -> None:
    state = index.__dict__.copy()
    state.pop("embedding_function", None)
    try:
        with open(FAISS_INDEX_FILE, "wb") as f:
            dill.dump(state, f)
    except Exception as e:
        st.error(f"Error persisting FAISS index: {e}")

def preprocess_document(doc: Any, filename: str) -> Any:
    doc.metadata = {"filename": filename}
    return doc

def save_uploaded_file(uploaded_file: Any) -> str:
    file_path = os.path.join(PDF_STORAGE_PATH, uploaded_file.name)
    try:
        with open(file_path, "wb") as file:
            file.write(uploaded_file.getbuffer())
    except Exception as e:
        st.error(f"Error saving file: {e}")
        raise e
    return file_path

async def process_file(uploaded_file) -> List[Any]:
    try:
        saved_path = await asyncio.to_thread(save_uploaded_file, uploaded_file)
        ext = os.path.splitext(saved_path)[1].lower()
        # For PDFs and DOCX files, convert the file content into plain text.
        if ext in [".pdf", ".docx"]:
            if ext == ".pdf":
                document_loader = PDFPlumberLoader(saved_path)
                docs = document_loader.load()
                # Join extracted text from all pages.
                text = "\n".join([d.page_content for d in docs])
            elif ext == ".docx":
                if docx2txt is None:
                    st.error("docx2txt is not installed. Please install it to load DOCX files.")
                    text = ""
                else:
                    text = docx2txt.process(saved_path)
            # If text was successfully extracted, write to a .txt file.
            if text:
                txt_path = os.path.splitext(saved_path)[0] + ".txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(text)
                saved_path = txt_path
        # Load documents (now in text format or original for .txt files).
        docs = await asyncio.to_thread(load_documents, saved_path)
        if not docs:
            st.error(f"Failed to load document: {uploaded_file.name}")
            return []
        return [preprocess_document(doc, uploaded_file.name) for doc in docs]
    except Exception as e:
        st.error(f"Error processing {uploaded_file.name}: {e}")
        return []

def load_documents(file_path: str) -> List[Any]:
    if file_path in _document_loading_cache:
        return _document_loading_cache[file_path]
    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".pdf":
            document_loader = PDFPlumberLoader(file_path)
            docs = document_loader.load()
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            docs = [type("Doc", (), {"page_content": text, "metadata": {}})()]
        elif ext == ".docx":
            if docx2txt is None:
                st.error("docx2txt not installed. Please install it to load DOCX files.")
                docs = []
            else:
                text = docx2txt.process(file_path)
                docs = [type("Doc", (), {"page_content": text, "metadata": {}})()]
        else:
            st.error(f"Unsupported file type: {ext}")
            docs = []
    except Exception as e:
        st.error(f"Error loading document: {e}")
        docs = []
    _document_loading_cache[file_path] = docs
    return docs

# Parallelized document splitting using all available CPU cores.
def chunk_documents(raw_documents: List[Any]) -> List[Any]:
    t0 = time.time()
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True
    )
    def split_doc(doc):
        return text_processor.split_documents([doc])
    chunks = []
    max_workers = os.cpu_count() or 4  # Use all available CPU cores.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(split_doc, doc) for doc in raw_documents]
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                chunks.extend(result)
            except Exception as e:
                st.error(f"Error during document splitting: {e}")
    # Filter out empty chunks.
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    t1 = time.time()
    st.write(f"Parallel splitting completed in {t1 - t0:.2f} seconds using {max_workers} workers.")
    return chunks

def index_documents(document_chunks: List[Any]) -> None:
    t0 = time.time()
    global DOCUMENT_VECTOR_DB
    try:
        DOCUMENT_VECTOR_DB.add_documents(document_chunks)
        persist_faiss_index(DOCUMENT_VECTOR_DB)
    except Exception as e:
        st.error(f"Error indexing documents: {e}")
    t1 = time.time()
    st.write(f"Indexing completed in {t1 - t0:.2f} seconds.")

def find_related_documents(query: str) -> List[Any]:
    try:
        return DOCUMENT_VECTOR_DB.similarity_search(query)
    except Exception as e:
        st.error(f"Error during similarity search: {e}")
        return []

@lru_cache(maxsize=128)
def generate_answer(user_query: str, context_text: str) -> str:
    conversation_prompt = CustomChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    try:
        return response_chain.invoke({"user_query": user_query, "document_context": context_text})
    except Exception as e:
        st.error(f"Error generating answer: {e}")
        return "Error generating answer."

QUERY_LOG = []

if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False
if "document_uploaded" not in st.session_state:
    st.session_state.document_uploaded = False

with st.container():
    st.title("📘 DocDive AI")
    st.markdown("### Your Intelligent Document Assistant")
    st.markdown("---")
    uploaded_files = st.file_uploader(
        "Upload Document(s) (PDF, TXT, DOCX)",
        type=["pdf", "txt", "docx"],
        help="Select one or more documents for analysis",
        accept_multiple_files=True
    )

    if uploaded_files:
        all_raw_docs = []
        t0 = time.time()
        # Use asyncio.run() to wrap file processing.
        results = asyncio.run(asyncio.gather(*[process_file(f) for f in uploaded_files]))
        for docs in results:
            all_raw_docs.extend(docs)
        if not all_raw_docs:
            st.error("No documents could be processed.")
            st.session_state.document_uploaded = False
        else:
            # Parallel chunking of documents.
            chunks = chunk_documents(all_raw_docs)
            # Index the chunks.
            index_documents(chunks)
            st.session_state.document_uploaded = True
            st.success(f"✅ Processed {len(all_raw_docs)} document(s) successfully in {time.time() - t0:.2f} seconds! You may now ask your questions.")

# Asynchronous response generation.
async def async_generate_response(user_input: str) -> str:
    ai_response_holder = {}

    async def generate_wrapper():
        try:
            relevant_docs = await asyncio.to_thread(find_related_documents, user_input)
            if not relevant_docs:
                ai_response_holder["response"] = "No related context found. Check your document content."
            else:
                context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                ai_response_holder["response"] = await asyncio.to_thread(generate_answer, user_input, context_text)
        except Exception as e:
            ai_response_holder["response"] = f"Error: {e}"

    task = asyncio.create_task(generate_wrapper())
    with st.spinner("Analyzing..."):
        while not task.done():
            await asyncio.sleep(0.1)
            if st.session_state.stop_generation:
                task.cancel()
                return "Generation was stopped by the user."
    await task
    QUERY_LOG.append({"query": user_input, "response": ai_response_holder.get("response", "")})
    return ai_response_holder.get("response", "No response generated.")

if not st.session_state.document_uploaded:
    st.info("Please upload document(s) to enable question answering.")
else:
    user_input = st.chat_input("Enter your question about the document(s)...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.stop_generation = False
        ai_response = asyncio.run(async_generate_response(user_input))
        st.session_state.messages.append({"role": "assistant", "content": ai_response})
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)