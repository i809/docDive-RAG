import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate
from typing import List, Dict, Any

# Custom prompt template allowing arbitrary types.
class CustomChatPromptTemplate(ChatPromptTemplate):
    model_config = {"arbitrary_types_allowed": True}

# Custom ChatOllama to allow arbitrary types in its model config.
class CustomChatOllama(ChatOllama):
    model_config = {"arbitrary_types_allowed": True}

def build_prompt_chain() -> CustomChatPromptTemplate:
    """
    Build the chat prompt chain using the current message log.
    """
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return CustomChatPromptTemplate.from_messages(prompt_sequence)

# Improved CSS styling with responsiveness.
st.markdown("""
<style>
    body {
        font-family: 'Segoe UI', sans-serif;
    }
    .main {
        max-width: 1200px;
        margin: auto;
        padding: 1rem;
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
        color: #ffffff;
        padding: 1rem;
        position: relative;
        z-index: 1000;
    }
    .stTextInput textarea {
        color: #ffffff !important;
        padding: 10px;
    }
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    .stSelectbox svg {
        fill: white !important;
    }
    .stSelectbox option, div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    /* Chat message styling */
    .stChatMessage {
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        max-width: 90%;
    }
    .stChatMessage:nth-child(odd) {
        background-color: #1E1E1E !important;
        border: 1px solid #3A3A3A;
    }
    .stChatMessage:nth-child(even) {
        background-color: #2A2A2A !important;
        border: 1px solid #404040;
    }
    .stChatMessage .avatar {
        background-color: #00FFAA !important;
        color: #000000 !important;
    }
    @media (max-width: 768px) {
       .stChatMessage, .stTextInput textarea {
          font-size: 0.9rem;
          padding: 10px;
       }
    }
    /* Button styling for deploy and sidebar buttons */
    div.stButton > button {
        background-color: #FFD700 !important;
        color: #000000 !important;
        border: none;
        border-radius: 5px;
        padding: 0.5em 1em;
        margin: 0.5em 0;
        position: relative;
        z-index: 1000;
        box-shadow: 0px 2px 4px rgba(0,0,0,0.5);
    }
    div.stButton > button:hover {
        background-color: #FFC107 !important;
    }
</style>
""", unsafe_allow_html=True)

# Set title and caption.
st.title("🧠 DeepSeek Code Companion")
st.caption("🚀 Your AI Pair Programmer with Debugging Superpowers")

# Sidebar configuration.
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model",
        ["deepseek-r1:1.5b", "deepseek-r1:3b"],
        index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design           
    """)
    st.divider()
    st.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initiate the chat engine using the custom subclass.
llm_engine = CustomChatOllama(
    model=selected_model,
    base_url="http://localhost:11434",
    temperature=0.3
)

# System prompt configuration.
system_prompt = SystemMessagePromptTemplate.from_template(
    "You are an expert AI coding assistant. Provide concise, correct solutions with strategic print statements for debugging. Always respond in English."
)

# Initialize session state for chat messages.
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}]

# Display previous chat messages.
with st.container():
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def generate_ai_response(prompt_chain: CustomChatPromptTemplate) -> str:
    """
    Generate an AI response by executing the prompt chain.
    """
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        st.error(f"Error generating AI response: {e}")
        return "Sorry, an error occurred while processing your request."

def build_prompt_chain_custom() -> CustomChatPromptTemplate:
    """
    Build the chat prompt chain using the current message log with the custom template.
    """
    return build_prompt_chain()

# Chat input and processing.
user_query = st.chat_input("Type your coding question here...")

if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.experimental_rerun()