import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
import time
from Backend import is_valid_youtube_url,create_vector_store,ask_question

# Page configuration
st.set_page_config(
    page_title="YouTube RAG Chatbot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling (using Streamlit's native styling)
st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .chat-container {
        height: 400px;
        overflow-y: auto;
        padding: 1rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Caching functions
@st.cache_resource
def initialize_llm():
    return OllamaLLM(model="llama3.1:8b", temperature=0.5)

@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

@st.cache_resource
def initialize_splitter():
    return RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=70, length_function=len)

@st.cache_resource
def initialize_prompt():
    return ChatPromptTemplate.from_template("""
        You are an expert technical assistant. Based on the following video transcript, answer the user's question clearly and concisely.

        If the answer is not in the context, reply with "Sorry, I couldn't find that information in the video."

        Context:
        {context}

        User Question:
        {question}

        Answer:
    """)



# Initialize session state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "video_title" not in st.session_state:
    st.session_state.video_title = None
if "video_url" not in st.session_state:
    st.session_state.video_url = None
if "processing" not in st.session_state:
    st.session_state.processing = False

# Main header
st.markdown('<div class="main-header">', unsafe_allow_html=True)
st.title("🎥 YouTube RAG Chatbot")
st.markdown("**Ask questions about any YouTube video using AI-powered conversation**")
st.markdown('</div>', unsafe_allow_html=True)

# Sidebar for video processing
with st.sidebar:
    st.header("📺 Video Processing")
    
    # Video URL input
    video_url = st.text_input(
        "YouTube URL",
        placeholder="https://youtube.com/watch?v=...",
        help="Paste any YouTube video URL here"
    )
    
    # Process button
    process = st.button("🚀 Process Video", type="primary", use_container_width=True)
    
    # Processing status
    if st.session_state.processing:
        st.info("🔄 Processing video... Please wait")
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
    
    # Clear chat button (only show if video is loaded)
    if st.session_state.vectorstore:
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.chat_history.clear()
            st.rerun()
    
    st.divider()
    
    # Instructions
    with st.expander("ℹ️ How to Use"):
        st.write("""
        1. **Paste YouTube URL** above
        2. **Click 'Process Video'** to analyze
        3. **Wait for processing** to complete
        4. **Start chatting** with the video content
        5. **Ask specific questions** for best results
        """)
    
    with st.expander("💡 Tips for Better Results"):
        st.write("""
        - Use specific questions
        - Reference topics from the video
        - Ask about explanations or examples
        - Try follow-up questions
        """)

# Main content area
st.empty()  # Remove the extra column layout for session info

# Video Processing
if process and video_url:
        if not is_valid_youtube_url(video_url):
            st.error("❌ Invalid YouTube URL. Please check the format.")
        else:
            st.session_state.processing = True
            
            with st.spinner("🔄 Processing video transcript..."):
                try:
                    llm = initialize_llm()
                    embeddings = initialize_embeddings()
                    splitter = initialize_splitter()
                    prompt_template = initialize_prompt()
                    
                    vectorstore, chunk_count= create_vector_store(video_url, splitter, embeddings)
                    
                    if vectorstore:
                        st.session_state.vectorstore = vectorstore
                        st.session_state.chunk_count = chunk_count
                        st.session_state.video_url = video_url
                        st.session_state.chat_history.clear()
                        st.session_state.processing = False
                        
                        st.success(f"✅ Video processed successfully! {chunk_count} text chunks extracted.")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("❌ Failed to process the video. Please try again.")
                        st.session_state.processing = False
                        
                except Exception as e:
                    st.error(f"❌ Error processing video: {str(e)}")
                    st.session_state.processing = False

# Chat Interface
if st.session_state.vectorstore:
        st.subheader("💬 Chat with Video")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_history:
                for i, (question, answer) in enumerate(st.session_state.chat_history):
                    # User message
                    with st.chat_message("user", avatar="👤"):
                        st.markdown(question)
                    
                    # Assistant message
                    with st.chat_message("assistant", avatar="🤖"):
                        st.markdown(answer)
            else:
                st.info("👋 Start by asking a question about the video!")
        
        # Chat input
        if prompt := st.chat_input("Ask a question about the video...", key="chat_input"):
            # Add user message to chat history immediately
            with st.chat_message("user", avatar="👤"):
                st.markdown(prompt)
            
            # Generate and display assistant response
            with st.chat_message("assistant", avatar="🤖"):
                with st.spinner("Thinking..."):
                    llm = initialize_llm()
                    prompt_template = initialize_prompt()
                    response = ask_question(st.session_state.vectorstore, prompt, llm, prompt_template)
                
                # Stream the response
                message_placeholder = st.empty()
                full_response = ""
                
                for chunk in response:
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
                    time.sleep(0.01)
                
                message_placeholder.markdown(full_response)
            
            # Add to chat history
            st.session_state.chat_history.append((prompt, response))
            st.rerun()
