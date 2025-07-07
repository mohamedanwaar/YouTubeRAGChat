# YouTube RAG Chatbot

A Streamlit web application that allows you to chat with the content of any YouTube video using Retrieval-Augmented Generation (RAG) techniques. Paste a YouTube URL, process the video, and ask questions about its content with AI-powered responses.

---

## Features
- **YouTube Video Transcript Extraction**: Automatically fetches and processes transcripts from YouTube videos.
- **Semantic Search with FAISS**: Efficiently indexes and retrieves relevant video segments using vector embeddings.
- **Conversational AI**: Uses Large Language Models (LLMs) to answer user questions based on video content.
- **Caching**: Speeds up repeated queries by caching processed videos.
- **Modern UI**: Clean, interactive interface built with Streamlit.

---

## Demo
![Demo Screenshot](demo_screenshot.png)

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/youtube-rag-chatbot.git
   cd youtube-rag-chatbot
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   *(Create a requirements.txt with the following packages if not present)*
   ```bash
   pip install streamlit langchain langchain-community langchain-ollama langchain-huggingface faiss-cpu sentence-transformers youtube-transcript-api
   ```
   - For Ollama LLM support, ensure you have [Ollama](https://ollama.com/) installed and running locally.

---

## Usage

1. **Start the Streamlit app:**
   ```bash
   streamlit run mainApp.py
   ```

2. **In your browser:**
   - Paste a YouTube video URL in the sidebar.
   - Click "Process Video" to extract and index the transcript.
   - Ask questions about the video content in the chat interface.

---

## Project Structure

- `mainApp.py` &mdash; Streamlit frontend and app logic
- `Backend.py` &mdash; Backend functions for transcript extraction, vector store management, and question answering
- `.cache/` &mdash; Stores cached vector stores for faster repeated access

---

## Requirements
- Python 3.8+
- [Ollama](https://ollama.com/) (for local LLM inference)

---

## Acknowledgements
- [Streamlit](https://streamlit.io/)
- [LangChain](https://python.langchain.com/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [HuggingFace Sentence Transformers](https://www.sbert.net/)
- [Ollama](https://ollama.com/)
- [YouTube Transcript API](https://github.com/jdepoix/youtube-transcript-api)

---

## License
MIT License 