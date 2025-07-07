from langchain_community.document_loaders import YoutubeLoader
from langchain_community.vectorstores import FAISS
import re
import streamlit as st
import os
from hashlib import sha256
import pickle
#Helper function to check the youtube url is valied or not 
def is_valid_youtube_url(url):
    pattern = r'(https?://)?(www\.)?(youtube|youtu|youtube-nocookie)\.(com|be)/'
    return re.match(pattern, url) is not None
#get the video transcript using YoutubeLoader
def get_video_transcript(url):
    try:
        loader = YoutubeLoader.from_youtube_url(url)
        return loader.load()
    except Exception as e:
        st.error(f"Failed to load video: {e}")
        return None
# Generate a unique cache path based on video URL
def get_cache_path(video_url):
    """This function generates a unique file path to store 
    or retrieve a cached version of the processed video (like its vectorstore), based on the video URL."""
    os.makedirs(".cache", exist_ok=True)
    video_hash = sha256(video_url.encode()).hexdigest()
    return os.path.join(".cache", f"{video_hash}.pkl")

# Create or load cached vector store
def create_vector_store(url, splitter, embeddings):
    # get the chche path for the url
    cache_path = get_cache_path(url)

    # Try to load from cache
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:

                vectorstore, chunk_count = pickle.load(f)
                st.warning(f"cached from eixting database")

            return vectorstore, chunk_count
        except Exception as e:
            st.warning(f"⚠️ Failed to load cache: {e}, reprocessing...")

    # If no cache or failed, generate fresh vector store
    docs = get_video_transcript(url)
    if docs:
        chunks = splitter.split_documents(docs)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        # Save to cache
        try:
            with open(cache_path, "wb") as f:
                pickle.dump((vectorstore, len(chunks)), f)
        except Exception as e:
            st.warning(f"⚠️ Failed to save cache: {e}")

        return vectorstore, len(chunks)

    return None, 0


def ask_question(vectorstore, question, llm, prompt_template):
    docs = vectorstore.similarity_search(question, k=3)
    context = "\n".join([doc.page_content for doc in docs])
    return llm.invoke(prompt_template.format(context=context, question=question))