import time
import nltk
import streamlit as st
from selfQeryRAG import RAGImplementation

# Check if the NLTK data is already downloaded, if not, download it
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

# Initialize the RAG implementation
rag_implementation = RAGImplementation()

def main():
    st.title("Steam Game info Demo")

    # Create a text input for the user's question in the sidebar
    with st.sidebar:
        user_question = st.text_area("Enter your question about ANY steam games:")
        col1, col2, col3, col4 = st.columns([1, 1, 1, 1])
        with col4:
            think_button = st.button("Think")

    # Main content area
    if think_button and user_question:
        # Get the response from the RAG implementation
        response = rag_implementation.ask(user_question)
        
        # Display the response using streaming output
        st.subheader("🤖")
        output = st.empty()
        for i in range(len(response)):
            output.write(response[:i+1])
            time.sleep(0.0015) 
    elif think_button:
        st.warning("Please enter a question in the sidebar.")

if __name__ == "__main__":
    main()
