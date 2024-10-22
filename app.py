import streamlit as st
import plotly.express as px
import pandas as pd
import time
from wiki import get_wikipedia_content, preprocess_text
from emb import create_embeddings
from qdrant import create_qdrant_index
from rag_pipe import rag_pipeline

st.title("🤖 RAG-powered Q&A System")

# Initialize session state
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'chunks' not in st.session_state:
    st.session_state.chunks = None
if 'embeddings' not in st.session_state:
    st.session_state.embeddings = None
if 'qdrant_client' not in st.session_state:
    st.session_state.qdrant_client = None
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'response_times' not in st.session_state:
    st.session_state.response_times = []

# Sidebar for topic input and processing
with st.sidebar:
    topic = st.text_input("Enter a topic:", "Artificial Intelligence")
    
    if st.button("Process Topic"):
        with st.spinner("Processing topic..."):
            # Collect and preprocess data
            raw_text = get_wikipedia_content(topic)
            st.session_state.chunks = preprocess_text(raw_text)
            
            # Create embeddings
            st.session_state.embeddings = create_embeddings(st.session_state.chunks)
            
            # Create Qdrant index
            st.session_state.qdrant_client, st.session_state.collection_name = create_qdrant_index(
                st.session_state.embeddings
            )
            
            st.session_state.initialized = True
            st.success(f"Processed {len(st.session_state.chunks)} chunks about {topic}")

    # Display statistics
    if st.session_state.initialized:
        st.markdown("### System Statistics")
        st.write(f"Number of chunks: {len(st.session_state.chunks)}")
        st.write(f"Number of QA pairs: {len(st.session_state.history)}")
        if st.session_state.response_times:
            st.write(f"Average response time: {sum(st.session_state.response_times)/len(st.session_state.response_times):.2f}s")

# Main area for Q&A
if st.session_state.initialized:
    # Question input
    question = st.text_input("Ask a question about the topic:")
    
    if question:
        with st.spinner("Generating answer..."):
            start_time = time.time()
            
            answer = rag_pipeline(
                query=question,
                embeddings=st.session_state.embeddings,
                qdrant_client=st.session_state.qdrant_client,
                collection_name=st.session_state.collection_name,
                chunks=st.session_state.chunks
            )
            
            response_time = time.time() - start_time
            
            # Update history and response times
            st.session_state.history.append({
                "question": question,
                "answer": answer,
                "response_time": response_time
            })
            st.session_state.response_times.append(response_time)
    
    # Display conversation history
    if st.session_state.history:
        st.markdown("### Conversation History")
        for i, qa in enumerate(st.session_state.history):
            st.markdown(f"**Q{i+1}: {qa['question']}**")
            st.markdown(f"A: {qa['answer']}")
            st.markdown("---")
        
        # Display response time graph
        st.markdown("### Response Times")
        df = pd.DataFrame(st.session_state.response_times, columns=["response_time"])
        fig = px.line(df, x=range(len(df)), y="response_time", title="Response Times")
        st.plotly_chart(fig)

else:
    st.info("Please process a topic to begin.")