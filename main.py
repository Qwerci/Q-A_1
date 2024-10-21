from wiki import get_wikipedia_content, preprocess_text
from emb import create_embeddings
from faiss import create_faiss_index, retrieve_relevant_passages
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from rag_pipe import rag_pipeline

def main():

    # Collect and preprocess data
    topic = "Artificial Intelligence"
    print(f"Collecting data on {topic}...")
    raw_text = get_wikipedia_content(topic)
    chunks = preprocess_text(raw_text)

    print(f"Collected amd preprocessed{len(chunks)} chuncks from Wikipedia on {topic}")

    # Create embedding
    print("Creating embeddings...")
    embeddings = create_embeddings(chunks)
    print(f"Created embeddings with shape: {embeddings.shape}")

    # Create FAISS index
    print("Creating FAISS index...")
    faiss_index = create_faiss_index(embeddings)
    print(f"Created FAISS index with {faiss_index.ntotal} vectors")

    # Load model and tokenizer
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Example usage
    while True: 
        query = input("\nEnter your question about Artificial Intelligence (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        print("Generating answer...")
        result = rag_pipeline(query, embeddings, faiss_index, chunks, model, tokenizer)
        print(f"\nQuery: {query}")
        print(f"\nAnswer: {result}")

    print("Thank you for using the AI Q&A system!")

if __name__ == "__main__":
    main()