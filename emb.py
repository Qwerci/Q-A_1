from sentence_transformers import SentenceTransformer
import torch

def create_embeddings(chunks):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(chunks, show_progress_bar=True)
    return embeddings


# # Collect and preprocess data
# topic = "Artificial Intelligence"
# raw_text = get_wikipedia_content(topic)
# preprocess_chunks = preprocess_text(raw_text)

# # Create embedding
# embeddings = create_embeddings(preprocess_chunks)
# # print(f"Created embeddings with shape: {embeddings.shape}")




