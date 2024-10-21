import faiss
import numpy as np


def create_faiss_index(embeddings):
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    return index

# # Create FAISS index
# faiss_index = create_faiss_index(embeddings)
# print(f"Created FAISS index with {faiss_index.ntotal} vectors")

# Function to retrieve relevant passages
def retrieve_relevant_passages(query_embedding, index, chunks, k=5):
    query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
    distances, indices = index.search(query_embedding.astype('float32'), k)
    return [chunks[i] for i in indices[0]]
