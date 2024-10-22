# import faiss
# import numpy as np


# def create_faiss_index(embeddings):
#     embeddings = np.array(embeddings).astype('float32')
#     dimension = embeddings.shape[1]
#     index = faiss.IndexFlatL2(dimension)
#     index.add(embeddings.astype('float32'))
#     return index

# # # Create FAISS index
# # faiss_index = create_faiss_index(embeddings)
# # print(f"Created FAISS index with {faiss_index.ntotal} vectors")

# # Function to retrieve relevant passages
# def retrieve_relevant_passages(query_embedding, index, chunks, k=5):
#     query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
#     distances, indices = index.search(query_embedding.astype('float32'), k)
#     return [chunks[i] for i in indices[0]]

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import numpy as np

def create_qdrant_index(embeddings, collection_name="documents"):
    # Convert embeddings to float32
    embeddings = np.array(embeddings).astype('float32')
    dimension = embeddings.shape[1]
    
    # Initialize Qdrant client (using in-memory storage)
    client = QdrantClient(":memory:")
    
    # Create collection
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=dimension, distance=Distance.COSINE)
    )
    
    # Add vectors to collection
    points = [
        PointStruct(
            id=idx,
            vector=embedding.tolist(),
            payload={}  # Can store additional metadata here if needed
        )
        for idx, embedding in enumerate(embeddings)
    ]
    
    client.upsert(
        collection_name=collection_name,
        points=points
    )
    
    return client, collection_name

def retrieve_relevant_passages(query_embedding, client, collection_name, chunks, k=5):
    # Ensure query embedding is in correct format
    query_embedding = np.array(query_embedding).astype('float32').reshape(-1)
    
    # Search for similar vectors
    search_result = client.search(
        collection_name=collection_name,
        query_vector=query_embedding.tolist(),
        limit=k
    )
    
    # Return corresponding chunks
    return [chunks[hit.id] for hit in search_result]
