from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from emb import create_embeddings
from faiss import retrieve_relevant_passages

# Load model and tokenizer
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def query_understanding(query):
    return query

def answer_generation(query, relevant_passages):
    context = "\n".join(relevant_passages)
    prompt = f"Based on the following context, answer the question: {query}\n\nContext: {context}\n\nAnswer"

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=150, num_return_sequences=1, temperature=0.7)

    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer 
    
def rag_pipeline(query, embeddings, faiss_index, chunks):
    # Query understanding
    processed_query = query_understanding(query)

    # Create query embedding
    query_embedding = create_embeddings([processed_query])

    # Retrieve relevant passages
    relevant_passages = retrieve_relevant_passages(query_embedding, faiss_index, chunks)

    # Generate answer
    answer = answer_generation(processed_query, relevant_passages)

    return answer