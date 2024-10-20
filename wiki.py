import wikipediaapi
from langchain.text_splitter import RecursiveCharacterTextSplitter

def get_wikipedia_content(topic):
    wiki = wikipediaapi.Wikipedia(language='en',user_agent='RagPipeline/1.0 (kwesiqwerci@gmail.com)')
    page = wiki.page(topic)
    return page.text

def preprocess_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 500,
        chunk_overlap = 50,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Collect and preprocess data
topic = "Artificial Intelligence"
raw_text = get_wikipedia_content(topic)
preprocess_chunks = preprocess_text(raw_text)

print(f"Collected {len(preprocess_chunks)} chuncks from Wikipedia on {topic}")