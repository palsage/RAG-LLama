import os
import concurrent.futures
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Define paths
data_input_folder = 'data_input'
vector_store_folder = 'vector_store'

# Ensure vector_store folder exists
if not os.path.exists(vector_store_folder):
    os.makedirs(vector_store_folder)

def process_file(file_path, vectorstore):
    # Read and split text content from the file
    with open(file_path, 'r', encoding='utf-8') as file:
        text_content = file.read()

    # Split text into chunks for embedding
    chunks = text_content.split('\n\n')  # Split by paragraphs
    for chunk in chunks:
        if chunk.strip():
            vectorstore.add_texts([chunk])

    # Remove processed file
    os.remove(file_path)
    print(f"Processed and deleted {file_path}")

def process_and_save_to_vector_store(embedding_model="plutonioumguy/bge-m3", max_workers=None):
    # Initialize embedding function and vector store
    embedding_function = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(persist_directory=vector_store_folder, embedding_function=embedding_function)

    # Get all txt files in data_input
    txt_files = [os.path.join(data_input_folder, f) for f in os.listdir(data_input_folder) if f.endswith('.txt')]

    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_file, file_path, vectorstore) for file_path in txt_files]

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)

    # Persist the vector store
    vectorstore.persist()

if __name__ == "__main__":
    # Use the number of CPUs as the max workers, or customize based on your preference
    process_and_save_to_vector_store(max_workers=5*os.cpu_count())
