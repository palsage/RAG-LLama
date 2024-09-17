import os
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

# Define paths
data_input_folder = 'data_input'
vector_store_folder = 'vector_store'

# Ensure vector_store folder exists
if not os.path.exists(vector_store_folder):
    os.makedirs(vector_store_folder)

def process_and_save_to_vector_store(embedding_model="plutonioumguy/bge-m3"):
    # Initialize embedding function and vector store
    embedding_function = OllamaEmbeddings(model=embedding_model)
    vectorstore = Chroma(persist_directory=vector_store_folder, embedding_function=embedding_function)

    # Process all txt files in data_input
    for filename in os.listdir(data_input_folder):
        if filename.endswith('.txt'):
            file_path = os.path.join(data_input_folder, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                text_content = file.read()

            # Split text into chunks for embedding
            chunks = text_content.split('\n\n')  # Split by paragraphs
            for chunk in chunks:
                if chunk.strip():
                    vectorstore.add_texts([chunk])

            # Remove processed file
            os.remove(file_path)
            print(f"Processed and deleted {filename}")

    # Persist the vector store
    vectorstore.persist()

if __name__ == "__main__":
    process_and_save_to_vector_store()
