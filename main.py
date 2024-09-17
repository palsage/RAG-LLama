from document_ingestion import DocumentIngestor
from multi_rag import MultiRAGManager

ingestor = DocumentIngestor()
ingestor.process_file(
    file_path='path_to_your_file.txt',
    vector_store_path='path_to_your_vector_store',
    record_file='path_to_your_record_file.json'
)


vector_store_paths = ['path_to_vector_store1', 'path_to_vector_store2']
rag_manager = MultiRAGManager(vector_store_paths=vector_store_paths)
question = "Your question here"
answer = rag_manager.query(question)
print("Answer:", answer)
