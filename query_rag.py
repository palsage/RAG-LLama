from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS

# Define paths
vector_store_folder = 'vector_store'

def build_qa_chain(vectorstore, llm):
    RAG_TEMPLATE = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

    <context>
    {context}
    </context>

    Answer the following question:

    {question}
    """

    prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        output_parser=StrOutputParser()
    )

    return chain

def query_rag():
    # Initialize embedding model and vector store
    embedding_model = "plutonioumguy/bge-m3"
    vectorstore = Chroma(persist_directory=vector_store_folder, embedding_function=OllamaEmbeddings(model=embedding_model))

    # Initialize LLM
    llm = Ollama(model="llama3.1", base_url="http://localhost:11434")

    # Query loop
    while True:
        query = input("\nEnter your query (type 'quit' to exit): ")
        if query.lower() == 'quit':
            break

        # Build the QA chain
        qa_chain = build_qa_chain(vectorstore, llm)

        # Retrieve documents from vector store
        docs = vectorstore.similarity_search(query)

        # Prepare input for the chain
        chain_input = {
            "context": docs,
            "question": query
        }

        # Get and print the answer
        answer = qa_chain.run(chain_input)
        print("Answer:", answer)

if __name__ == "__main__":
    query_rag()
