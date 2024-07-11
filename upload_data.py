from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, AzureAIDocumentIntelligenceLoader, BSHTMLLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

import os

def list_documents(folder_path):
    try:
        # List all files in the given folder
        files = os.listdir(folder_path)
        
        # Filter out only document files (for example, .txt, .pdf, .docx, etc.)
        document_files = [file for file in files if file.endswith(('.txt', '.pdf', '.docx', '.html'))]
        
        return document_files
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

if __name__ == "__main__":
    base_path = '/Users/danielmolina/Documents/GitFolder/vector-db-uploads'
    documents = list_documents(base_path)
    for doc_name in documents:
        print(f'Starting with {doc_name} ...')
        file_path = os.path.join(base_path, doc_name)
        if doc_name.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif doc_name.endswith('.txt'):
            loader = TextLoader(file_path)
        elif doc_name.endswith('.html'):
            loader = BSHTMLLoader(file_path)
        else:
            # assume docx file. Azure AI can handle more file formats such as JPEG, PNG, PDF, and TIFF
            loader = AzureAIDocumentIntelligenceLoader(
                api_endpoint=os.environ.get("AZURE_DOC_INTELLIGENCE_ENDPOINT"), 
                api_key=os.environ.get("AZURE_API_KEY"), 
                file_path=file_path,
                mode='markdown'
            )
        document = loader.load()

        print(f'Splitting {doc_name} ...')
        # chunking must be small enough to fit in context window
        # big enough to read it and know what chunk means
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        texts = text_splitter.split_documents(document)
        # read the chunks (texts) which should make sense since they have a semantic value
        print(f"created {len(texts)} chunks")

        embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))

        # we can write this ourself. But what about switching embeddings or vector store
        # Langchain implements using threading, async I/O (concurrently), batches, rate limits, etc
        # (ready for production usage)
        print('ingesting fr this time...')
        PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ.get("INDEX_NAME"))
