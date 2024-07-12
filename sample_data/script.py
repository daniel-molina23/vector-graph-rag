import argparse
import os
import logging
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader, PyPDFLoader, BSHTMLLoader, AzureAIDocumentIntelligenceLoader
from PIL import Image
import pytesseract
import fitz

# Load environment variables
load_dotenv()

# Access environment variables
graphrag_api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def file_exists(file_path):
    if os.path.isfile(file_path):
        print(f"File '{file_path}' already exists.")
        return True
    else:
        print(f"File '{file_path}' does not exist. Creating it...")
        return False

def is_valid_azure_upload_doc_type(doc_name):
    # these are the supported file types for AZURE: PDF, JPEG/JPG, PNG, BMP, TIFF, HEIF, DOCX, XLSX, PPTX and HTML
    return doc_name.endswith('.pdf') or doc_name.endswith('.jpeg') or doc_name.endswith('.jpg') or doc_name.endswith('.png') or doc_name.endswith('.bmp') or doc_name.endswith('.tiff') or doc_name.endswith('.heif') or doc_name.endswith('.docx') or doc_name.endswith('.xlsx') or doc_name.endswith('.pptx') or doc_name.endswith('.html')

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as pdf_document:
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text("text")
    return text

def load_document(file_path):
    if file_path.endswith('.pdf'):
        # loader = PyPDFLoader(file_path)
        # documents = loader.load()
        # return "\n".join([doc.page_content for doc in documents])
        return extract_text_from_pdf(file_path)
    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    elif file_path.endswith('.html'):
        loader = BSHTMLLoader(file_path)
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    elif file_path.endswith(('.png', '.jpg', '.jpeg')):
        # Use pytesseract to extract text from image files
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)
        return text
    elif is_valid_azure_upload_doc_type(file_path):
        loader = AzureAIDocumentIntelligenceLoader(
                    api_endpoint=os.environ.get("AZURE_DOC_INTELLIGENCE_ENDPOINT"), 
                    api_key=os.environ.get("AZURE_API_KEY"), 
                    file_path=file_path,
                    mode='page'
                )
        documents = loader.load()
        return "\n".join([doc.page_content for doc in documents])
    else:
        raise ValueError(f'Unsupported file format: {file_path}')

def preprocess_files(old_file_dir, new_file_dir):
    for root, _, files in os.walk(old_file_dir):
        for file in files:
            old_file_path = os.path.join(root, file)
            new_file_path = os.path.join(new_file_dir, f"{os.path.splitext(file)[0]}.txt")
            print(f"old_file_path: {old_file_path}")
            print(f"new_file_path: {new_file_path}")
            try:
                if file_exists(new_file_path):
                    continue
                else:
                    text = load_document(old_file_path)
                    with open(new_file_path, 'w', encoding='utf-8') as f:
                        f.write(text)
                    logger.info(f"Processed and saved {file} to {new_file_path}")
            except Exception as e:
                logger.error(f"Error processing file {file}: {e}")

def run_indexing(subfolder):
    new_file_dir = os.path.join(subfolder, 'input') # final directory to store postprocessed files
    old_file_dir = os.path.join(subfolder, 'temp_input') # temp directory to store preprocessed files
    
    # Preprocess files
    preprocess_files(old_file_dir, new_file_dir)
    
    # Run the GraphRAG indexing pipeline
    os.system(f"python -m graphrag.index --root {subfolder}")

    # Example logging statement
    logger.info("Indexing completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run GraphRAG indexing pipeline.')
    parser.add_argument('subfolder', type=str, help='Subfolder to index (e.g., general, level_a, admin)')
    
    args = parser.parse_args()
    run_indexing(args.subfolder)


# Run the script examples:
# python script.py general
# python script.py level_q
# python script.py admin
