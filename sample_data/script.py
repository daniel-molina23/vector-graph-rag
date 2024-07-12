import argparse
import os
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Access environment variables
graphrag_api_key = os.getenv('OPENAI_API_KEY')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_indexing(subfolder):
    index_root = os.path.join('.', subfolder)
    
    # Verify environment variables
    print(f"GraphRAG API Key: {graphrag_api_key}")
    print(f"Index Root: {index_root}")
    
    # Run the GraphRAG indexing pipeline
    os.system(f"python -m graphrag.index --root {index_root}")

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
