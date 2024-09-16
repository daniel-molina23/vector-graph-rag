from preprocessing import preprocess_files
import os
import logging
import argparse
from indexing import create_index, search_documents, index_documents

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_indexing(subfolder):
    new_file_dir = os.path.join(subfolder, 'input') # final directory to store postprocessed files
    # if new_file_dir does not exist, create it
    if not os.path.exists(new_file_dir):
        os.makedirs(new_file_dir)
    old_file_dir = os.path.join(subfolder, 'raw_input') # temp directory to store preprocessed files
    
    # Preprocess files
    preprocess_files(old_file_dir, new_file_dir)
    logger.info('Finished preprocessing files')
    
    # Run the milvus indexing pipeline
    input_dir = new_file_dir
    index_name = f"{subfolder}_index"
    # create_index(index_name)
    index_documents(input_dir, index_name)
    logger.info('Finished indexing collection')

    # Perform a search
    query = "When did the usage of guns occur?"
    docs_returned = search_documents(query, index_name)
    
    # Example logging statement
    logger.info("Indexing completed successfully")
    logging.info(f"Query: {query}")
    logging.info(f"Results: {docs_returned}")

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Run local vector milvus indexing pipeline.')
    # parser.add_argument('subfolder', type=str, help='Subfolder to index (e.g., general, level_q, admin)')
    
    # args = parser.parse_args()
    # run_indexing(args.subfolder)
    run_indexing('general')