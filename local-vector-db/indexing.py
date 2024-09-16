import os
import json
import pandas as pd
from elasticsearch import Elasticsearch, helpers
from transformers import AutoTokenizer, AutoModel
import torch
from embeddings import generate_single_text_embedding, generate_directory_embeddings
import numpy as np

# Elasticsearch settings
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Function to index documents
def index_documents(input_dir, index_name):
    actions = []

    df = generate_directory_embeddings(input_dir)
    for _, row in df.iterrows():
        document_text = row["chunk_text"]
        embedding = row["embedding"]
        actions.append({
            "_index": index_name,
            "_source": {
                "document_text": document_text,
                "embedding": embedding
            }
        })

    helpers.bulk(es, actions)

# Create an index
def create_index(index_name):
    if es.indices.exists(index=index_name):
        es.indices.delete(index=index_name)

    es.indices.create(
        index=index_name,
        body={
            "mappings": {
                "properties": {
                    "document_text": {"type": "text"},
                    "embedding": {"type": "dense_vector", "dims": 768}
                }
            }
        }
    )

def search_documents(query, index_name):
    query_vector = generate_single_text_embedding(query)
    query_vector = np.array(query_vector)

    script_query = {
        "script_score": {
            "query": {
                "match_all": {}
            },
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {
                    "query_vector": query_vector
                }
            }
        }
    }

    response = es.search(
        index=index_name,
        body={
            "size": 5,
            "query": script_query,
            "_source": ["document_text"]
        }
    )

    return response

