import os
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.decomposition import PCA

# Function to generate embeddings
def generate_single_text_embedding(text, token_length=512):
    # Load a pre-trained model and tokenizer 
    model_name = "sentence-transformers/all-mpnet-base-v2" # 768 dimensional embeddings
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    inputs = tokenizer(text, return_tensors='pt', max_length=token_length, truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
    return embeddings

def chunk_text(text, max_length=512, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_length - overlap):
        chunk = " ".join(words[i:i + max_length])
        chunks.append(chunk)
    return chunks

def generate_directory_embeddings(input_directory, dimension=1536, token_length=512):
    """
    input_directory: str, path to the directory containing the input files
    """

    # List all files in the directory and sort them alphabetically
    file_list = sorted(os.listdir(input_directory))

    # Initialize an empty list to store document texts and embeddings
    data = []
    embeddings_list = []

    # Read the content of each file, get the embedding, and append to the data list
    for filename in file_list:
        file_path = os.path.join(input_directory, filename)
        with open(file_path, 'r', encoding='utf-8') as file:
            document_text = file.read()
            # Check the length of the document and chunkify if necessary
            if len(document_text.split()) > 512:
                chunks = chunk_text(document_text)
                for chunk in chunks:
                    embedding = generate_single_text_embedding(chunk)
                    data.append({"document_text": chunk, "embedding": embedding})
            else:
                embedding = generate_single_text_embedding(document_text)
                data.append({"document_text": document_text, "embedding": embedding})

    

    # Create a DataFrame from the data list
    df = pd.DataFrame(data)

    # Extract embeddings and apply PCA to adjust dimensionality
    embeddings = np.array(df["embedding"].tolist())

    n_components = min(dimension, len(embeddings), embeddings.shape[1])

    # Apply PCA to adjust embeddings to specific dimensions
    pca = PCA(n_components=n_components)
    adjusted_embeddings = pca.fit_transform(embeddings)

    # Update the DataFrame with adjusted embeddings and create a list of these embeddings
    df["embedding"] = list(adjusted_embeddings)
    embeddings_list = list(adjusted_embeddings)

    # Optionally, print the DataFrame and list of embeddings
    print(df.head())
    print(embeddings_list)

    return df
