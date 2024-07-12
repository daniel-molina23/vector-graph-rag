import faiss
import numpy as np

# Generate some random vectors for example
dimension = 128  # Dimension of vectors
num_vectors = 1000  # Number of vectors
vectors = np.random.random((num_vectors, dimension)).astype('float32')

# Create a FAISS index
index = faiss.IndexFlatL2(dimension)  # Use L2 distance metric
index.add(vectors)  # Add vectors to the index

# Generate a random query vector
query_vector = np.random.random((1, dimension)).astype('float32')

# Perform a search
k = 5  # Number of nearest neighbors
distances, indices = index.search(query_vector, k)
print(f"Top {k} nearest neighbors are {indices} with distances {distances}")
