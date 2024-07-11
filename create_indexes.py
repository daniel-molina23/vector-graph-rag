from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
import os 

# THIS IS A ONE TIME CALL!!!

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

def create_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

create_index(os.environ.get("ADMIN_INDEX_NAME"))
create_index(os.environ.get("CLEARANCE_Q_INDEX_NAME"))
create_index(os.environ.get("GENERAL_INDEX_NAME"))
