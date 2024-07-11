import json
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader, AzureAIDocumentIntelligenceLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain.graphs import Neo4jGraph

load_dotenv()


# graph = Neo4jGraph(
#     url=os.getenv("NEO_4J_URL"), 
#     username=os.getenv("NEO_4J_USERNAME"), 
#     password=os.getenv("NEO_4J_PASSWORD")
# )

# import requests

# url = "https://gist.githubusercontent.com/tomasonjo/08dc8ba0e19d592c4c3cde40dd6abcc3/raw/da8882249af3e819a80debf3160ebbb3513ee962/microservices.json"
# import_query = requests.get(url).json()['query']
# graph.query(
#     import_query
# )


vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    url=os.getenv("NEO_4J_URL"), 
    username=os.getenv("NEO_4J_USERNAME"), 
    password=os.getenv("NEO_4J_PASSWORD"),
    index_name='tasks',
    node_label="Task",
    text_node_properties=['name', 'description', 'status'],
    embedding_node_property='embedding',
)

# index_name: name of the vector index.
# node_label: node label of relevant nodes.
# text_node_properties: properties to be used to calculate embeddings and retrieve from the vector index.
# embedding_node_property: which property to store the embedding values to.

response = vector_index.similarity_search(
    "How will RecommendationService be updated?"
)
print(response[0].page_content)


from langchain.chains import RetrievalQA
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_community.chat_models import ChatOpenAI

vector_qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-3.5-turbo"),
    chain_type="stuff",
    retriever=vector_index.as_retriever()
)
output = vector_qa.invoke(
    "How will recommendation service be updated?"
)

print(f"Any output: {output['result']}")
# The RecommendationService is currently being updated to include a new feature 
# that will provide more personalized and accurate product recommendations to 
# users. This update involves leveraging user behavior and preference data to 
# enhance the recommendation algorithm. The status of this update is currently
# in progress.
