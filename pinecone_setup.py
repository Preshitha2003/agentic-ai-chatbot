from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists
if INDEX_NAME not in [index["name"] for index in pc.list_indexes()]:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")  
    )

print(f"✅ Pinecone index '{INDEX_NAME}' is ready!")

