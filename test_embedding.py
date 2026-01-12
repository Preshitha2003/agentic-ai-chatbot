# from pinecone import Pinecone, ServerlessSpec
# import os
# from dotenv import load_dotenv

# load_dotenv()

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# # Create a new index with 384 dimensions
# pc.create_index(
#     name="local-embeddings-index",
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )

# print("✅ New Pinecone index created with 384 dimensions!")

#----------------------------------------------------------------------------------------------------------#

# in the above code we have created a pinecone index with 384 dimensions
# in the below code we will create embeddings using sentence transformers and store them in pinecone
# so both the code snippets are related to each other

#-----------------------------------------------------------------------------------------------------------#



from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Initialize local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Example text
text = "Sales performance of Q1 shows positive growth due to marketing campaigns."

# Generate embedding
vector = model.encode(text).tolist()

# Store vector in Pinecone
index.upsert([("sample-id-1", vector, {"text": text})])

print("✅ Embedding created and stored successfully using SentenceTransformers!")
