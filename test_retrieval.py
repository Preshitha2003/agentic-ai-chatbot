# from pinecone import Pinecone
# from sentence_transformers import SentenceTransformer
# import os
# from dotenv import load_dotenv

# # Load environment variables
# load_dotenv()

# # Initialize Pinecone client
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# # Initialize the same local embedding model
# model = SentenceTransformer("all-MiniLM-L6-v2")

# # Query to test retrieval
# query = "What are the key sales KPIs mentioned in the Q1 report?"
# vector = model.encode(query).tolist()

# # Query Pinecone for the most similar text chunks
# results = index.query(vector=vector, top_k=3, include_metadata=True)

# print("\n🔍 Top retrieved results:")
# for match in results["matches"]:
#     print(f"\nScore: {match['score']:.4f}")
#     print(match["metadata"]["text"][:300], "...")


from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initialize local embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

print("🤖 PDF Retrieval Chatbot (type 'exit' to quit)\n")

# Continuous chat loop
while True:
    user_query = input("🧠 Ask your question: ").strip()
    if user_query.lower() in ["exit", "quit"]:
        print("👋 Exiting chat. Goodbye!")
        break

    # Convert query to embedding
    query_vector = model.encode(user_query).tolist()

    # Retrieve most similar chunks from Pinecone
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)

    if not results["matches"]:
        print("⚠️ No relevant results found.")
        continue

    print("\n🔍 Top retrieved context:")
    for i, match in enumerate(results["matches"], start=1):
        print(f"\nResult {i} (Score: {match['score']:.4f})")
        print(match["metadata"]["text"][:500], "...")

    print("\n" + "-" * 80 + "\n")



