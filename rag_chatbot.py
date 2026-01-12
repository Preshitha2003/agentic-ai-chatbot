import os
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# === Load environment variables ===
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))

# Initialize embedding + summarization models
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("🤖 AI PDF Chatbot Ready! (type 'exit' to quit)\n")

# === Chat Loop ===
while True:
    query = input("🧠 Ask your question: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break

    # Step 1: Embed query and retrieve top context
    query_vector = embed_model.encode(query).tolist()
    results = index.query(vector=query_vector, top_k=3, include_metadata=True)

    if not results["matches"]:
        print("⚠️ No relevant content found.")
        continue

    # Step 2: Combine retrieved text
    combined_context = " ".join([m["metadata"]["text"] for m in results["matches"]])

    # Step 3: Summarize into a concise answer
    print("\n💡 Generating concise answer...")
    summary = summarizer(
        combined_context[:3000],  # limit input size for model
        max_length=200,
        min_length=50,
        do_sample=False
    )[0]["summary_text"]

    print("\n🧾 Answer:")
    print(summary)
    print("\n" + "-" * 80 + "\n")
