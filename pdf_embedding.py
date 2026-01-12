import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
from tqdm import tqdm

# === Load environment ===
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# === Initialize Pinecone ===
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# === Initialize local embedding model ===
model = SentenceTransformer("all-MiniLM-L6-v2")

# === Step 1: Load PDFs ===
pdf_dir = "data"  # ✅ PDFs are directly inside 'data/'
pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]

documents = []
for file in pdf_files:
    loader = PyPDFLoader(os.path.join(pdf_dir, file))
    docs = loader.load()
    documents.extend(docs)

print(f"📄 Loaded {len(documents)} PDF pages from {len(pdf_files)} files.")

# === Step 2: Split into chunks ===
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
print(f"🔹 Split into {len(chunks)} text chunks.")

# === Step 3: Embed and upsert into Pinecone ===
for i, chunk in enumerate(tqdm(chunks, desc="Embedding and uploading")):
    vector = model.encode(chunk.page_content).tolist()
    index.upsert(vectors=[(f"doc-{i}", vector, {"text": chunk.page_content})])

print("✅ PDF embeddings created and stored successfully in Pinecone!")
