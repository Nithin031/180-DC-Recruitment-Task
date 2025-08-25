import json
import numpy as np
import faiss
import ollama

# 1. Load your JSON file
with open("legal_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

# 2. Helper function to embed text using Ollama
def get_embedding(text, model="nomic-embed-text"):
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]

# 3. Prepare texts to embed (combine important fields for context)
texts = []
for case in cases:
    text = f"{case['case_name']} - {case['case_type']} - {case['jurisdiction']} - {case['year_of_judgment']} - {', '.join(case['key_legal_principles'])} - {case['case_outcome']}"
    texts.append(text)

# 4. Generate embeddings
print("🔄 Generating embeddings...")
embeddings = [get_embedding(t) for t in texts]

# Convert to numpy float32 (FAISS requirement)
embeddings_np = np.array(embeddings).astype("float32")

# 5. Build FAISS index (cosine similarity via inner product)
dimension = embeddings_np.shape[1]  # length of embedding vector
index = faiss.IndexFlatIP(dimension)  
faiss.normalize_L2(embeddings_np)  # normalize for cosine similarity
index.add(embeddings_np)

print(f"✅ FAISS index built with {index.ntotal} cases")

# Instead of fixed query, take user input
query = input("Enter your legal query: ")

query_emb = get_embedding(query)
query_emb = np.array([query_emb]).astype("float32")
faiss.normalize_L2(query_emb)

# Search top 3
D, I = index.search(query_emb, k=3)

print("\n📌 Top Matches:")
for idx, score in zip(I[0], D[0]):
    print(f"  - {cases[idx]['case_name']} ({cases[idx]['year_of_judgment']}) [score={score:.3f}]")
