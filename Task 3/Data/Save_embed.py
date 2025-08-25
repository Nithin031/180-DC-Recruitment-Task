import json
import numpy as np
import ollama

# Load your JSON cases
with open("legal_cases.json", "r", encoding="utf-8") as f:
    cases = json.load(f)

# Helper function to combine important fields into one string
def case_to_text(case):
    return f"{case['case_name']} - {case['case_type']} - {case['jurisdiction']} - {case['year_of_judgment']} - {', '.join(case['key_legal_principles'])} - {case['case_outcome']}"

# Generate embeddings
embeddings = []
for case in cases:
    text = case_to_text(case)
    emb = ollama.embeddings(model="nomic-embed-text", prompt=text)["embedding"]
    embeddings.append(emb)

# Convert to numpy array
embeddings_np = np.array(embeddings).astype("float32")

# Save to disk
np.save("embeddings.npy", embeddings_np)
print("✅ Embeddings saved to embeddings.npy")
