from ollama import chat
import json
import os
import re

# Input file with your 120 cases
INPUT_FILE = "cases.txt"
OUTPUT_FILE = "legal_cases.json"
CHUNK_SIZE = 10   # number of cases per batch
MODEL_NAME = "llama3.2:3b"

# Read all cases
with open(INPUT_FILE, "r", encoding="utf-8") as f:
    all_cases = f.read()

# Split cases (assuming blank line separation)
cases_list = all_cases.split("\n\n")

json_results = []

for i in range(0, len(cases_list), CHUNK_SIZE):
    chunk_cases = "\n\n".join(cases_list[i:i+CHUNK_SIZE])

    prompt = f"""
    Convert the following legal cases into a strict JSON array only, with no extra commentary.
    Schema:
    {{
      "case_name": "",
      "case_type": "",
      "jurisdiction": "",
      "year_of_judgment": "",
      "key_legal_principles": [],
      "plaintiff_defendant_details": "",
      "case_outcome": ""
    }}

    Cases:
    {chunk_cases}
    """

    response = chat(model=MODEL_NAME, messages=[{"role": "user", "content": prompt}])

    # Extract JSON string
    json_text = response.message.content


    # Try parsing directly
    try:
        cases_json = json.loads(json_text)
        json_results.extend(cases_json)
    except json.JSONDecodeError:
        # Try extracting array if extra text is added
        match = re.search(r'(\[.*\])', json_text, re.DOTALL)
        if match:
            try:
                cases_json = json.loads(match.group(1))
                json_results.extend(cases_json)
            except json.JSONDecodeError:
                print(f"⚠️ Failed to parse JSON in chunk {i//CHUNK_SIZE + 1}")
        else:
            print(f"⚠️ No JSON found in chunk {i//CHUNK_SIZE + 1}")

    # Print progress every 10 chunks
    if (i // CHUNK_SIZE + 1) % 10 == 0:
        print(f"✅ Processed {i + CHUNK_SIZE} cases so far...")

# Save to JSON file
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(json_results, f, indent=2, ensure_ascii=False)

print(f"🎉 All {len(json_results)} cases saved to {OUTPUT_FILE}")
