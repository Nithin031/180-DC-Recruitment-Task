# Dataset Creation – Jargon Scoring and Simplification

## Overview  
This project focuses on building a dataset of original and simplified sentences, evaluated for their **jargon density** and **complexity**. The goal was to create training data that balances technical heaviness with simplified alternatives, making it useful for future AI tasks such as text simplification and readability optimization.  

## Approach  
- Researched **text simplification techniques**, **BLEU evaluation**, **readability metrics**, and **LLM-based scoring methods**.  
- Used **BLEU scores** to measure similarity between original and simplified text.  
- Applied **Ollama-based scoring** to evaluate jargon and complexity in each sentence.  
- Computed a **Final Rating** by combining BLEU and Ollama scores.  
- Merged results into a structured dataset containing:  
  - Original Sentence  
  - Simplified Sentence  
  - BLEU Score  
  - Ollama Score  
  - Final Rating  

## Datasets  
- **Training Dataset** – Contains thousands of sentence pairs with evaluation scores.  
- **Test Dataset** – A separate set prepared to validate the scoring process on unseen data.  

## Outcome  
The final dataset is compact, structured, and enriched with both linguistic and AI-driven evaluation metrics, making it suitable for **model training and benchmarking** tasks.  
