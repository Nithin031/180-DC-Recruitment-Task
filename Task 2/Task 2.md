# Dataset Creation – Jargon Scoring and Simplification

## Overview
In this project, I focused on creating a dataset of original and simplified sentences and evaluated them for **jargon density** and **complexity**. The goal was to prepare training data that balances technical heaviness with simpler alternatives. This dataset can be used for classical ML tasks like **text simplification using KNN + TF-IDF** and readability analysis.

---

## Approach
- I prepared a **training dataset** (`Final_Train.csv`) and a **test dataset** (`Final_test.csv`) with columns `Original` and `Simplified`.
- I used **TF-IDF** to convert sentences into vectors, with unigrams and bigrams, lowercase conversion, and sublinear term frequency scaling.
- I trained a **KNN model** on the TF-IDF vectors to find the closest sentence in the training set for each test sentence.
- For each test sentence, the **simplified sentence from the nearest neighbor** in the training set was used as the prediction.
- The process optionally included **spell checking** and text cleaning, though the main focus was on vector similarity.
