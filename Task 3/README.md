# Courtroom Clash – AI Lawyers Debate System

## Introduction
For Task 3, the challenge was to build a system where two AI “lawyers” argue in a courtroom setup while a judge (the user) moderates and controls the flow of the trial.  

When I first saw this task, I was completely puzzled since I had **zero backend or frontend experience**. The only thing I knew was how RAG (Retrieval-Augmented Generation) worked. With the help of LLMs and a lot of trial and error, I pulled together both backend and frontend for this project, while learning embeddings, retrieval, and interactive system design.

The system features:
- **RAG Lawyer (Prosecution)** → fact-based, uses retrieval from curated legal cases.  
- **Chaos Lawyer (Defense)** → creative, improvisational, not reliant on retrieval.  
- **Judge (User/AI)** → moderates the trial, introduces evidence, objections, or role reversals, and delivers the verdict.  

---

## Data Collection & Metadata
Since no suitable law dataset was available, I **curated ~500 legal cases manually**. With the help of Ollama’s *llama3.2:3b* model, I structured the dataset into JSON with the following fields:

- `case_name` → *e.g., Smith v. Talking Pets*  
- `case_type` → *defamation, property dispute*  
- `jurisdiction` → *US, UK, India*  
- `year_of_judgment` → *1994, 2020*  
- `key_legal_principles` → *libel, negligence, contract law*  
- `case_outcome` → *dismissed, guilty, settled*  
- `facts & summary` → short case description  

This structured **metadata** enabled filtering and ranking during retrieval. For instance, a defamation query could prioritize precedents with “libel” or “slander.”

---

## Retrieval Strategy
Embeddings were generated with **Ollama’s `nomic-embed-text`** model and stored in a **FAISS index**. I used a **hybrid retrieval strategy**:

1. **Semantic Search** → cosine similarity on embeddings (α = 0.6).  
2. **Keyword Overlap** → string matching score (β = 0.3).  
3. **Metadata Boosting** → extra weight for exact metadata matches (γ = 0.1).  

Even with a small dataset (~500 cases), results were consistent, usually giving similarity > 0.5.

---

## System Workflow
Two LLMs power the system:  
- **OpenAI GPT-3.5 Turbo** → RAG Lawyer (structured, fact-based).  
- **Gemini 2.5 Pro** → Chaos Lawyer (improvisational, exaggerated).  

**Flow:**
1. **Case Generation** → random quirky case (from a list of 5) or user-provided case.  
2. **Debate Initialization** → RAG Lawyer starts with fact-based arguments using precedents.  
3. **Chaos Response** → Gemini generates counterarguments.  
4. **Rounds Continue** → Judge can add evidence, raise objections, or reverse roles.  
5. **Final Verdict** → AI generates a courtroom-style verdict summary.  

Debates are **multi-round** to allow continuity for features like evidence and objections.

---

## Features
- **Evidence Introduction** → Judge inputs evidence against one lawyer; normalized by AI; effect applies in next round.  
- **Objections** → Raised manually or auto-generated; apply for one round; force opponent to address.  
- **Role Reversal** → Prosecution/Defense roles can switch mid-trial (applies next round).  
- **Summaries & Verdicts** → On-demand summaries, plus final verdict when case is closed.  
- **Multi-Case Handling** → Reset backend/frontend state to run multiple debates.  

---

## System Design
- **Backend** → FastAPI with endpoints:  
  `/generate_case`, `/debate`, `/next_round`, `/judge_event`, `/objection`, `/role_reversal`, `/judge_decision`, `/verdict`, `/reset`.  

- **Frontend** → Streamlit UI for the Judge.  
  - Start debate, view timeline, raise objections, add evidence, reverse roles, and submit verdict.  

- **State Management** →  
  - Tracks rounds, roles, objections, effects, verdicts.  
  - Effects are one-round only and expire automatically.  
  - Full timeline stored to preserve case history.  

---

## Conclusion
This project was a **huge learning curve** for me. I began with no backend/frontend knowledge but ended up building a **complete interactive system**. Along the way I learned:  

- How to curate and structure a dataset for RAG.  
- Embeddings + FAISS for hybrid retrieval.  
- Multi-round debate logic with dynamic judge interventions.  
- How to connect backend and frontend for an interactive AI app.  

In the end, I created a courtroom simulation where **fact-based retrieval meets creative reasoning**. This project gave me hands-on experience with AI orchestration, metadata-driven pipelines, and interactive system design.  

---
## Demo Video  
You can watch a brief demo showcasing the full system in action here:  
[Courtroom Clash Live Demo](https://drive.google.com/file/d/1ty_I-qDcKkN0ZhBFSEUwlzGzSI64yxze/view?usp=sharing)
