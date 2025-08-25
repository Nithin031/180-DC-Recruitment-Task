import json
import os
import re
import random
from collections import Counter
from datetime import datetime
import numpy as np
import faiss
import ollama
from fastapi import FastAPI, Body
from openai import AzureOpenAI
from google import genai  # pip install google-genai

app = FastAPI(title="Courtroom Clash (Azure RAG + Gemini Judge/Defense/Evidence)")

# ----------------------
# Config
# ----------------------
# Azure OpenAI (RAG)
AZURE_ENDPOINT = os.getenv("ENDPOINT_URL")
AZURE_DEPLOYMENT = os.getenv("DEPLOYMENT_NAME")
AZURE_KEY = os.getenv("AZURE_KEY")
print(f"Azure Endpoint: {AZURE_ENDPOINT}")
print(f"Azure Deployment: {AZURE_DEPLOYMENT}")
print(f"Azure Key: {'set' if AZURE_KEY else 'not set'}")

try:
    if AZURE_ENDPOINT and AZURE_DEPLOYMENT and AZURE_KEY:
        azure_client = AzureOpenAI(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_KEY,
            api_version="2025-01-01-preview"
        )
    else:
        azure_client = None
except Exception as e:
    print(f"Azure init failed: {e}")
    azure_client = None

# Embeddings via Ollama
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

# Gemini (Defense/Judge/Evidence + RAG fallback)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_CHAOS = os.getenv("GEMINI_MODEL_CHAOS", "gemini-2.5-pro")
GEMINI_MODEL_JUDGE = os.getenv("GEMINI_MODEL_JUDGE", "gemini-2.5-pro")
try:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else genai.Client()
    print(f"Gemini OK. CHAOS={GEMINI_MODEL_CHAOS}, JUDGE={GEMINI_MODEL_JUDGE}")
except Exception as e:
    print(f"Gemini init failed: {e}")
    gemini_client = None

CASES_FILE = "legal_cases.json"

# ----------------------
# Load cases + build FAISS index
# ----------------------
def _safe_join(items):
    if isinstance(items, (list, tuple)):
        return ", ".join(map(str, items))
    return str(items) if items is not None else ""

def get_embedding(text, model=OLLAMA_EMBED_MODEL):
    resp = ollama.embeddings(model=model, prompt=text)
    return resp["embedding"]

if not os.path.exists(CASES_FILE):
    raise RuntimeError(f"{CASES_FILE} not found. Create it before starting the backend.")

with open(CASES_FILE, "r", encoding="utf-8") as f:
    cases = json.load(f)

texts = [
    f"{case.get('case_name','')} - {case.get('case_type','')} - {case.get('jurisdiction','')} - "
    f"{case.get('year_of_judgment','')} - {_safe_join(case.get('key_legal_principles', []))} - {case.get('case_outcome','')}"
    for case in cases
]

print("Generating embeddings...")
embeddings = [get_embedding(t) for t in texts]
if not embeddings or not embeddings[0]:
    raise RuntimeError("No embeddings generated; check embedding model/configuration.")

embeddings_np = np.array(embeddings, dtype="float32")
if embeddings_np.ndim != 2:
    raise RuntimeError(f"Unexpected embedding shape: {embeddings_np.shape}")

dimension = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(embeddings_np)
index.add(embeddings_np)
print(f"FAISS index built with {index.ntotal} cases using {OLLAMA_EMBED_MODEL}")

# ----------------------
# Keyword scoring prep (hybrid retrieval)
# ----------------------
def tokenize(s):
    return re.findall(r"[a-z0-9]+", (s or "").lower())

DOC_TOKENS = [Counter(tokenize(t)) for t in texts]
DOC_LENS = [sum(c.values()) for c in DOC_TOKENS]

def keyword_score(query, doc_idx):
    q = Counter(tokenize(query))
    d = DOC_TOKENS[doc_idx]
    if not d:
        return 0.0
    overlap = sum(min(q[t], d.get(t, 0)) for t in q)
    return overlap / (DOC_LENS[doc_idx] or 1)

def metadata_boost(case, filters):
    if not filters:
        return 0.0
    boost = 0.0
    if "case_type" in filters and case.get("case_type") == filters["case_type"]:
        boost += 0.2
    if "jurisdiction" in filters and case.get("jurisdiction") == filters["jurisdiction"]:
        boost += 0.2
    if "year_of_judgment" in filters and str(case.get("year_of_judgment")) == str(filters["year_of_judgment"]):
        boost += 0.1
    if "key_legal_principles" in filters:
        qs = set(filters["key_legal_principles"])
        cs = set(case.get("key_legal_principles") or [])
        if qs & cs:
            boost += 0.2
    return boost

def retrieve_case(query, filters=None, top_k=3, alpha=0.6, beta=0.3, gamma=0.1):
    # semantic
    q_emb = np.array([get_embedding(query)], dtype="float32")
    faiss.normalize_L2(q_emb)
    D, I = index.search(q_emb, k=min(50, index.ntotal))
    cand = list(zip(I[0], D[0]))  # (doc_idx, sem_score)

    # re-score with keyword + metadata
    rescored = []
    for doc_idx, sem in cand:
        key = keyword_score(query, doc_idx)
        meta = metadata_boost(cases[doc_idx], filters)
        final = alpha * float(sem) + beta * key + gamma * meta
        rescored.append((doc_idx, final))
    rescored.sort(key=lambda x: x[1], reverse=True)

    # hard filter if provided
    picks = []
    if filters:
        for doc_idx, score in rescored:
            c = cases[doc_idx]
            ok = True
            for k, v in filters.items():
                if k == "key_legal_principles":
                    if not (set(v) & set(c.get(k) or [])):
                        ok = False
                        break
                else:
                    if str(c.get(k)) != str(v):
                        ok = False
                        break
            if ok:
                picks.append((doc_idx, score))
        if not picks:
            picks = rescored[:top_k]
        else:
            picks = picks[:top_k]
    else:
        picks = rescored[:top_k]

    return [(cases[idx], score) for idx, score in picks]

# ----------------------
# Gemini helper
# ----------------------
def gemini_generate(prompt, model, max_retries=2):
    last_err = None
    for _ in range(max_retries):
        try:
            if gemini_client is None:
                break
            resp = gemini_client.models.generate_content(model=model, contents=prompt)
            text = getattr(resp, "text", None)
            if text and text.strip():
                usage = getattr(resp, "usage_metadata", None)
                return text.strip(), resp, usage
            try:
                cand = resp.candidates[0].content.parts[0].text
                if cand and cand.strip():
                    usage = getattr(resp, "usage_metadata", None)
                    return cand.strip(), resp, usage
            except Exception:
                pass
        except Exception as e:
            last_err = e
    if last_err:
        print(f"[gemini_generate] failed on model={model}: {last_err}")
    return None, None, None

def usage_to_tokens(usage):
    if not usage:
        return None
    return {
        "prompt_tokens": getattr(usage, "prompt_token_count", None),
        "completion_tokens": getattr(usage, "candidates_token_count", None),
        "total_tokens": getattr(usage, "total_token_count", None),
    }

# ----------------------
# Debate State
# ----------------------
debate_state = {
    "case": None,
    "filters": None,          # user-provided retrieval filters
    "arguments": [],          # chronological events (rag/chaos/judge)
    "events": [],             # judge evidence/surprises
    "round_no": 0,
    "roles": {
        "rag": "RAG Lawyer (Prosecution)",
        "chaos": "Chaos Lawyer (Defense)"
    },
    "judge_decision": "Pending",
    "last_objection": None,
    "last_rag": None,
    "last_chaos": None,
    "closed": False,
    "pending_objection": None,  # carry objection to next round for explicit response
    "winner_key": None,         # records final winner idempotently
    "verdict_cache": None,      # caches verdict text after close to avoid duplicates
    "effects": {"rag": [], "chaos": []},  # one-round advantages/constraints
}

# ----------------------
# Utility
# ----------------------
def recent_events_text(n=2):
    if not debate_state["events"]:
        return ""
    tail = debate_state["events"][-n:]
    s = " | ".join(e.get("normalized", e.get("argument", "")) for e in tail if e)
    return f"Judge events: {s}" if s else ""

def now_utc_iso():
    return datetime.utcnow().isoformat() + "Z"

def grant_effect(side, kind, summary, expires_in_rounds=1, strength=1, source=None):
    # kind: "evidence_benefit", "constraint", "objection_benefit", etc.
    eff = {
        "kind": kind,
        "summary": (summary or "").strip(),
        "expires_round": (debate_state["round_no"] or 0) + expires_in_rounds,
        "strength": strength,
        "source": source,
        "created_at": now_utc_iso(),
    }
    debate_state["effects"][side].append(eff)

def pop_effects_for_round(side, round_no):
    # Return effects that become active in this round, and remove them
    bucket = debate_state["effects"][side]
    active = [e for e in bucket if e.get("expires_round") == round_no]
    debate_state["effects"][side] = [e for e in bucket if e not in active]
    return active

def split_effects(effects):
    # Separate “benefit” vs “constraint” for prompting
    benefits, constraints = [], []
    for e in effects:
        k = (e.get("kind") or "").lower()
        if "benefit" in k or k in ("evidence", "objection_sustained"):
            benefits.append(e)
        elif "constraint" in k:
            constraints.append(e)
        else:
            benefits.append(e)  # default
    return benefits, constraints

def summarize_effects(effs, max_len=280):
    lines = []
    for e in effs:
        s = e.get("summary", "")
        if not s:
            continue
        lines.append(s if len(s) <= max_len else s[:max_len] + "…")
    return lines

def mk_directives(benefit_lines, constraint_lines, side_word="you"):
    dirs = []
    if constraint_lines:
        dirs.append(f"Constraint for {side_word}: address and cure/narrow: " + " | ".join(constraint_lines))
    if benefit_lines:
        dirs.append(f"Leverage for {side_word}: use to support your theory: " + " | ".join(benefit_lines))
    return "\n".join(dirs) if dirs else "No special directives."

# ----------------------
# Health/state/reset
# ----------------------
@app.get("/state")
def state():
    return {
        "case": debate_state["case"],
        "roles": debate_state["roles"],
        "judge_decision": debate_state["judge_decision"],
        "closed": debate_state["closed"],
        "round_no": debate_state["round_no"],
        "num_events": len(debate_state["arguments"]),
        "filters": debate_state["filters"],
        "pending_objection": debate_state.get("pending_objection"),
        "winner_key": debate_state.get("winner_key"),
        "effects": debate_state["effects"],
    }

@app.post("/reset")
def reset():
    debate_state["case"] = None
    debate_state["filters"] = None
    debate_state["arguments"].clear()
    debate_state["events"].clear()
    debate_state["round_no"] = 0
    debate_state["roles"] = {
        "rag": "RAG Lawyer (Prosecution)",
        "chaos": "Chaos Lawyer (Defense)"
    }
    debate_state["judge_decision"] = "Pending"
    debate_state["last_objection"] = None
    debate_state["last_rag"] = None
    debate_state["last_chaos"] = None
    debate_state["closed"] = False
    debate_state["pending_objection"] = None
    debate_state["winner_key"] = None
    debate_state["verdict_cache"] = None
    debate_state["effects"] = {"rag": [], "chaos": []}
    return {"status": "ok", "message": "Debate reset."}

# ----------------------
# 0) Generate quirky case
# ----------------------
QUIRKY_CASES = [
    "A man sues a parrot for defamation.",
    "A robot accuses its owner of unpaid overtime.",
    "A pizza shop claims copyright over the phrase 'extra cheesy'.",
    "A landlord sues a plant for structural damage.",
    "A dog files a noise complaint against a vacuum cleaner.",
]

@app.post("/generate_case")
def generate_case(payload: dict = Body(None)):
    user_case = (payload or {}).get("case")
    if user_case and user_case.strip():
        scenario = user_case.strip()
    else:
        scenario = random.choice(QUIRKY_CASES)

    suggested = {
        "case_type": "defamation" if "defamation" in scenario.lower() else "property dispute",
        "jurisdiction": "US",
        "year_hint": None,
        "key_legal_principles": [],
    }
    return {"case": scenario, "suggested_metadata": suggested}

# ----------------------
# 1) Start Debate (accepts filters)
# ----------------------
@app.post("/debate")
def start_debate(payload: dict = Body(...)):
    case = payload.get("case")
    if not case:
        return {"error": "Provide 'case' in payload."}

    filters = payload.get("filters") or None

    if debate_state["case"] and not debate_state["closed"]:
        return {"error": "Debate already started. Use /next_round for further rounds."}

    if debate_state["closed"]:
        debate_state["arguments"].clear()
        debate_state["events"].clear()
        debate_state["round_no"] = 0
        debate_state["roles"] = {
            "rag": "RAG Lawyer (Prosecution)",
            "chaos": "Chaos Lawyer (Defense)"
        }
        debate_state["judge_decision"] = "Pending"
        debate_state["last_objection"] = None
        debate_state["last_rag"] = None
        debate_state["last_chaos"] = None
        debate_state["closed"] = False
        debate_state["pending_objection"] = None
        debate_state["winner_key"] = None
        debate_state["verdict_cache"] = None
        debate_state["effects"] = {"rag": [], "chaos": []}

    debate_state["case"] = case
    debate_state["filters"] = filters
    return next_round()

# ----------------------
# 2) Objection (targeted, in-place, same round) + asymmetric effects
# ----------------------
@app.post("/objection")
def objection(payload: dict = Body(...)):
    if debate_state["closed"]:
        return {"error": "Debate is closed. No objections allowed after verdict."}
    if not debate_state["arguments"]:
        return {"error": "No arguments to object to."}

    target = (payload or {}).get("target")  # 'rag' or 'chaos'
    reason = (payload or {}).get("reason")

    if target not in ("rag", "chaos"):
        return {"error": "Specify target as 'rag' or 'chaos' along with 'reason'."}

    # Find most recent argument from the targeted lawyer
    target_index = None
    for i in range(len(debate_state["arguments"]) - 1, -1, -1):
        a = debate_state["arguments"][i]
        if a.get("lawyer") == target:
            target_index = i
            break
    if target_index is None:
        return {"error": f"No arguments by {target} found to object to."}

    target_event = debate_state["arguments"][target_index]
    argument_text = target_event.get("argument", "")
    round_no = target_event.get("round")

    # Auto-reason via Gemini when not provided or "skip"
    if not reason or str(reason).lower() == "skip":
        judge_prompt = f"""
                You are a courtroom judge AI. Provide one concise, valid objection to the latest {target.upper()} argument.

                Argument:
                {argument_text}

                Rules:
                - 1 line only, 4–16 words.
                - Format: "Objection – <keyword>: <brief reason>."
                """.strip()
        auto, _, _ = gemini_generate(judge_prompt, model=GEMINI_MODEL_JUDGE)
        reason = (auto or "Objection – Relevance: not tied to key facts.").strip()

    now_iso = now_utc_iso()

    # Record judge event
    judge_event = {
        "lawyer": "judge",
        "role": "Judge",
        "event_type": "objection",
        "argument": f"{reason}",
        "target_index": target_index,
        "round": round_no,
        "created_at": now_iso,
        "provider": "gemini",
        "model": GEMINI_MODEL_JUDGE
    }
    debate_state["arguments"].append(judge_event)

    # Reframe targeted argument in-place (same round)
    if debate_state["roles"][target].endswith("(Prosecution)"):
        new_text = "• Element narrowed to one provable fact\n• Direct precedent cited\n• Causation tied to record"
    else:
        new_text = "• Limit claim to verifiable facts\n• Exclude speculation\n• Address objectioned element only"

    target_event["argument"] = new_text
    target_event["reframed"] = True
    target_event["objection_reason"] = reason
    target_event["reframed_at"] = now_iso

    # Cache last text
    if target == "rag":
        debate_state["last_rag"] = new_text
    else:
        debate_state["last_chaos"] = new_text

    debate_state["last_objection"] = {
        "objector": "judge",
        "reason": reason,
        "target_lawyer": target,
        "target_index": target_index,
        "round": round_no,
        "created_at": now_iso,
        "provider": "gemini",
        "model": GEMINI_MODEL_JUDGE
    }

    # Carry objection into next round for explicit first response
    debate_state["pending_objection"] = {
        "target": target,
        "reason": reason,
        "raised_round": round_no,
        "handled_in_round": None
    }

    # Asymmetric one-round effects: objection against target benefits the opponent
    beneficiary = "chaos" if target == "rag" else "rag"
    summary_txt = f"Objection noted: {reason}"
    grant_effect(beneficiary, kind="objection_benefit", summary=summary_txt, expires_in_rounds=1, source="objection")
    grant_effect(target, kind="constraint", summary=summary_txt, expires_in_rounds=1, source="objection")

    return {
        "status": "ok",
        "lawyer": target,
        "target_index": target_index,
        "target_round": round_no,
        "updated_argument": new_text,
        "objection_reason": reason,
    }

# ----------------------
# 3) Judge Decision (idempotent, closes debate)
# ----------------------
@app.post("/judge_decision")
def judge_decision(winner: str = Body(..., embed=True)):
    if winner not in ["rag", "chaos"]:
        return {"error": "Winner must be 'rag' or 'chaos'"}

    # Already decided: return current state without creating another event
    if debate_state.get("closed"):
        prev = debate_state.get("winner_key")
        decision_text = debate_state.get("judge_decision", "Pending")
        return {
            "status": "Already decided",
            "winner": prev,
            "winner_role": debate_state["roles"].get(prev) if prev else None,
            "judge_decision": decision_text,
            "closed": True
        }

    decision_text = f"Judge rules in favor of {debate_state['roles'][winner]}."
    debate_state["judge_decision"] = decision_text
    debate_state["winner_key"] = winner
    debate_state["closed"] = True
    debate_state["pending_objection"] = None  # no more objections after verdict

    debate_state["arguments"].append({
        "lawyer": "judge",
        "role": "Judge",
        "event_type": "decision",
        "argument": decision_text,
        "round": debate_state["round_no"],
        "created_at": now_utc_iso(),
        "provider": "system",
        "model": None
    })
    return {
        "status": "Decision recorded successfully.",
        "winner": winner,
        "winner_role": debate_state["roles"][winner],
        "judge_decision": decision_text,
        "closed": True
    }

# ----------------------
# 4) Role Reversal (role-aware prompts)
# ----------------------
@app.post("/role_reversal")
def role_reversal():
    if debate_state["closed"]:
        return {"error": "Debate is closed. Role reversal not allowed."}
    if debate_state["roles"]["rag"].endswith("(Prosecution)"):
        debate_state["roles"]["rag"] = "RAG Lawyer (Defense)"
        debate_state["roles"]["chaos"] = "Chaos Lawyer (Prosecution)"
    else:
        debate_state["roles"]["rag"] = "RAG Lawyer (Prosecution)"
        debate_state["roles"]["chaos"] = "Chaos Lawyer (Defense)"
    return {"status": "Roles reversed!", "new_roles": debate_state["roles"]}

# ----------------------
# 5) Judge Event: New Evidence/Surprise (Gemini-normalized) + asymmetric effects
# ----------------------
@app.post("/judge_event")
def judge_event(payload: dict = Body(...)):
    if debate_state["closed"]:
        return {"error": "Debate is closed."}
    event_type = (payload or {}).get("type")  # "evidence" | "surprise" | "constraint"
    content = (payload or {}).get("content")
    target = (payload or {}).get("target")  # optional: "rag" | "chaos" | None
    if not content:
        return {"error": "content required."}

    # Normalize evidence with Gemini
    norm_prompt = f"""
                You are a courtroom judge AI. Normalize the following new input into a concise record.

                Input:
                {content}

                Output format EXACTLY:
                Summary: <one sentence, 12–20 words, neutral, no new facts>.
                Implication: <one clause, ≤ 12 words, note effect on proof or defense>.
                """.strip()
    norm, _, _ = gemini_generate(norm_prompt, model=GEMINI_MODEL_JUDGE)
    normalized = norm or f"Summary: {content[:100]}... Implication: affects evaluation."

    now_iso = now_utc_iso()
    ev = {
        "lawyer": "judge",
        "role": "Judge",
        "event_type": event_type or "evidence",
        "argument": content,
        "normalized": normalized,
        "target": target,
        "created_at": now_iso,
        "round": debate_state["round_no"],
        "provider": "gemini",
        "model": GEMINI_MODEL_JUDGE
    }
    debate_state["events"].append(ev)
    debate_state["arguments"].append(ev)

    # Derive beneficiary/target effects (one round)
    if target in ("rag", "chaos"):
        beneficiary = "chaos" if target == "rag" else "rag"
        summary_txt = (normalized or content).strip()
        grant_effect(beneficiary, kind="evidence_benefit", summary=summary_txt, expires_in_rounds=1, source="judge_event")
        grant_effect(target, kind="constraint", summary=summary_txt, expires_in_rounds=1, source="judge_event")

    return {"status": "ok", "event": ev}

# ----------------------
# 6) Verdict summary (idempotent; no duplicates)
# ----------------------
@app.get("/verdict")
def verdict():
    if not debate_state["arguments"]:
        return {"verdict": "No debate yet."}

    # If case closed and cached verdict exists, return it
    if debate_state.get("closed") and debate_state.get("verdict_cache"):
        return {"verdict": debate_state["verdict_cache"]["text"]}

    debate_text = "\n".join([f"{a.get('role','Unknown')}: {a.get('argument','')}" for a in debate_state["arguments"]])
    judge_text = f"Judge Decision: {debate_state['judge_decision']}"

    prompt = f"""
                Summarize the debate strictly from the transcript below. Be concise and neutral.
                Rules:
                - 3–4 sentences total.
                - No new facts or embellishment.
                - Final sentence: who prevailed and the single key reason.

                Debate:
                {debate_text}

                {judge_text}
                """.strip()

    summary, _, _ = gemini_generate(prompt, model=GEMINI_MODEL_JUDGE)
    verdict_text = summary or f"Final Verdict: {debate_state['judge_decision']}"

    # Append verdict event only once per closed case
    if debate_state.get("closed"):
        already_appended = any(e.get("event_type") == "verdict" for e in debate_state["arguments"])
        if not already_appended:
            event = {
                "lawyer": "judge",
                "role": "Judge",
                "event_type": "verdict",
                "argument": verdict_text,
                "round": debate_state["round_no"],
                "created_at": now_utc_iso(),
                "provider": "gemini",
                "model": GEMINI_MODEL_JUDGE
            }
            debate_state["arguments"].append(event)
        # Cache verdict to avoid duplicates
        debate_state["verdict_cache"] = {
            "text": verdict_text,
            "round_no": debate_state["round_no"],
            "created_at": now_utc_iso()
        }

    return {"verdict": verdict_text}

# ----------------------
# 7) Next Round (concise, role-aware, filters, events, objection-first, asymmetric effects)
# ----------------------
@app.post("/next_round")
def next_round():
    if debate_state["closed"]:
        return {"error": "Debate is closed. Verdict recorded. No further rounds allowed."}
    if not debate_state["case"]:
        return {"error": "No debate started. Use /debate to start first."}

    # Increment round
    debate_state["round_no"] += 1
    round_no = debate_state["round_no"]

    # Effects that come due this round (created in previous round)
    rag_effects = pop_effects_for_round("rag", round_no)
    chaos_effects = pop_effects_for_round("chaos", round_no)

    rag_benefits, rag_constraints = split_effects(rag_effects)
    chaos_benefits, chaos_constraints = split_effects(chaos_effects)

    rag_benefit_lines = summarize_effects(rag_benefits)
    rag_constraint_lines = summarize_effects(rag_constraints)
    chaos_benefit_lines = summarize_effects(chaos_benefits)
    chaos_constraint_lines = summarize_effects(chaos_constraints)

    # Retrieval with per-side augmentation
    base_case = debate_state["case"]
    rag_query_extra = " ".join(rag_benefit_lines + rag_constraint_lines)
    chaos_query_extra = " ".join(chaos_benefit_lines + chaos_constraint_lines)

    # RAG retrieval (augmented)
    rag_query = f"{base_case} {rag_query_extra}".strip()
    rag_retrieved = retrieve_case(rag_query, filters=debate_state.get("filters"), top_k=3)
    if rag_retrieved:
        rag_best, _ = rag_retrieved[0]
        rag_context = (
            f"Precedent: {rag_best.get('case_name','Unknown')} "
            f"({rag_best.get('year_of_judgment','N/A')}) — {rag_best.get('case_outcome','Unknown')}"
        )
        rag_citation = f"{rag_best.get('case_name','Unknown')}, {rag_best.get('year_of_judgment','N/A')}"
        rag_case_meta = rag_best
    else:
        rag_context = "No directly relevant case; use foundational legal principles."
        rag_citation = None
        rag_case_meta = {}

    # CHAOS retrieval (augmented)
    chaos_query = f"{base_case} {chaos_query_extra}".strip()
    chaos_retrieved = retrieve_case(chaos_query, filters=debate_state.get("filters"), top_k=3)
    if chaos_retrieved:
        chaos_best, _ = chaos_retrieved[0]
        chaos_context = (
            f"Precedent: {chaos_best.get('case_name','Unknown')} "
            f"({chaos_best.get('year_of_judgment','N/A')}) — {chaos_best.get('case_outcome','Unknown')}"
        )
        chaos_citation = f"{chaos_best.get('case_name','Unknown')}, {chaos_best.get('year_of_judgment','N/A')}"
        chaos_case_meta = chaos_best
    else:
        chaos_context = "No directly relevant case; use foundational legal principles."
        chaos_citation = None
        chaos_case_meta = {}

    events_text = recent_events_text()
    last_chaos = debate_state.get("last_chaos") or "(no prior argument)"
    last_rag = debate_state.get("last_rag") or "(no prior argument)"

    rag_is_prosecution = debate_state["roles"]["rag"].endswith("(Prosecution)")
    chaos_is_prosecution = debate_state["roles"]["chaos"].endswith("(Prosecution)")

    # Objection carry-over: targeted side must answer; opponent capitalizes
    pending = debate_state.get("pending_objection")
    address_now = (
        pending is not None
        and pending.get("handled_in_round") != round_no
        and (pending.get("raised_round") is None or pending.get("raised_round") < round_no)
    )

    rag_objection_block = "Objection: None."
    chaos_objection_block = "Objection: None."
    last_obj_text = ""

    if pending:
        last_obj_text = f'Judge objection: {pending.get("reason","")}'
        tgt = pending.get("target")
        if address_now:
            if tgt == "rag":
                rag_objection_block = f'''Objection response:
- Quote: "{pending["reason"]}"
- In 2–3 crisp sentences, cure/narrow precisely and move on.'''
                chaos_objection_block = '''Capitalize on objection:
- Add 1 short line showing opponent’s claim was narrowed or weakened.'''
            elif tgt == "chaos":
                chaos_objection_block = f'''Objection response:
- Quote: "{pending["reason"]}"
- In 2–3 crisp sentences, cure/narrow precisely and move on.'''
                rag_objection_block = '''Capitalize on objection:
- Add 1 short line showing opponent’s claim was narrowed or weakened.'''

    # Evidence/effect directives injected into prompts
    rag_directives = mk_directives(rag_benefit_lines, rag_constraint_lines, "you")
    chaos_directives = mk_directives(chaos_benefit_lines, chaos_constraint_lines, "you")

    # --- RAG Lawyer via Azure (fallback to Gemini) ---
    rag_role_word = "Prosecution" if rag_is_prosecution else "Defense"
    rag_prompt = f"""
                Act as {rag_role_word}.

                Case: {debate_state['case']}
                Context: {rag_context}
                {events_text}
                {last_obj_text}

                Directives:
                {rag_directives}

                {rag_objection_block}

                Direct response to opponent (one sentence): Respond concisely to:
                "{last_chaos}"

                Now provide your main argument:
                - Please stick to Indian English not too difficult terms
                - Try to think and verify your points be like a professional Lawyer
                - 4 short bullet points (10–18 words each).
                - Keep total length under 240 words.
                - Avoid rhetoric; argue facts and law only.
                """.strip()

    rag_text = None
    rag_usage = None
    provider = "azure"
    model_used = AZURE_DEPLOYMENT

    try:
        if azure_client is None:
            raise RuntimeError("Azure client not configured")
        rag_resp = azure_client.chat.completions.create(
            model=AZURE_DEPLOYMENT,
            messages=[{"role": "user", "content": rag_prompt}],
            max_tokens=300,
            temperature=0.3
        )
        rag_text = (rag_resp.choices[0].message.content or "").strip()
        usage_obj = getattr(rag_resp, "usage", None)
        if usage_obj:
            rag_usage = {
                "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                "total_tokens": getattr(usage_obj, "total_tokens", None),
            }
    except Exception as e:
        print(f"[Azure RAG] fallback: {e}")
        provider = "gemini"
        model_used = GEMINI_MODEL_JUDGE
        alt_text, _, alt_usage = gemini_generate(rag_prompt, model=GEMINI_MODEL_JUDGE)
        rag_text = alt_text or "• Unable to proceed due to configuration\n• Provide minimal legal analysis\n• Request brief adjournment"
        if alt_usage:
            rag_usage = usage_to_tokens(alt_usage)

    rag_argument = {
        "lawyer": "rag",
        "role": debate_state["roles"]["rag"],
        "round": round_no,
        "argument": rag_text,
        "prompt": rag_prompt,
        "citation": rag_citation,
        "metadata": rag_case_meta,
        "created_at": now_utc_iso(),
        "provider": provider,
        "model": model_used
    }
    if rag_usage:
        rag_argument["usage"] = rag_usage

    # --- Chaos Lawyer via Gemini (role-aware) ---
    chaos_role_word = "Prosecution" if chaos_is_prosecution else "Defense"
    chaos_tone_hint = ""
    if chaos_benefit_lines:
        chaos_tone_hint = "- Tone: confident and a little pleased (not gloating)."

    chaos_prompt = f"""
            Act as {chaos_role_word}.

            Case: {debate_state['case']}
            Context: {chaos_context}
            {events_text}
            {last_obj_text}

            Directives:
            {chaos_directives}

            {chaos_objection_block}

            Direct response to opponent (one sentence): Respond concisely to:
            "{last_rag}"

            Now provide your main argument:
            - Please stick to Indian English not too difficult terms
            - Don't think before answering. Try to be funny
            - 4 short bullet points (10–18 words each).
            - Keep total length under 240 words.
            - Avoid rhetoric; argue facts and law only.
            {chaos_tone_hint}
            """.strip()

    chaos_text, chaos_resp, chaos_usage = gemini_generate(chaos_prompt, model=GEMINI_MODEL_CHAOS)
    if not chaos_text:
        chaos_text = "• Defense cannot elaborate now\n• Limits position to verifiable facts\n• Requests clarification"

    chaos_argument = {
        "lawyer": "chaos",
        "role": debate_state["roles"]["chaos"],
        "round": round_no,
        "argument": chaos_text,
        "prompt": chaos_prompt,
        "citation": chaos_citation,
        "metadata": chaos_case_meta,
        "rhetoric": None,
        "created_at": now_utc_iso(),
        "provider": "gemini",
        "model": GEMINI_MODEL_CHAOS
    }
    if chaos_usage:
        chaos_argument["usage"] = usage_to_tokens(chaos_usage)

    # Update state
    debate_state["arguments"].extend([rag_argument, chaos_argument])
    debate_state["last_rag"] = rag_text
    debate_state["last_chaos"] = chaos_text

    # Clear the pending objection immediately after the succeeding round and mark handled
    if address_now and debate_state.get("pending_objection"):
        debate_state["pending_objection"]["handled_in_round"] = round_no
        debate_state["pending_objection"] = None
        debate_state["last_objection"] = None

    return {
        "case": debate_state["case"],
        "rag_lawyer": rag_argument,
        "chaos_lawyer": chaos_argument,
        "judge_decision": debate_state["judge_decision"]
    }
