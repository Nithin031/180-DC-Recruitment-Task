import json
import streamlit as st
import requests

st.set_page_config(page_title="Courtroom Clash", layout="wide")
st.title("Courtroom Clash: AI Lawyers Battle It Out")

# Sidebar
if "backend_url" not in st.session_state:
    st.session_state.backend_url = "http://127.0.0.1:8000"

st.sidebar.header("Settings")
st.session_state.backend_url = st.sidebar.text_input("Backend URL", st.session_state.backend_url)


def reset_ui():
    for k in [
        "timeline", "debate_started", "case_query", "judge_decision", "closed", "filters",
        "events_log", "verdict_submitted", "last_summary", "effects", "pending_objection",
        "roles", "round_no"
    ]:
        if k in st.session_state:
            del st.session_state[k]


if st.sidebar.button("Reset UI (client)"):
    reset_ui()
    st.rerun()

if st.sidebar.button("Reset Backend Debate"):
    try:
        resp = requests.post(f"{st.session_state.backend_url}/reset", timeout=30)
        if resp.ok:
            st.sidebar.success("Backend reset.")
        else:
            st.sidebar.error(f"Reset failed: {resp.status_code}")
    except Exception as e:
        st.sidebar.error(f"Reset error: {e}")

# Session state
if "timeline" not in st.session_state:
    st.session_state.timeline = []  # [{round, rag:{}, chaos:{}, rag_objection, chaos_objection}]
if "debate_started" not in st.session_state:
    st.session_state.debate_started = False
if "case_query" not in st.session_state:
    st.session_state.case_query = ""
if "judge_decision" not in st.session_state:
    st.session_state.judge_decision = "Pending"
if "closed" not in st.session_state:
    st.session_state.closed = False
if "filters" not in st.session_state:
    st.session_state.filters = {}
if "events_log" not in st.session_state:
    st.session_state.events_log = []
if "verdict_submitted" not in st.session_state:
    st.session_state.verdict_submitted = False
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""
if "effects" not in st.session_state:
    st.session_state.effects = {"rag": [], "chaos": []}
if "pending_objection" not in st.session_state:
    st.session_state.pending_objection = None
if "roles" not in st.session_state:
    st.session_state.roles = {"rag": "RAG Lawyer (Prosecution)", "chaos": "Chaos Lawyer (Defense)"}
if "round_no" not in st.session_state:
    st.session_state.round_no = 0

BACKEND_URL = st.session_state.backend_url

# Helpers
def post_json(path, payload=None, timeout=90):
    url = f"{BACKEND_URL}{path}"
    try:
        r = requests.post(url, json=payload or {}, timeout=timeout)
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        if not r.ok:
            data.setdefault("error", f"HTTP {r.status_code}")
        return data
    except Exception as e:
        return {"error": str(e)}

def get_json(path, timeout=90):
    url = f"{BACKEND_URL}{path}"
    try:
        r = requests.get(url, timeout=timeout)
        data = r.json() if r.headers.get("content-type","").startswith("application/json") else {}
        if not r.ok:
            data.setdefault("error", f"HTTP {r.status_code}")
        return data
    except Exception as e:
        return {"error": str(e)}

def refresh_state():
    s = get_json("/state")
    if "error" not in s:
        st.session_state.judge_decision = s.get("judge_decision", "Pending")
        st.session_state.closed = s.get("closed", False)
        st.session_state.roles = s.get("roles", st.session_state.roles)
        st.session_state.effects = s.get("effects", st.session_state.effects)
        st.session_state.pending_objection = s.get("pending_objection", None)
        st.session_state.round_no = s.get("round_no", st.session_state.round_no)
    return s

def render_argument(label, arg):
    st.subheader(label)
    st.write(arg.get("argument",""))
    if arg.get("citation"):
        st.caption(f"Citation: {arg['citation']}")
    colA, colB = st.columns([1,1])
    with colA:
        st.caption(f"Provider: {arg.get('provider','N/A')}")
        st.caption(f"Model: {arg.get('model','N/A')}")
        st.caption(f"Created: {arg.get('created_at','N/A')}")
    with colB:
        usage = arg.get("usage",{})
        if usage:
            st.caption(f"Tokens — prompt: {usage.get('prompt_tokens','?')}, completion: {usage.get('completion_tokens','?')}, total: {usage.get('total_tokens','?')}")
    if arg.get("metadata"):
        with st.expander("Precedent Metadata"):
            for k,v in arg["metadata"].items():
                st.markdown(f"- {k}: {v}")
    if arg.get("prompt"):
        with st.expander("Prompt Used"):
            st.code(arg["prompt"])

def is_benefit(kind: str) -> bool:
    k = (kind or "").lower()
    # Treat unknown kinds not containing "constraint" as benefits by default
    return ("benefit" in k) or (k in ("evidence", "objection_sustained")) or ("constraint" not in k)

def group_effects(effects_list):
    benefits, constraints = [], []
    for eff in effects_list or []:
        if is_benefit(eff.get("kind","")):
            benefits.append(eff)
        else:
            constraints.append(eff)
    return benefits, constraints

def preview_effects_for_next_round(effects_by_side, current_round_no: int):
    # Backend pops effects with expires_round == next round_no at the start of /next_round
    target_round = (current_round_no or 0) + 1
    out = {"rag": {"benefits": [], "constraints": []}, "chaos": {"benefits": [], "constraints": []}}
    for side in ["rag","chaos"]:
        upcoming = [e for e in (effects_by_side.get(side) or []) if e.get("expires_round") == target_round]
        ben, con = group_effects(upcoming)
        out[side]["benefits"] = ben
        out[side]["constraints"] = con
    return out

def render_effects_preview(effects_preview):
    st.subheader("Effects Preview (will apply next round)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"#### {st.session_state.roles.get('rag','RAG')}")
        ben = effects_preview["rag"]["benefits"]
        con = effects_preview["rag"]["constraints"]
        if not ben and not con:
            st.caption("No special directives.")
        if ben:
            st.success("Leverage:")
            for e in ben:
                st.markdown(f"- {e.get('summary','')}")
        if con:
            st.warning("Constraint:")
            for e in con:
                st.markdown(f"- {e.get('summary','')}")
    with c2:
        st.markdown(f"#### {st.session_state.roles.get('chaos','CHAOS')}")
        ben = effects_preview["chaos"]["benefits"]
        con = effects_preview["chaos"]["constraints"]
        if not ben and not con:
            st.caption("No special directives.")
        if ben:
            st.success("Leverage:")
            for e in ben:
                st.markdown(f"- {e.get('summary','')}")
        if con:
            st.warning("Constraint:")
            for e in con:
                st.markdown(f"- {e.get('summary','')}")


# Case generation and filters
st.header("Case and Retrieval Controls")

gen_col1, gen_col2, _ = st.columns([3,1,1])
with gen_col1:
    case_input = st.text_area("Enter a case scenario (optional, leave empty for random):", value=st.session_state.case_query, height=80)
with gen_col2:
    if st.button("Generate Case", key="btn_generate_case"):
        payload = {"case": case_input.strip()} if case_input.strip() else {}
        data = post_json("/generate_case", payload)
        if "error" in data:
            st.error(data["error"])
        else:
            st.session_state.case_query = data.get("case","")
            st.success("Case generated. You can edit it above.")

with st.expander("RAG Filters (optional)"):
    case_type = st.text_input("Case Type (e.g., defamation, property dispute)")
    jurisdiction = st.text_input("Jurisdiction (e.g., US, UK, EU)")
    year = st.text_input("Year of Judgment (e.g., 1994)")
    principles = st.text_input("Key Legal Principles (comma-separated, e.g., libel, intent)")
    if st.button("Save Filters", key="btn_save_filters"):
        filt = {}
        if case_type.strip(): filt["case_type"] = case_type.strip()
        if jurisdiction.strip(): filt["jurisdiction"] = jurisdiction.strip()
        if year.strip(): filt["year_of_judgment"] = year.strip()
        if principles.strip():
            filt["key_legal_principles"] = [p.strip() for p in principles.split(",") if p.strip()]
        st.session_state.filters = filt
        st.success("Filters saved.")

# Start debate
st.header("Judge: Start Debate")
if not st.session_state.debate_started:
    if st.button("Start Debate", key="btn_start_debate"):
        if not st.session_state.case_query.strip():
            st.warning("Please generate or enter a case first.")
        else:
            with st.spinner("Starting debate..."):
                data = post_json("/debate", {"case": st.session_state.case_query.strip(), "filters": st.session_state.filters})
                if "error" in data:
                    st.error(f"Failed to start debate: {data['error']}")
                else:
                    round_entry = {
                        "round": data.get("rag_lawyer",{}).get("round") or 1,
                        "rag": data.get("rag_lawyer",{}),
                        "chaos": data.get("chaos_lawyer",{}),
                        "rag_objection": None,
                        "chaos_objection": None
                    }
                    st.session_state.timeline = [round_entry]
                    st.session_state.debate_started = True
                    st.session_state.judge_decision = data.get("judge_decision","Pending")
                    refresh_state()
                    st.success("Debate started.")

# Next round and events
if st.session_state.debate_started:
    st.header("Proceedings")

    # Roles + round status
    rr1, rr2, rr3 = st.columns([2,1,1])
    with rr1:
        st.caption(f"Roles — RAG: {st.session_state.roles.get('rag','RAG')}; CHAOS: {st.session_state.roles.get('chaos','CHAOS')}")
    with rr2:
        st.caption(f"Round: {st.session_state.round_no}")
    with rr3:
        st.caption(f"Decision: {st.session_state.judge_decision}")

    # Pending objection banner (who cures vs who capitalizes)
    if st.session_state.pending_objection:
        po = st.session_state.pending_objection
        tgt = po.get("target")
        reason = po.get("reason","")
        st.warning(f"Pending Objection: Target={tgt.upper()} | Reason: {reason}")
        cpo1, cpo2 = st.columns(2)
        if tgt == "rag":
            with cpo1:
                st.error("RAG must cure/narrow in next round.")
            with cpo2:
                st.success("CHAOS will capitalize on objection.")
        elif tgt == "chaos":
            with cpo1:
                st.error("CHAOS must cure/narrow in next round.")
            with cpo2:
                st.success("RAG will capitalize on objection.")

    # Effects preview for next round (derived from /state, before pressing Next Round)
    effects_preview = preview_effects_for_next_round(st.session_state.effects, st.session_state.round_no)
    with st.expander("Show Effects Preview (applies next round)", expanded=True):
        render_effects_preview(effects_preview)

    prc1, prc2 = st.columns([1,2])
    with prc1:
        if st.button("Next Round", key="btn_next_round", disabled=st.session_state.closed):
            with st.spinner("Running next round..."):
                data = post_json("/next_round")
                if "error" in data:
                    st.error(data["error"])
                else:
                    round_entry = {
                        "round": data.get("rag_lawyer",{}).get("round") or (len(st.session_state.timeline)+1),
                        "rag": data.get("rag_lawyer",{}),
                        "chaos": data.get("chaos_lawyer",{}),
                        "rag_objection": None,
                        "chaos_objection": None
                    }
                    st.session_state.timeline.append(round_entry)
                    st.session_state.judge_decision = data.get("judge_decision","Pending")
                    # Effects used this round will be consumed on backend; refresh to reflect it
                    refresh_state()
                    st.success(f"Round {round_entry['round']} added.")

    with prc2:
        st.subheader("Introduce Evidence or Surprise (AI-normalized)")
        ev_c1, ev_c2, ev_c3 = st.columns([3,1,1])
        with ev_c1:
            ev_text = st.text_input("Describe new evidence/surprise/constraint", key="ev_text", disabled=st.session_state.closed)
        with ev_c2:
            ev_type = st.selectbox("Type", ["evidence","surprise","constraint"], key="ev_type", disabled=st.session_state.closed)
        with ev_c3:
            ev_target = st.selectbox("Target (optional)", ["none","rag","chaos"], key="ev_target", disabled=st.session_state.closed)
        if st.button("Submit Event", key="btn_submit_event", disabled=st.session_state.closed):
            if not ev_text.strip():
                st.warning("Enter event text.")
            else:
                payload = {"type": ev_type, "content": ev_text.strip(), "target": (None if ev_target=="none" else ev_target)}
                resp = post_json("/judge_event", payload)
                if "error" in resp:
                    st.error(resp["error"])
                else:
                    ev = resp.get("event", {})
                    st.session_state.events_log.append(ev)
                    refresh_state()  # update effects preview immediately
                    st.success("Event recorded and normalized. Effects will apply next round.")

# Timeline
if st.session_state.debate_started and st.session_state.timeline:
    st.header("Debate Timeline")
    for entry in st.session_state.timeline:
        st.markdown(f"### Round {entry['round']}")
        c1, c2 = st.columns(2)
        with c1:
            render_argument(entry["rag"].get("role","RAG"), entry["rag"])
            if entry.get("rag_objection"):
                st.markdown(f"- Objection: {entry['rag_objection']}")
        with c2:
            render_argument(entry["chaos"].get("role","CHAOS"), entry["chaos"])
            if entry.get("chaos_objection"):
                st.markdown(f"- Objection: {entry['chaos_objection']}")

    # Events log with clearer labeling
    if st.session_state.events_log:
        with st.expander("Judge Events (latest first)"):
            for ev in reversed(st.session_state.events_log):
                etype = ev.get("event_type","evidence").upper()
                tgt = ev.get("target", None)
                tgt_str = f" → {tgt.upper()}" if tgt in ("rag","chaos") else ""
                norm = ev.get("normalized", ev.get("argument",""))
                st.markdown(f"- [{etype}{tgt_str}] {norm}")

# Judge controls
if st.session_state.debate_started:
    st.header("Judge Controls")
    st.info(f"Current Judge Decision: {st.session_state.judge_decision}")
    if st.session_state.closed:
        st.warning("Debate is closed. No further actions allowed.")

    ctrl1, ctrl2 = st.columns([1,2])

    # Role reversal
    with ctrl1:
        if st.button("Role Reversal", key="btn_role_reversal", disabled=st.session_state.closed):
            with st.spinner("Reversing roles..."):
                data = post_json("/role_reversal")
                if "error" in data:
                    st.error(data["error"])
                else:
                    refresh_state()
                    st.success("Roles reversed for next round.")

    # Targeted objection (asymmetric effects kick in next round)
    with ctrl2:
        st.subheader("Raise Objection (same-round update)")
        obj_c1, obj_c2, obj_c3 = st.columns([1,2,1])
        with obj_c1:
            target = st.radio("Target", ["rag","chaos"], horizontal=True, key="obj_target", disabled=st.session_state.closed)
        with obj_c2:
            reason = st.text_input("Reason (or leave empty and click Auto)", key="obj_reason", disabled=st.session_state.closed)
        with obj_c3:
            auto = st.button("Auto", key="btn_auto_objection", disabled=st.session_state.closed)

        submit = st.button("Submit Objection", key="btn_submit_objection", disabled=st.session_state.closed)
        if (submit or auto) and not st.session_state.closed:
            payload = {"target": target, "reason": (reason.strip() if not auto else "skip")}
            with st.spinner("Submitting objection..."):
                resp = post_json("/objection", payload)
                if "error" in resp:
                    st.error(resp["error"])
                else:
                    t = resp.get("lawyer")
                    round_no = resp.get("target_round")
                    updated = resp.get("updated_argument")
                    obj_reason = resp.get("objection_reason","")

                    # Find round entry by round number
                    idx = None
                    for i,e in enumerate(st.session_state.timeline):
                        if e.get("round") == round_no:
                            idx = i
                            break
                    if idx is None and st.session_state.timeline:
                        idx = len(st.session_state.timeline)-1

                    if idx is not None and updated:
                        if t == "rag":
                            st.session_state.timeline[idx]["rag"]["argument"] = updated
                            st.session_state.timeline[idx]["rag_objection"] = obj_reason
                        elif t == "chaos":
                            st.session_state.timeline[idx]["chaos"]["argument"] = updated
                            st.session_state.timeline[idx]["chaos_objection"] = obj_reason
                        st.success(f"Objection applied to {t.upper()} in Round {st.session_state.timeline[idx]['round']}")

                    # Refresh to show pending_objection and effects preview
                    refresh_state()
                    st.info("Objection noted: target must cure next round; opponent will capitalize.")

# Judge Verdict (atomic form, idempotent backend, no duplicate summary)
if st.session_state.debate_started:
    st.header("Judge Verdict")

    # Separate summary fetch (GET only)
    summary_btn = st.button("Get Current Summary", key="btn_summary_only", disabled=False)

    with st.form("verdict_form", clear_on_submit=False):
        winner = st.radio("Choose the winner:", ["rag","chaos"], key="verdict_choice_radio", disabled=st.session_state.closed)
        submit_verdict = st.form_submit_button("Submit Verdict and Get Summary", disabled=st.session_state.closed)

    if submit_verdict and not st.session_state.closed and not st.session_state.verdict_submitted:
        with st.spinner("Submitting verdict and fetching summary..."):
            data = post_json("/judge_decision", {"winner": winner})
            if "error" in data:
                st.error(data["error"])
            else:
                st.session_state.verdict_submitted = True
                st.session_state.judge_decision = data.get("judge_decision","Pending")
                st.session_state.closed = True
                summary = get_json("/verdict").get("verdict","Summary not available.")
                st.session_state["last_summary"] = summary
                st.success(f"Verdict submitted: {winner}")
                st.info(summary)
                # Optional: st.rerun()

    if summary_btn:
        with st.spinner("Fetching summary..."):
            summary = get_json("/verdict").get("verdict","Summary not available.")
            st.session_state["last_summary"] = summary
            st.info(summary)

# Footer
with st.expander("Sync with Backend State"):
    if st.button("Refresh State", key="btn_refresh_state"):
        s = refresh_state()
        if "error" in s:
            st.error(s["error"])
        else:
            st.write(s)
