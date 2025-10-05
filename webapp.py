# webapp.py
# Streamlit Prompt Engineering Workbench using Google Gemini 2.5 Flash Lite
# -------------------------------------------------------------
# How to run locally:
#   1) pip install -r requirements.txt
#   2) Set your key in environment:  export GOOGLE_API_KEY="your_key_here"  (macOS/Linux)
#                                    setx GOOGLE_API_KEY "your_key_here"   (Windows, new terminal)
#   3) streamlit run webapp.py
#
# Deploy on Streamlit Community Cloud:
#   - Push webapp.py and requirements.txt to GitHub
#   - Create a new Streamlit app from that repo
#   - In "Secrets", add GOOGLE_API_KEY: your_key_here
#
# Notes:
#   - This app focuses on prompt engineering. You can integrate your ML model
#     by implementing the stub `run_project_inference()` below.

import os
import time
import json
from typing import List, Dict, Any, Optional

import streamlit as st
import pandas as pd

# Google Generative AI SDK
import google.generativeai as genai

# ------------------ Config ------------------
# Read key from Streamlit secrets if present; else from env.
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", os.getenv("GOOGLE_API_KEY", ""))

# Fail early with a friendly hint if key missing
if not GOOGLE_API_KEY:
    st.warning("GOOGLE_API_KEY is not set. Add it in Streamlit Secrets or your environment.", icon="⚠️")

# Configure the client and pick the model
try:
    genai.configure(api_key=GOOGLE_API_KEY)
    MODEL_NAME = "gemini-2.5-flash-lite"
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    # We'll still render the UI; calls will fail gracefully.
    model = None
    st.error(f"Problem configuring Google Generative AI client: {e}")

# ------------------ Utilities ------------------
def call_gemini(prompt: str,
                temperature: float = 0.4,
                top_p: float = 0.95,
                system_instruction: Optional[str] = None) -> str:
    """Call Gemini with a single prompt and return text."""
    if model is None:
        return "Gemini client not configured (missing or invalid GOOGLE_API_KEY)."
    try:
        generation_config = {
            "temperature": temperature,
            "top_p": top_p,
        }
        # For latest SDKs, you can pass system_instruction to GenerativeModel at init;
        # here we simulate by prefixing if provided.
        full_prompt = f"System: {system_instruction}\n\nUser: {prompt}" if system_instruction else prompt
        resp = model.generate_content(full_prompt, generation_config=generation_config)
        # Newer SDK returns resp.text; older may require parsing candidates
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"[Gemini error] {e}"

def chat_gemini(messages: List[Dict[str, str]], temperature: float = 0.5, top_p: float = 0.95) -> str:
    """Chat-style interface: messages = [{'role':'user'|'model','content':'...'}, ...]"""
    if model is None:
        return "Gemini client not configured (missing or invalid GOOGLE_API_KEY)."
    try:
        chat = model.start_chat(history=messages[:-1] if messages else [])
        last_user_msg = messages[-1]["content"] if messages else ""
        resp = chat.send_message(last_user_msg, generation_config={"temperature": temperature, "top_p": top_p})
        return getattr(resp, "text", str(resp))
    except Exception as e:
        return f"[Gemini chat error] {e}"

def run_project_inference(user_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Stub for your project model inference.
    Replace with calls to your trained model / pipeline.
    user_input: dict built from the UI (could be text or structured features).
    Returns a dictionary that the UI will render.
    """
    # TODO: integrate your model here (e.g., XGBoostClassifier). Example:
    # y_pred = clf.predict(pd.DataFrame([user_input]))[0]
    # proba = clf.predict_proba(pd.DataFrame([user_input]))[0].tolist()
    # return {"prediction": str(y_pred), "probas": proba}
    return {"note": "Project inference not implemented yet. Replace run_project_inference() with your logic.",
            "echo_input": user_input}

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Prompt Engineering Workbench (Gemini 2.5 Flash Lite)", page_icon="✨", layout="wide")

st.title("✨ Prompt Engineering Workbench — Gemini 2.5 Flash Lite")
st.caption("Build, test, and optimize prompts; chat with the model; and (optionally) call your own project inference.")

with st.expander("🔑 API & Session", expanded=False):
    st.write("The app reads **GOOGLE_API_KEY** from Streamlit Secrets or your environment.")
    if GOOGLE_API_KEY:
        st.success("GOOGLE_API_KEY found.", icon="✅")
    else:
        st.info("Set GOOGLE_API_KEY in **Settings → Secrets** (Streamlit Cloud) or your local environment.", icon="ℹ️")

col_left, col_right = st.columns([2, 1], gap="large")

with col_right:
    st.subheader("⚙️ Generation Settings")
    temperature = st.slider("temperature", 0.0, 1.0, 0.4, 0.05, help="Higher = more creative")
    top_p = st.slider("top_p", 0.0, 1.0, 0.95, 0.01, help="Nucleus sampling")

    st.subheader("🧭 Guidance")
    system_instruction = st.text_area(
        "System instruction (optional)",
        value="You are an expert prompt engineer. Be concise and actionable.",
        height=120
    )

with col_left:
    tab_build, tab_batch, tab_opt, tab_chat, tab_project = st.tabs(
        ["🛠️ Build Prompt", "📦 Batch Tester", "🚀 Optimizer", "💬 Chat Sandbox", "🧪 Project Hook"]
    )

    # -------- Build Prompt Tab --------
    with tab_build:
        st.subheader("🛠️ Prompt Builder")
        with st.form("builder"):
            objective = st.text_area("Objective / Task", height=120,
                                     placeholder="Describe what you want the model to do...")
            constraints = st.text_area("Constraints / Rules (optional)",
                                       placeholder="- Be concise\n- Use bullet points\n- Cite sources if possible")
            variables_raw = st.text_area("Variables (JSON, optional)",
                                         placeholder='{"product":"wireless earbuds","audience":"runners"}')
            examples = st.text_area("Few-shot Examples (optional)",
                                    placeholder="Input: ...\nOutput: ...\n---\nInput: ...\nOutput: ...")
            submit_build = st.form_submit_button("Generate Draft Prompt ✨")

        if submit_build:
            try:
                variables = json.loads(variables_raw) if variables_raw.strip() else {}
            except json.JSONDecodeError as e:
                st.error(f"Variables JSON error: {e}")
                variables = {}

            template = f"""You are assisting with prompt engineering.
Goal:
{objective}

Constraints:
{constraints or 'None'}

Variables (JSON):
{json.dumps(variables, indent=2) if variables else 'None'}

Few-shot examples:
{examples or 'None'}

Task:
Draft a high-quality prompt for a generative model. Include:
- Clear role and objective
- Input placeholders for variables (if any)
- Output format instructions
- Evaluation checklist
Return only the prompt block.
"""
            with st.spinner("Generating prompt…"):
                draft = call_gemini(template, temperature=temperature, top_p=top_p, system_instruction=system_instruction)
            st.markdown("#### 📄 Draft Prompt")
            st.code(draft, language="markdown")

            st.download_button("Download Prompt.md", data=draft, file_name="prompt_draft.md", mime="text/markdown")

    # -------- Batch Tester Tab --------
    with tab_batch:
        st.subheader("📦 Batch Tester")
        st.write("Run a prompt across many inputs and collect outputs. Upload a CSV with a column named `input`.")
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        prompt_for_batch = st.text_area("Prompt (use {input} placeholder)", height=120,
                                        placeholder="Summarize in 3 bullets: {input}")
        run_batch = st.button("Run Batch ▶️")
        if run_batch:
            if uploaded is None:
                st.error("Please upload a CSV first.")
            elif not prompt_for_batch.strip():
                st.error("Please provide a prompt with {input}.")
            else:
                df = pd.read_csv(uploaded)
                if "input" not in df.columns:
                    st.error("CSV must have a column named 'input'.")
                else:
                    outputs: List[str] = []
                    progress = st.progress(0)
                    for i, text in enumerate(df["input"].astype(str).tolist()):
                        filled = prompt_for_batch.replace("{input}", text)
                        out = call_gemini(filled, temperature=temperature, top_p=top_p, system_instruction=system_instruction)
                        outputs.append(out)
                        progress.progress(int(((i + 1) / len(df)) * 100))
                        time.sleep(0.05)
                    df["output"] = outputs
                    st.success("Batch complete.", icon="✅")
                    st.dataframe(df.head(50), use_container_width=True)
                    st.download_button("Download Results CSV", df.to_csv(index=False), "batch_results.csv", "text/csv")

    # -------- Optimizer Tab --------
    with tab_opt:
        st.subheader("🚀 Prompt Optimizer")
        seed_prompt = st.text_area("Seed Prompt", height=140,
                                   placeholder="Write an onboarding email for new premium users…")
        rubric = st.text_area("Quality Rubric",
                              value="- Clear structure\n- Actionable steps\n- Friendly tone\n- Max 150 words", height=120)
        num_variants = st.slider("Variants", 1, 6, 3, 1)
        run_opt = st.button("Generate Variants & Scores")
        if run_opt:
            if not seed_prompt.strip():
                st.error("Enter a seed prompt to optimize.")
            else:
                variants: List[Dict[str, Any]] = []
                for _ in range(num_variants):
                    variant_req = f"""Improve the following prompt per the rubric.

Rubric:
{rubric}

Original Prompt:
\"\"\"
{seed_prompt}
\"\"\"

Return:
1) An improved prompt
2) A brief bullet list explaining changes
"""
                    improved = call_gemini(variant_req, temperature=temperature, top_p=top_p, system_instruction=system_instruction)

                    score_req = f"""Score this prompt from 0–10 for each rubric item and compute an average.

Rubric:
{rubric}

Prompt to Score:
\"\"\"
{improved}
\"\"\"

Return JSON with keys: scores (dict), average (number), rationale (string).
"""
                    scored = call_gemini(score_req, temperature=0.1, top_p=top_p, system_instruction="Return strict JSON only.")
                    try:
                        parsed = json.loads(scored)
                    except Exception:
                        parsed = {"scores": {}, "average": None, "rationale": "Could not parse JSON."}
                    variants.append({"prompt": improved, "eval": parsed})

                # Display table
                rows = []
                for i, v in enumerate(variants, start=1):
                    rows.append({
                        "Variant": i,
                        "Average": v["eval"].get("average"),
                        "Scores": json.dumps(v["eval"].get("scores", {})),
                        "Rationale": v["eval"].get("rationale", ""),
                    })
                st.dataframe(pd.DataFrame(rows))

                # Best pick
                best = max(variants, key=lambda x: x["eval"].get("average") or -1)
                st.markdown("#### 🏆 Best Variant")
                st.code(best["prompt"], language="markdown")

                st.download_button("Download Best Prompt.md", best["prompt"], "best_prompt.md", "text/markdown")

    # -------- Chat Sandbox Tab --------
    with tab_chat:
        st.subheader("💬 Chat with Gemini")
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []  # [{'role':'user'|'model', 'content': '...'}]

        user_msg = st.text_input("Your message")
        colA, colB = st.columns([1,1])
        with colA:
            send = st.button("Send ✅")
        with colB:
            clear = st.button("Clear Chat 🧹")

        if clear:
            st.session_state["chat_history"] = []
            st.experimental_rerun()

        if send and user_msg.strip():
            st.session_state["chat_history"].append({"role": "user", "content": user_msg})
            reply = chat_gemini(st.session_state["chat_history"], temperature=temperature, top_p=top_p)
            st.session_state["chat_history"].append({"role": "model", "content": reply})

        for m in st.session_state["chat_history"][-20:]:
            if m["role"] == "user":
                st.markdown(f"**You:** {m['content']}")
            else:
                st.markdown(f"**Gemini:** {m['content']}")

    # -------- Project Hook Tab --------
    with tab_project:
        st.subheader("🧪 Project Inference Hook")
        st.write("Wire this up to your ML project by implementing `run_project_inference()` in the code.")
        st.caption("Provide any JSON payload your project expects (dict).")
        payload_raw = st.text_area("Payload (JSON)", value='{"text": "demo input"}', height=140)
        run = st.button("Run Inference 🔍")
        if run:
            try:
                payload = json.loads(payload_raw)
            except json.JSONDecodeError as e:
                st.error(f"Invalid JSON: {e}")
            else:
                result = run_project_inference(payload)
                st.json(result)
