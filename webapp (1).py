# webapp.py
# GenAI Adoption & Workforce Impact Advisor (Streamlit)
# Uses Google Gemini 2.5 Flash Lite for prompt-engineered guidance.

import os
import json
from typing import Dict, Any, Optional

import streamlit as st
import pandas as pd
import numpy as np
import google.generativeai as genai

# ------------------ Configure Gemini ------------------
key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=key)
model = genai.GenerativeModel("gemini-2.5-flash-lite")

# ------------------ Page & Intro ------------------
st.set_page_config(page_title="GenAI Adoption & Workforce Impact Advisor", page_icon="🤝", layout="wide")
st.title(":orange[GenAI Adoption & Workforce Impact Advisor]")
st.caption("Strategize adoption, estimate ROI, and plan workforce upskilling with Gemini 2.5 Flash Lite.")

st.markdown('''
**What this app does**
- Lets you **upload or preview** your enterprise GenAI dataset (optional).
- Collects **context** about your organization.
- Uses **prompt engineering** to generate a practical **adoption roadmap**, **risk & governance plan**, **workforce impact**, and **KPIs**.
''')

if not key:
    st.warning("`GOOGLE_API_KEY` is not set. Add it in Streamlit Secrets or your environment to enable generation.", icon="⚠️")

# ------------------ Data Loading ------------------
with st.expander("📄 Data (optional)", expanded=False):
    uploaded = st.file_uploader("Upload your CSV (optional)", type=["csv"])
    df = None
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.success(f"Uploaded: {uploaded.name}", icon="✅")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    else:
        # Try local file path as a convenience for local runs
        default_path = "Enterprise_GenAI_Adoption_Impact.csv"
        if os.path.exists(default_path):
            try:
                df = pd.read_csv(default_path)
                st.info(f"Loaded local dataset: {default_path}", icon="ℹ️")
            except Exception as e:
                st.error(f"Error reading local CSV: {e}")
    if df is not None:
        st.dataframe(df.head(50), use_container_width=True)
        st.write(f"Rows: {len(df):,} • Columns: {len(df.columns)}")

# ------------------ Helper: dynamic controls ------------------
def build_controls_from_row(row: pd.Series) -> Dict[str, Any]:
    payload = {}
    for col, val in row.items():
        # Coerce numpy types to python types for JSON safety
        if pd.isna(val):
            continue
        if isinstance(val, (np.integer,)):
            payload[col] = int(val)
        elif isinstance(val, (np.floating,)):
            payload[col] = float(val)
        else:
            payload[col] = str(val)
    return payload

def dynamic_form_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    # Create inputs based on dataframe column types/uniques. Returns a dict payload.
    payload = {}
    st.write("Auto-built form from your dataset columns. (You can edit values before generating.)")
    for col in df.columns:
        series = df[col]
        # Skip fully empty
        if series.dropna().empty:
            continue
        # If low-cardinality categorical, use selectbox
        uniques = series.dropna().unique()
        if series.dtype == object or series.dtype.name == "category":
            if len(uniques) > 0 and len(uniques) <= 20:
                choice = st.selectbox(f"{col}", options=sorted(map(str, uniques.astype(str))), key=f"sb_{col}")
                payload[col] = choice
            else:
                txt = st.text_input(f"{col}", value=str(uniques[0]) if len(uniques) else "", key=f"ti_{col}")
                payload[col] = txt
        else:
            # Numeric column
            try:
                min_v = float(np.nanmin(series.astype(float)))
                max_v = float(np.nanmax(series.astype(float)))
                default = float(series.dropna().median())
                # Ensure slider works even for large ranges
                if np.isfinite(min_v) and np.isfinite(max_v) and min_v < max_v:
                    step = max((max_v - min_v) / 100.0, 0.01)
                    payload[col] = st.slider(f"{col}", min_value=min_v, max_value=max_v, value=default, step=step)
                else:
                    payload[col] = st.number_input(f"{col}", value=default)
            except Exception:
                payload[col] = st.text_input(f"{col}", value=str(series.dropna().iloc[0]) if not series.dropna().empty else "")
    return payload

# ------------------ Sidebar: Guided Org Context ------------------
st.sidebar.header("🔧 Organization Context")
org_name = st.sidebar.text_input("Organization / Team Name", value="Acme Corp")
industry = st.sidebar.selectbox("Industry", ["Technology","Finance","Retail","Healthcare","Manufacturing","Education","Gov/Public","Other"])
region = st.sidebar.selectbox("Region", ["North America","Europe","APAC","LATAM","MEA","Global"])
company_size = st.sidebar.selectbox("Company Size (employees)", ["1-50","51-200","201-1,000","1,001-5,000","5,001-10,000","10,000+"])
data_sensitivity = st.sidebar.selectbox("Primary Data Sensitivity", ["Low","Moderate","High","Regulated (PII/PHI/PCI)"])
risk_tolerance = st.sidebar.select_slider("Risk Tolerance", options=["Very Low","Low","Medium","High"])
budget_tier = st.sidebar.selectbox("GenAI Budget Tier (annual)", ["<$50k","$50k-$250k","$250k-$1M",">$1M"])
timeline = st.sidebar.selectbox("Adoption Timeline Target", ["0-3 months","3-6 months","6-12 months",">12 months"])
union_presence = st.sidebar.selectbox("Union Presence", ["None","Some units","Broad coverage"])
train_hours = st.sidebar.slider("Planned Training Hours per Employee", 0, 80, 16, step=2)
goal = st.sidebar.text_area("Primary Outcome Goal", value="Improve employee productivity and reduce time-to-complete routine tasks.")

# ------------------ Tabs ------------------
tab_form, tab_row, tab_prompt = st.tabs(["📝 Form-based Inputs", "🧷 Use a CSV Row", "🧠 View/Export Prompt"])

with tab_form:
    st.subheader("📝 Provide Key Inputs")
    st.write("Fill these fields to generate a tailored GenAI adoption & workforce plan.")

    # Representative feature inputs (edit freely)
    dept = st.selectbox("Primary Department Focus", ["Customer Support","Sales","Marketing","HR","Finance","Operations","Engineering","Legal","Cross-Functional"])
    adoption_stage = st.selectbox("Current Adoption Stage", ["Exploration","Pilot","Limited Rollout","Scaled","Mature"])
    employee_sentiment = st.select_slider("Employee Sentiment Toward GenAI", options=["Very Negative","Negative","Neutral","Positive","Very Positive"])
    tasks_targeted = st.text_area("Targeted Tasks / Use Cases", value="Email drafting; summarization of tickets; knowledge search; report generation.")
    constraints = st.text_area("Constraints / Compliance", value="- SOC2 required\n- PII must not leave VPC\n- Legal approval for external tools")
    kpis = st.text_area("Key KPIs to Track", value="- Average handle time\n- CSAT\n- Resolution rate\n- Cost per ticket")

    # Build payload
    form_payload: Dict[str, Any] = {
        "organization": org_name,
        "industry": industry,
        "region": region,
        "company_size": company_size,
        "data_sensitivity": data_sensitivity,
        "risk_tolerance": risk_tolerance,
        "budget_tier": budget_tier,
        "timeline": timeline,
        "union_presence": union_presence,
        "train_hours": train_hours,
        "goal": goal,
        "department": dept,
        "adoption_stage": adoption_stage,
        "employee_sentiment": employee_sentiment,
        "targeted_tasks": tasks_targeted,
        "constraints": constraints,
        "kpis": kpis,
    }

with tab_row:
    st.subheader("🧷 Build Inputs from a CSV Row (optional)")
    row_payload: Optional[Dict[str, Any]] = None
    if df is None:
        st.info("Upload a CSV above or place 'Enterprise_GenAI_Adoption_Impact.csv' next to this app to enable this feature.", icon="ℹ️")
    else:
        index_choice = st.number_input("Row index to use", min_value=0, max_value=len(df)-1, value=0, step=1)
        selected_row = df.iloc[int(index_choice)]
        st.dataframe(pd.DataFrame(selected_row).T, use_container_width=True)
        # Offer dynamic form built from dataset columns, so the user can tweak
        st.markdown("#### 🔧 Edit Auto-Built Inputs")
        row_payload = dynamic_form_from_df(pd.DataFrame([selected_row]))

with tab_prompt:
    st.subheader("🧠 Prompt Preview / Export")
    st.write("This is the prompt that will be sent to Gemini. You can copy or download it.")

# ------------------ Prompt Construction ------------------
def build_prompt(context_payload: Dict[str, Any], extra_payload: Optional[Dict[str, Any]] = None) -> str:
    combined = {**context_payload}
    if extra_payload:
        combined.update({k: v for k, v in extra_payload.items() if v is not None})
    # Pretty JSON for visibility
    payload_json = json.dumps(combined, indent=2)

    prompt = f'''
Assume you are a senior enterprise AI strategist and workforce transformation expert.
Using the **structured context** below, produce a practical, phased **GenAI adoption plan** with clear actions.

<CONTEXT-JSON>
{payload_json}
</CONTEXT-JSON>

Return the output in the following exact structure (use bullet points and concise language):
1) Greeting to the organization by name.
2) Executive Summary (3–5 bullets).
3) Key Use Cases (ranked, with quick value hypothesis).
4) Pilot Plan (30–60–90 day milestones, owners, success criteria).
5) Workforce Impact & Change Management
   - Roles affected (by skill level), tasks augmented, reskilling plan (timeline, training hours, modalities).
   - Communication plan (cadence, champions, FAQs).
6) Risk & Governance
   - Data security, privacy, compliance, model risk, quality review.
   - Guardrails and approval flow.
7) Architecture & Tooling
   - Build vs buy recommendations, integration points, minimal viable stack diagram (textual).
8) ROI & KPIs
   - Baseline metrics, target ranges, simple ROI estimate with assumptions.
9) Phased Rollout Roadmap (Gantt-like in text): phases, duration, dependencies.
10) Budget Bands & Resourcing (lean/standard/ambitious).
11) One-page Table: "Action | Owner | Start | End | KPI | Risk | Mitigation".
12) Final Positive Note + Summary.
'''
    return prompt

# Keep current working payload (form or row)
active_payload = form_payload.copy()
if 'row_payload' in locals() and row_payload:
    # Merge row-derived fields under a namespaced key to keep clarity
    active_payload["csv_sample_overrides"] = row_payload

final_prompt = build_prompt(active_payload)

# Show prompt in the tab
with tab_prompt:
    st.code(final_prompt, language="markdown")
    st.download_button("Download Prompt.md", data=final_prompt, file_name="genai_adoption_prompt.md", mime="text/markdown")

# ------------------ Generate ------------------
st.markdown("---")
st.subheader("🚀 Generate Strategy")
user_question = st.text_area("Optional: Add any question or extra instruction you'd like the plan to address")
col_gen1, col_gen2 = st.columns([1,1])

def call_gemini(prompt_text: str) -> str:
    try:
        response = model.generate_content(prompt_text)
        return getattr(response, "text", str(response))
    except Exception as e:
        return f"[Gemini error] {e}"

with col_gen1:
    if st.button("Generate Plan with Gemini 2.5 Flash Lite ✨"):
        plan_prompt = final_prompt
        if user_question.strip():
            plan_prompt += f"\n\nAdditional instruction from user:\n{user_question}\n"
        with st.spinner("Generating..."):
            output_text = call_gemini(plan_prompt)
        st.markdown("### 📄 Plan")
        st.write(output_text)

with col_gen2:
    st.markdown("**Tips**")
    st.write("- Adjust org context from the sidebar.\n- Use a CSV row to seed realistic values.\n- Edit the prompt in the 'View/Export Prompt' tab for fine control.")

st.markdown("---")
st.caption("Built with Streamlit + Google Gemini 2.5 Flash Lite.")
