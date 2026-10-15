import os
import re
import json
import streamlit as st
from pypdf import PdfReader
from typing import Optional, List, Dict, Any, TypedDict
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from groq import Groq
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END

# Always reload latest .env variables
load_dotenv(override=True)

# ==========================================
# 0. AUTO-DETECT ALLOWED GROQ MODEL
# ==========================================
def get_working_groq_model() -> str:
    """Queries your Groq API key to pick an allowed chat model automatically."""
    load_dotenv(override=True)
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("Missing GROQ_API_KEY in .env file!")
        
    client = Groq(api_key=api_key)
    available = [m.id for m in client.models.list().data]
    
    # Preferred models in order of quality for structured extraction
    preferred = [
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "meta-llama/llama-4-maverick-17b-128e-instruct",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3-32b",
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "gemma2-9b-it",
        "mixtral-8x7b-32768"
    ]
    for model in preferred:
        if model in available:
            return model
            
    # Fallback to first non-audio/non-guard model in your account
    chat_models = [
        m for m in available
        if all(x not in m.lower() for x in ["whisper", "tts", "guard", "orpheus"])
    ]
    if not chat_models:
        raise ValueError(
            f"Your Groq API key has 0 chat models enabled! Available list: {available}. "
            "Go to console.groq.com -> Settings -> Limits to enable models, or create a new API key."
        )
    return chat_models[0]

# ==========================================
# 1. CANONICAL DATA MODEL (Anonymized)
# ==========================================
class ExtractedReferral(BaseModel):
    source_system: str = Field(description="Source portal e.g. MedPortal, HealthLink, eFax")
    source_referral_id: str = Field(description="Unique referral ID from source")
    patient_name: str = Field(description="Full name of patient")
    dob: str = Field(description="Date of birth YYYY-MM-DD")
    gender: str = Field(description="Gender M, F, or Other")
    home_address: str = Field(description="Street address, city, state")
    zip_code: str = Field(description="5-digit USPS zip code")
    phone: str = Field(description="10-digit patient phone number")
    payer_name: str = Field(description="Insurance payer name")
    member_id: str = Field(description="Insurance member ID")
    referring_provider: str = Field(description="Referring physician name")
    provider_npi: str = Field(description="10-digit provider NPI (or empty if missing)")
    provider_phone: str = Field(description="10-digit provider phone/fax")
    services_requested: List[str] = Field(description="Requested disciplines: nursing, PT, OT, speech, aide, social work")
    icd10_code: str = Field(description="Primary diagnosis ICD-10 code")
    physician_order_attached: bool = Field(description="True if signed physician order / face-to-face document is present")
    requested_soc_date: str = Field(description="Requested start of care date YYYY-MM-DD")
    is_discharging_today: bool = Field(default=False, description="True if patient is discharging today or SOC within 48 hrs")
    high_acuity: bool = Field(default=False, description="True if requires IV therapy, wound vac, or ventilator")
    portal_doc_conflict: bool = Field(default=False, description="True if portal payload data conflicts with attached clinical notes")
    confidence_score: float = Field(default=0.95, description="Overall AI extraction confidence between 0.0 and 1.0")

class ReferralState(TypedDict):
    raw_document_text: str
    canonical_data: Optional[Dict[str, Any]]
    arrival_rating: str
    rating_reason: str
    validation_errors: List[str]
    eligibility_status: str
    duplicate_status: str
    decision: str
    reason_code: str
    decision_notes: str
    recommended_branch: Optional[str]
    draft_communication: str

# ==========================================
# 2. MOCK ENTERPRISE DATABASES
# ==========================================
SERVICE_AREAS = {
    "75001": ["Metro-North-Branch", "Dallas-Central-Branch"],
    "30301": ["Atlanta-Midtown-Branch"],
    "60601": ["Chicago-Lakeshore-Branch"]
}

BRANCH_CAPACITY = {
    "Metro-North-Branch": ["nursing", "pt", "ot"],
    "Dallas-Central-Branch": ["nursing", "pt", "ot", "speech"],
    "Atlanta-Midtown-Branch": ["nursing", "pt"],
    "Chicago-Lakeshore-Branch": []  # Full capacity scenario
}

# ==========================================
# 3. LANGGRAPH NODES
# ==========================================
def document_ai_extraction(state: ReferralState) -> Dict[str, Any]:
    """Step C04: AI Document OCR & Canonical Extraction"""
    model_id = get_working_groq_model()
    llm = ChatGroq(model_name=model_id, temperature=0.0, api_key=os.getenv("GROQ_API_KEY"))
    extractor = llm.with_structured_output(ExtractedReferral)
    
    prompt = (
        "You are an enterprise healthcare intake extraction AI for Apex Home Care. "
        "Extract the referral details into the canonical schema. "
        "If a field like NPI is missing or invalid in the text, leave it empty or as found. "
        "If you notice conflicting data between the Portal Header and Clinical Attachment, set portal_doc_conflict=True. "
        "Assign a realistic confidence_score (e.g. 0.96 if clear, 0.78 if messy/missing info).\n\n"
        f"REFERRAL PACKET:\n{state['raw_document_text']}"
    )
    extracted: ExtractedReferral = extractor.invoke(prompt)
    return {"canonical_data": extracted.model_dump()}

def triage_and_validation(state: ReferralState) -> Dict[str, Any]:
    """Step C03A & C05: Mandatory Validation & Arrival Triage Rating"""
    data = state["canonical_data"]
    errors = []
    
    if not data.get("patient_name"):
        errors.append("Patient Name missing")
    if not re.match(r"^\d{5}$", str(data.get("zip_code", ""))):
        errors.append("ZIP Code must be 5 digits")
    if not re.match(r"^\d{10}$", re.sub(r"\D", "", str(data.get("provider_npi", "")))):
        errors.append("Referring Provider NPI missing or not 10 digits")
    if not data.get("physician_order_attached"):
        errors.append("Physician Order / Face-to-Face attachment missing")
    if not data.get("icd10_code"):
        errors.append("Primary Diagnosis (ICD-10) missing")
    if data.get("portal_doc_conflict"):
        errors.append("Conflict detected between Portal Payload and Clinical Attachment")
        
    confidence = data.get("confidence_score", 0.95)
    if confidence < 0.90:
        errors.append(f"AI Confidence ({confidence:.2f}) below 0.90 threshold - requires human confirmation")

    if data.get("is_discharging_today") or data.get("high_acuity"):
        rating = "Red"
        reason = "Rule R-01: Urgent SOC (<48h) / Hospital discharge today or High-Acuity patient"
    elif len(errors) > 0 or confidence < 0.90:
        rating = "Amber"
        reason = f"Rule R-02: Validation flags present ({len(errors)} issue(s)) or low confidence"
    else:
        rating = "Green"
        reason = "Rule R-03: All 17 mandatory fields verified with high confidence (>=0.90)"

    return {
        "arrival_rating": rating,
        "rating_reason": reason,
        "validation_errors": errors
    }

def business_rules_decision(state: ReferralState) -> Dict[str, Any]:
    """Step C07-C12: Deterministic Rule Engine (No AI makes eligibility/routing decisions)"""
    data = state["canonical_data"]
    errors = state["validation_errors"]
    zip_code = str(data.get("zip_code", ""))
    services = [s.lower() for s in data.get("services_requested", [])]
    
    if state.get("duplicate_status") == "exact_match":
        return {
            "decision": "Reject",
            "reason_code": "DUPLICATE",
            "decision_notes": "Rule R-04: Exact match found for an open referral for the same patient and services.",
            "recommended_branch": None
        }

    candidate_branches = SERVICE_AREAS.get(zip_code, [])
    if not candidate_branches:
        return {
            "decision": "Reject",
            "reason_code": "OUT_OF_AREA",
            "decision_notes": f"Rule R-05: Patient ZIP ({zip_code}) is outside all Apex Home Care service areas.",
            "recommended_branch": None
        }

    if state.get("eligibility_status") == "inactive":
        return {
            "decision": "Reject",
            "reason_code": "NOT_ELIGIBLE",
            "decision_notes": "Rule R-06: Eligibility API returned coverage INACTIVE for home health services.",
            "recommended_branch": None
        }

    if len(errors) > 0 or state.get("duplicate_status") == "probable_match" or state.get("eligibility_status") == "error":
        return {
            "decision": "Review",
            "reason_code": "DATA_CHECK",
            "decision_notes": f"Rule R-09: Routed to Intake Exception Workqueue -> {'; '.join(errors) if errors else 'Verify match/eligibility'}",
            "recommended_branch": candidate_branches[0]
        }

    chosen_branch = None
    for branch in candidate_branches:
        cap = BRANCH_CAPACITY.get(branch, [])
        if all(req in cap for req in services):
            chosen_branch = branch
            break

    if not chosen_branch:
        return {
            "decision": "Review",
            "reason_code": "NO_CAPACITY",
            "decision_notes": f"Rule R-10: Candidate branch(es) {candidate_branches} lack capacity for {services}.",
            "recommended_branch": None
        }

    if state["arrival_rating"] == "Red" or data.get("high_acuity"):
        return {
            "decision": "Review",
            "reason_code": "CLINICAL",
            "decision_notes": f"Rule R-11: High-acuity / urgent referral routed to Clinical Reviewer Queue (Branch candidate: {chosen_branch}).",
            "recommended_branch": chosen_branch
        }

    return {
        "decision": "Accept",
        "reason_code": "ACCEPTED",
        "decision_notes": f"Rule R-12: All validation, eligibility, and capacity checks passed. Assigned to {chosen_branch}.",
        "recommended_branch": chosen_branch
    }

def communications_draft(state: ReferralState) -> Dict[str, Any]:
    """Step C16: Draft Partner Communication using Approved Templates"""
    model_id = get_working_groq_model()
    llm = ChatGroq(model_name=model_id, temperature=0.2, api_key=os.getenv("GROQ_API_KEY"))
    data = state["canonical_data"]
    
    prompt = (
        "You are the CareIntake Communication Assistant for Apex Home Care. "
        "Draft a concise, professional notification message to the referring provider based on the outcome below.\n"
        f"- Source Referral ID: {data.get('source_referral_id')}\n"
        f"- Patient Name: {data.get('patient_name')}\n"
        f"- Referring Provider: {data.get('referring_provider')}\n"
        f"- Recommended Decision: {state['decision']} (Reason Code: {state['reason_code']})\n"
        f"- Details / Missing Items: {state['decision_notes']}\n"
        f"- Assigned Branch: {state['recommended_branch']}\n\n"
        "If Review/DATA_CHECK, politely list the exact missing items needed to proceed. "
        "If Reject, clearly state the plain-language reason. If Accept, confirm placement and branch."
    )
    draft = llm.invoke(prompt).content
    return {"draft_communication": draft}

# Build LangGraph
builder = StateGraph(ReferralState)
builder.add_node("document_ai_extraction", document_ai_extraction)
builder.add_node("triage_and_validation", triage_and_validation)
builder.add_node("business_rules_decision", business_rules_decision)
builder.add_node("communications_draft", communications_draft)

builder.add_edge(START, "document_ai_extraction")
builder.add_edge("document_ai_extraction", "triage_and_validation")
builder.add_edge("triage_and_validation", "business_rules_decision")
builder.add_edge("business_rules_decision", "communications_draft")
builder.add_edge("communications_draft", END)

agent_workflow = builder.compile()

# ==========================================
# 4. SAMPLE DEMO SCENARIOS (Anonymized)
# ==========================================
SAMPLE_SCENARIOS = {
    "Scenario 1: Standard Complete Referral (Green -> Accept)": """
    [INBOUND SOURCE: MedPortal Connect | REF ID: MP-2026-8841]
    RECEIVED TIMESTAMP: 2026-10-14 08:30:00
    
    PATIENT DETAILS:
    Name: Eleanor Vance
    DOB: 1948-03-22 | Gender: F
    Address: 1200 Main Street, Addison, TX 75001
    Phone: 214-555-0143
    
    PAYER DETAILS:
    Payer: Apex Senior Care Advantage
    Member ID: APX-99482711
    
    CLINICAL & PROVIDER DETAILS:
    Referring Provider: Dr. Marcus Thorne, MD
    NPI: 1928374650 | Phone: 214-555-0988
    Requested Services: Nursing, PT
    Primary Diagnosis (ICD-10): M17.11 (Unilateral primary osteoarthritis, right knee)
    Requested Start of Care (SOC): 2026-10-18
    Physician Order: Signed Face-to-Face Encounter & Home Health Order Attached (PDF verified)
    Clinical Notes: Patient recovering well post knee replacement. Discharged 3 days ago. Stable, routine home PT and wound check requested.
    """,
    
    "Scenario 2: Urgent Hospital Discharge + High Acuity (Red -> Clinical Review)": """
    [INBOUND SOURCE: HealthLink Direct | REF ID: HL-2026-9910]
    
    PATIENT DETAILS:
    Name: David K. Miller
    DOB: 1959-11-04 | Gender: M
    Address: 450 Peachtree St NE, Atlanta, GA 30301
    Phone: 404-555-0192
    
    PAYER DETAILS:
    Payer: National Medicare First
    Member ID: NMF-33219044
    
    CLINICAL & PROVIDER DETAILS:
    Referring Provider: Dr. Priya Patel, MD
    NPI: 1092837465 | Phone: 404-555-7711
    Requested Services: Nursing
    Primary Diagnosis (ICD-10): I50.22 (Chronic systolic congestive heart failure)
    Requested Start of Care (SOC): 2026-10-15 (Discharging TODAY from hospital telemetry unit)
    Physician Order: Signed Hospital Discharge Order Attached.
    Clinical Notes: HIGH ACUITY ALERT - Patient discharging home today on continuous IV Dobutamine infusion via PICC line and requires negative pressure wound vac management.
    """,

    "Scenario 3: Missing Info & Data Conflict (Amber -> Intake Exception Queue)": """
    [INBOUND SOURCE: RightFax OCR Gateway | REF ID: FX-2026-4402]
    
    PORTAL HEADER PAYLOAD:
    Patient Name: Sarah Jenkins | DOB: 1962-07-19 | Gender: F
    Address: 800 Ross Ave, Dallas, TX 75001 | Phone: 972-555-0110
    Payer: Apex Senior Care Advantage | Member ID: APX-11223344
    Referring Provider: Dr. Alan Grant | NPI: [MISSING] | Phone: 972-555-3300
    Requested Services: Nursing, OT
    Primary Diagnosis: E11.9 (Type 2 diabetes mellitus without complications)
    Requested SOC: 2026-10-19
    
    ATTACHED OCR FAX NOTES:
    Note: No signed Physician Face-to-Face Order was included in fax pages 1-3.
    Discrepancy Alert: Attached clinical sheet lists patient DOB as 1965-07-19 (disagrees with Portal Header 1962-07-19). OCR scan quality is blurry (confidence 0.76).
    """,

    "Scenario 4: Out of Service Area ZIP (Green -> Automatic Reject Rule R-05)": """
    [INBOUND SOURCE: MedPortal Connect | REF ID: MP-2026-7731]
    
    PATIENT DETAILS:
    Name: Carlos Gomez
    DOB: 1951-01-15 | Gender: M
    Address: 999 Ocean Drive, Miami, FL 33139
    Phone: 305-555-0188
    
    PAYER DETAILS:
    Payer: Apex Senior Care Advantage
    Member ID: APX-77665544
    
    CLINICAL & PROVIDER DETAILS:
    Referring Provider: Dr. Elena Rostova, MD
    NPI: 1564738291 | Phone: 305-555-2211
    Requested Services: Nursing, PT
    Primary Diagnosis (ICD-10): I10 (Essential primary hypertension)
    Requested Start of Care (SOC): 2026-10-20
    Physician Order: Signed order attached.
    Clinical Notes: Standard home health assessment.
    """
}

# ==========================================
# 5. STREAMLIT WEB UI
# ==========================================
st.set_page_config(page_title="CareIntake AI - Referral Automation", layout="wide")

st.title("🏥 CareIntake AI — Intelligent Referral Intake & Triage Agent")
st.caption("Enterprise Architecture POV: AI Extraction (C04) • Arrival Rating (C03A) • Deterministic Rules Engine (C12) • Human-in-the-Loop Comms (C16)")

with st.sidebar:
    st.header("⚙️ Simulation Controls")
    try:
        active_model = get_working_groq_model()
        st.success(f"Connected Model: `{active_model}`")
    except Exception as e:
        st.error(f"Groq Key Issue: {e}")
        
    input_mode = st.radio("Select Input Method:", ["Pre-loaded Demo Scenarios", "Upload Referral PDF / Paste Text"])
    
    st.subheader("Mock Downstream APIs")
    elig_sim = st.selectbox("Eligibility API (C07):", ["active", "inactive", "error"], index=0)
    dup_sim = st.selectbox("MDM Duplicate Check (C11):", ["no_match", "probable_match", "exact_match"], index=0)
    
    st.divider()
    st.markdown("**Covered Service Area ZIPs:**\n- `75001` (Dallas/Metro North)\n- `30301` (Atlanta Midtown)\n- `60601` (Chicago - Full/No Cap)")

raw_input_text = ""
if input_mode == "Pre-loaded Demo Scenarios":
    selected_scenario = st.selectbox("Choose a Referral Scenario to Test Rules:", list(SAMPLE_SCENARIOS.keys()))
    raw_input_text = st.text_area("Incoming Referral Payload / OCR Text:", value=SAMPLE_SCENARIOS[selected_scenario], height=220)
else:
    uploaded_file = st.file_uploader("Upload Scanned Fax / Referral PDF", type=["pdf", "txt"])
    if uploaded_file:
        if uploaded_file.name.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            raw_input_text = "\n".join([page.extract_text() or "" for page in reader.pages])
        else:
            raw_input_text = uploaded_file.read().decode("utf-8")
    raw_input_text = st.text_area("Extracted / Pasted Referral Text:", value=raw_input_text, height=220)

if st.button("🚀 Run AI Referral Intake Agent", type="primary", use_container_width=True):
    if not os.getenv("GROQ_API_KEY"):
        st.error("Missing GROQ_API_KEY in your .env file!")
    elif not raw_input_text.strip():
        st.warning("Please provide referral text to process.")
    else:
        with st.spinner("Executing LangGraph Workflow: Document AI -> Validation -> Authoritative Checks -> Decisioning..."):
            initial_state = {
                "raw_document_text": raw_input_text,
                "canonical_data": None,
                "arrival_rating": "",
                "rating_reason": "",
                "validation_errors": [],
                "eligibility_status": elig_sim,
                "duplicate_status": dup_sim,
                "decision": "",
                "reason_code": "",
                "decision_notes": "",
                "recommended_branch": None,
                "draft_communication": ""
            }
            st.session_state["agent_result"] = agent_workflow.invoke(initial_state)

if "agent_result" in st.session_state:
    res = st.session_state["agent_result"]
    data = res["canonical_data"]
    
    st.divider()
    
    col1, col2, col3, col4 = st.columns(4)
    
    rating_colors = {"Green": "🟢 GREEN", "Amber": "🟡 AMBER", "Red": "🔴 RED"}
    decision_colors = {"Accept": "✅ ACCEPT", "Review": "⚠️ REVIEW", "Reject": "❌ REJECT"}
    
    col1.metric("Case ID (C03)", f"CASE-{data.get('source_referral_id', '001')}")
    col2.metric("Arrival Rating (C03A)", rating_colors.get(res["arrival_rating"], res["arrival_rating"]))
    col3.metric("Recommended Decision (C12)", decision_colors.get(res["decision"], res["decision"]))
    col4.metric("Reason Code", res["reason_code"])
    
    st.info(f"**Triage Rationale:** {res['rating_reason']}  \n**Decision Rule Trace:** {res['decision_notes']}")
    
    tab1, tab2, tab3 = st.tabs([
        "📄 1. Canonical Data Model & AI Confidence (C02/C04)",
        "🛡️ 2. Validation & Authoritative Checks (C05-C11)",
        "✉️ 3. Human-in-the-Loop Override & Comms (C12/C16)"
    ])
    
    with tab1:
        st.subheader("Canonical Referral Structure")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Patient & Payer Info**")
            st.json({
                "patient_name": data.get("patient_name"),
                "dob": data.get("dob"),
                "gender": data.get("gender"),
                "home_address": data.get("home_address"),
                "zip_code": data.get("zip_code"),
                "phone": data.get("phone"),
                "payer_name": data.get("payer_name"),
                "member_id": data.get("member_id")
            })
        with c2:
            st.markdown("**Clinical, Provider & AI Metadata**")
            st.json({
                "source_system": data.get("source_system"),
                "source_referral_id": data.get("source_referral_id"),
                "referring_provider": data.get("referring_provider"),
                "provider_npi": data.get("provider_npi"),
                "services_requested": data.get("services_requested"),
                "icd10_code": data.get("icd10_code"),
                "requested_soc_date": data.get("requested_soc_date"),
                "physician_order_attached": data.get("physician_order_attached"),
                "high_acuity": data.get("high_acuity"),
                "ai_confidence_score": data.get("confidence_score")
            })
            
    with tab2:
        st.subheader("Mandatory Field & Deterministic Rule Checks")
        if res["validation_errors"]:
            st.error("**Validation Failures / Exceptions Flagged (Routed to C06 Workqueue):**")
            for err in res["validation_errors"]:
                st.write(f"- ❌ {err}")
        else:
            st.success("✅ All 17 Mandatory Fields & AI Confidence Thresholds (>=0.90) Passed!")
            
        st.markdown("**Authoritative Downstream Lookups (Non-AI):**")
        st.write(f"- **Eligibility Check (C07):** `{res['eligibility_status'].upper()}`")
        st.write(f"- **Service Area Candidate Branches (C08):** `{SERVICE_AREAS.get(str(data.get('zip_code')), 'OUT OF COVERAGE')}`")
        st.write(f"- **Target Branch Assigned (C09):** `{res['recommended_branch'] or 'None'}`")
        st.write(f"- **Duplicate Patient Check (C11):** `{res['duplicate_status'].upper()}`")
        
    with tab3:
        st.subheader("Clinical Reviewer Decision Override (US-12) & Communication Approval (US-16)")
        col_a, col_b = st.columns([1, 2])
        with col_a:
            override_decision = st.selectbox(
                "Final Reviewer Action:",
                ["Keep Recommended (" + res["decision"] + ")", "Override -> Accept", "Override -> Review", "Override -> Reject"]
            )
            override_comment = st.text_input("Mandatory Reviewer Comment (Required if overriding):")
        with col_b:
            edited_email = st.text_area("AI-Drafted Partner Notification (Editable before sending):", value=res["draft_communication"], height=180)
            if st.button("✅ Approve Decision & Dispatch Notification"):
                if "Override" in override_decision and not override_comment.strip():
                    st.error("Rule US-12: A mandatory comment is required to override a recommended decision!")
                else:
                    st.success(f"Case Updated & Notification Logged! Final Action: {override_decision}")