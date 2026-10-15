# 🏥 CareIntake AI — Intelligent E-Referral Intake & Triage Agent

An enterprise-grade AI Agent prototype built with **LangGraph**, **Groq LLM**, **Pydantic**, and **Streamlit** to automate healthcare home-care referral intake, canonical data extraction, priority triage, deterministic business rule validation, and human-in-the-loop communications.

## 🏗️ Architecture & Workflow
1. **Document AI & Canonical Extraction (C02 / C04):** Extracts unstructured referral payloads and PDF faxes into a standardized 17-field Canonical Data Model with AI confidence scoring.
2. **Arrival Priority Triage & Validation (C03A / C05):** Evaluates mandatory fields, regex formats (NPI, ZIP, ICD-10), confidence thresholds (`>= 0.90`), and assigns **Green / Amber / Red** priority ratings (Rules R-01 to R-03).
3. **Deterministic Decision Engine (C07 – C12):** Zero-hallucination rule engine validating service area coverage, payer eligibility, duplicate records, branch capacity, and clinical acuity (Rules R-04 to R-12).
4. **Human-in-the-Loop Override & Communications (C12 / C16):** Drafts template-aligned provider notifications and enforces mandatory audit comments on clinical overrides.

## 🚀 Quickstart

```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure environment variables (.env)
# GROQ_API_KEY=your_groq_api_key

# 4. Run the application
streamlit run app.py