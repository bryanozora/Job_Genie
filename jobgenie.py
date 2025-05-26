# --- FILE: jobgenie.py (Multilingual Ready) ---
import streamlit as st
import pandas as pd
import fitz
import tempfile
import joblib
import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from datetime import datetime
import os
from langdetect import detect
from auth_utils import authenticate, save_user, authenticate_business, save_business_user

# ------------------ CONFIG ------------------
st.set_page_config(page_title="JobGenie Matcher", page_icon="💞", layout="wide")

APPLICATIONS_DB = "applications.csv"

# ------------------ APPLICATION SAVE ------------------
def save_application(user, job_title, company, row):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_app = pd.DataFrame([{
        "username": user,
        "job_title": job_title,
        "company": company,
        "match_score": round(row["Final Score"] * 100, 2),
        "category_match": round(row["Category Match"] * 100, 2),
        "skill_match": round(row["Skill Match"] * 100, 2),
        "semantic_similarity": round(row["Semantic Similarity"] * 100, 2),
        "skills": ", ".join(sorted(st.session_state.resume_skills)),
        "resume_text": st.session_state.resume_text,
        "applied_at": now
    }])
    try:
        existing = pd.read_csv(APPLICATIONS_DB)
        updated = pd.concat([existing, new_app], ignore_index=True)
    except FileNotFoundError:
        updated = new_app
    updated.to_csv(APPLICATIONS_DB, index=False)

# ------------------ BUSINESS DASHBOARD ------------------
def view_applications_dashboard():
    st.title("📊 Applications Dashboard")
    try:
        applications = pd.read_csv(APPLICATIONS_DB)
    except FileNotFoundError:
        st.warning("No applications have been submitted yet.")
        return
    company_name = st.session_state.business_name
    company_apps = applications[applications["company"] == company_name]
    if company_apps.empty:
        st.info("You haven't received any applications yet.")
        return
    st.markdown(f"### Applicants for jobs posted by `{company_name}`")
    for idx, row in company_apps.iterrows():
        st.markdown(f"#### 👤 {row['username']}")
        st.markdown(f"**Job Title:** {row['job_title']}")
        st.markdown(f"**Match Score:** {row['match_score']}%")
        st.markdown(f"🔹 Skill Match: `{row['skill_match']}%`")
        st.markdown(f"🔹 Category Match: `{row['category_match']}%`")
        st.markdown(f"🔹 Semantic Similarity: `{row['semantic_similarity']}%`")
        st.markdown(f"🧠 Extracted Skills: `{row['skills']}`")
        with st.expander("📄 View Extracted Resume"):
            st.text_area("Resume", row["resume_text"], height=300)
        st.markdown("---")

# ------------------ SESSION INIT ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""
if "business_logged_in" not in st.session_state:
    st.session_state.business_logged_in = False
    st.session_state.business_name = ""

# ------------------ LOGIN SCREENS ------------------
def login_screen():
    st.title("🔐 Job Seeker Login / Sign Up")
    option = st.radio("Choose an option", ["Login", "Sign Up"])
    if option == "Login":
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if authenticate(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Logged in successfully!")
                st.rerun()
            else:
                st.error("Invalid credentials.")
    else:
        username = st.text_input("Choose a username")
        email = st.text_input("Your email")
        password = st.text_input("Choose a password", type="password")
        if st.button("Sign Up"):
            success, msg = save_user(username, email, password)
            if success:
                st.success(msg)
                st.info("You can now login.")
            else:
                st.error(msg)

def business_login_screen():
    st.title("🏢 Business Login / Sign Up")
    option = st.radio("Choose an option", ["Login", "Sign Up"], key="biz_auth_option")
    if option == "Login":
        biz_name = st.text_input("Business Name")
        password = st.text_input("Password", type="password")
        if st.button("Login", key="biz_login_btn"):
            if authenticate_business(biz_name, password):
                st.session_state.business_logged_in = True
                st.session_state.business_name = biz_name
                st.success("Logged in as business!")
                st.rerun()
            else:
                st.error("Invalid business credentials.")
    else:
        biz_name = st.text_input("Choose a business name")
        email = st.text_input("Business email")
        password = st.text_input("Choose a password", type="password")
        if st.button("Sign Up", key="biz_signup_btn"):
            success, msg = save_business_user(biz_name, email, password)
            if success:
                st.success(msg)
                st.info("You can now login.")
            else:
                st.error(msg)

# ------------------ BUSINESS MODE ROUTER ------------------
mode = st.sidebar.selectbox("Select Mode", ["🎯 Job Seeker", "🏢 Business"])
if mode == "🏢 Business":
    if not st.session_state.business_logged_in:
        business_login_screen()
        st.stop()
    else:
        st.sidebar.success(f"Logged in as {st.session_state.business_name}")
        choice = st.sidebar.radio("Business Panel", ["📄 Submit Job", "📊 View Applications"])
        if choice == "📄 Submit Job":
            st.title("📄 Submit a Job Posting")
            job_title = st.text_input("Job Title")
            company = st.text_input("Company Name", value=st.session_state.business_name)
            location = st.text_input("Location")
            salary = st.text_input("Salary")
            job_link = st.text_input("Job Link (optional)")
            category_options = [
                "accountant", "advocate", "agriculture", "apparel", "arts",
                "automobile", "aviation", "banking", "bpo", "business development",
                "chef", "construction", "consultant", "designer", "digital marketing",
                "engineering", "finance", "fitness", "healthcare", "hr",
                "information technology", "public relations", "sales", "teacher"
            ]
            category = st.selectbox("Job Category", category_options)
            job_description = st.text_area("Job Description", height=300)
            if st.button("Submit Job"):
                if not job_title or not company or not location or not salary or not category or not job_description:
                    st.error("Please fill in all required fields.")
                    st.stop()
                job_entry = pd.DataFrame([{"Job Title": job_title, "Company": company, "Location": location, "Salary": salary, "Job Link": job_link, "Category": category, "Job Description": job_description}])
                try:
                    existing_jobs = pd.read_csv("jobs.csv")
                    updated_jobs = pd.concat([existing_jobs, job_entry], ignore_index=True)
                except FileNotFoundError:
                    updated_jobs = job_entry
                updated_jobs.to_csv("jobs.csv", index=False)
                st.success("✅ Job posted successfully!")
        else:
            view_applications_dashboard()
        st.stop()

if not st.session_state.logged_in:
    login_screen()
    st.stop()

# ------------------ LOAD MODELS & UTILITIES ------------------
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained("bert_resume_model")
    bert_model = BertForSequenceClassification.from_pretrained("bert_resume_model")
    bert_model.eval()
    label2id = joblib.load("label2id.pkl")
    id2label = {v: k for k, v in label2id.items()}
    nlp = spacy.load("en_core_web_sm")  # Add Indonesian patterns manually
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.from_disk("jz_skill_patterns.jsonl")
    embed_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return bert_model, tokenizer, nlp, embed_model, id2label

bert_model, tokenizer, ner_model, embed_model, id2label = load_models()

def clean_text(text):
    text = re.sub(r"http\\S+|RT|cc|#\\S+|@\\S+", " ", text)
    text = re.sub(r"[!\"#$%&()*+,-./:;<=>?@[\\]^_`{|}~]", " ", text)
    text = re.sub(r"[^\\x00-\\x7f]", " ", text)
    return re.sub(r"\\s+", " ", text).strip()

def predict_category(text):
    inputs = tokenizer(clean_text(text), return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return id2label[torch.argmax(outputs.logits, dim=1).item()]

def extract_skills(text, nlp_model):
    doc = nlp_model(text)
    return set(ent.text.strip().lower() for ent in doc.ents if ent.label_ == "SKILL")

def extract_resume_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
    with fitz.open(tmp_file.name) as doc:
        return ' '.join([page.get_text() for page in doc])

def match_jobs(resume_text, job_df, resume_category, resume_skills):
    resume_embedding = embed_model.encode([resume_text])[0]
    results = []
    for _, row in job_df.iterrows():
        job_text = row["Job Description"]
        job_embedding = embed_model.encode([job_text])[0]
        job_skills = extract_skills(job_text, ner_model)
        job_category = row.get("Category", "").strip().upper()
        semantic_score = cosine_similarity([resume_embedding], [job_embedding])[0][0]
        skill_score = len(resume_skills & job_skills) / len(job_skills) if job_skills else 0
        category_score = 1.0 if resume_category == job_category else 0.0
        final_score = (0.5 * category_score) + (0.3 * skill_score) + (0.2 * semantic_score)
        results.append({"Job Title": row["Job Title"], "Company": row["Company"], "Location": row["Location"], "Salary": row["Salary"], "Job Link": row["Job Link"], "Job Category": job_category, "Job Description": job_text, "Category Match": category_score, "Skill Match": skill_score, "Semantic Similarity": semantic_score, "Final Score": final_score})
    return pd.DataFrame(results).sort_values(by="Final Score", ascending=False).head(10)

# ------------------ JOB SEEKER INTERFACE ------------------
st.title("🧙‍♂️ JobGenie – Smart Resume Matcher")
uploaded_pdf = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])

if uploaded_pdf:
    if "uploaded_filename" not in st.session_state:
        st.session_state.uploaded_filename = uploaded_pdf.name
    elif uploaded_pdf.name != st.session_state.uploaded_filename:
        st.session_state.clear()
        st.session_state.uploaded_filename = uploaded_pdf.name
        st.rerun()

    if "resume_text" not in st.session_state:
        try:
            job_df = pd.read_csv("jobs.csv")
        except FileNotFoundError:
            st.error("❌ jobs.csv not found.")
            st.stop()
        if job_df.empty:
            st.error("❌ jobs.csv is empty.")
            st.stop()
        with st.spinner("🔍 Analyzing your resume..."):
            resume_text = extract_resume_text(uploaded_pdf)
            resume_lang = detect(resume_text)
            resume_category = predict_category(resume_text).upper()
            resume_skills = extract_skills(resume_text, ner_model)
            top_matches = match_jobs(resume_text, job_df, resume_category, resume_skills)
            st.session_state.resume_text = resume_text
            st.session_state.resume_category = resume_category
            st.session_state.resume_skills = resume_skills
            st.session_state.top_matches = top_matches
            st.session_state.resume_lang = resume_lang

if "top_matches" in st.session_state:
    st.success("✅ Resume analysis loaded.")
    st.markdown(f"**Detected Language:** `{st.session_state.resume_lang}`")
    st.markdown(f"**Predicted Resume Category:** `{st.session_state.resume_category}`")
    st.write(", ".join(sorted(st.session_state.resume_skills)) or "_None detected_")
    with st.expander("📄 View Extracted Resume Text"):
        st.text_area("Resume Content", st.session_state.resume_text, height=300)
    st.subheader("🎯 Top Job Matches")
    for idx, row in st.session_state.top_matches.iterrows():
        st.markdown(f"### {row['Job Title']} at {row['Company']}")
        st.progress(min(row["Final Score"], 1.0))
        st.markdown(f"**Compatibility:** {round(row['Final Score'] * 100)}%")
        st.markdown(f"📍 **Location:** {row['Location']}")
        st.markdown(f"💵 **Salary:** {row['Salary']}")
        st.markdown(f"🧠 **Skill Match:** {round(row['Skill Match']*100)}%")
        st.markdown(f"🗂️ **Category Match:** {'✅' if row['Category Match'] == 1 else '❌'}")
        st.markdown(f"🧠 **Semantic Similarity:** {round(row['Semantic Similarity']*100)}%")
        st.markdown(f"🏷️ **Job Category:** `{row['Job Category']}`")
        st.text_area(label=f"Job Description {idx}", value=row["Job Description"][:800] + "...", height=180, label_visibility="collapsed")
        col1, col2 = st.columns([1, 2])
        with col1:
            if st.button("Apply Now", key=f"apply_{idx}"):
                save_application(st.session_state.username, row["Job Title"], row["Company"], row)
                st.success("✅ Application submitted!")
        with col2:
            if row["Job Link"]:
                st.markdown(f"""
                    <a href="{row['Job Link']}" target="_blank">
                        <button style='background-color: #4CAF50; color: white; padding: 6px 16px; border: none; border-radius: 5px; cursor: pointer;'>Open Job Link</button>
                    </a>
                """, unsafe_allow_html=True)
        st.markdown("---")
