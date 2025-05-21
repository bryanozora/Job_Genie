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
import os

# ------------------ CONFIG ------------------
st.set_page_config(page_title="JobGenie Matcher", page_icon="🧞", layout="wide")

# ------------------ USER AUTH SYSTEM ------------------

USER_DB = "users.csv"

def load_users():
    if not os.path.exists(USER_DB):
        return pd.DataFrame(columns=["username", "email", "password"])
    return pd.read_csv(USER_DB)

def save_user(username, email, password):
    users = load_users()
    if username in users["username"].values:
        return False, "Username already exists."
    new_user = pd.DataFrame([[username, email, password]], columns=["username", "email", "password"])
    users = pd.concat([users, new_user], ignore_index=True)
    users.to_csv(USER_DB, index=False)
    return True, "Account created."

def authenticate(username, password):
    users = load_users()
    if ((users["username"] == username) & (users["password"] == password)).any():
        return True
    return False

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.username = ""

def login_screen():
    st.title("🔐 JobGenie Login")
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

if not st.session_state.logged_in:
    login_screen()
    st.stop()

# ------------------ LOAD MODELS ------------------
@st.cache_resource
def load_models():
    tokenizer = BertTokenizer.from_pretrained("bert_resume_model")
    bert_model = BertForSequenceClassification.from_pretrained("bert_resume_model")
    bert_model.eval()

    label2id = joblib.load("label2id.pkl")
    id2label = {v: k for k, v in label2id.items()}

    nlp = spacy.load("en_core_web_sm")
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    ruler.from_disk("jz_skill_patterns.jsonl")
    ner_model = nlp

    embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return bert_model, tokenizer, ner_model, embed_model, id2label

bert_model, tokenizer, ner_model, embed_model, id2label = load_models()

# ------------------ CLEANING ------------------
def clean_text(text):
    text = re.sub(r'http\S+\s', ' ', text)
    text = re.sub(r'RT|cc', ' ', text)
    text = re.sub(r'#\S+\s', ' ', text)
    text = re.sub(r'@\S+', ' ', text)
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~]', ' ', text)
    text = re.sub(r'[^\x00-\x7f]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# ------------------ CATEGORY PREDICTOR ------------------
def predict_category(text):
    cleaned = clean_text(text)
    inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    predicted_class = torch.argmax(outputs.logits, dim=1).item()
    return id2label[predicted_class]

# ------------------ SKILL EXTRACTOR ------------------
def extract_skills(text, nlp_model):
    doc = nlp_model(text)
    return set(ent.text.strip().lower() for ent in doc.ents if ent.label_ == "SKILL")

# ------------------ RESUME PARSER ------------------
def extract_resume_text(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_file_path = tmp_file.name
    text = ""
    with fitz.open(tmp_file_path) as doc:
        for page in doc:
            text += page.get_text()
    return ' '.join(text.split())

# ------------------ MATCHING FUNCTION ------------------
def match_jobs(resume_text, job_df, resume_category, resume_skills):
    resume_embedding = embed_model.encode([resume_text])[0]

    results = []

    for _, row in job_df.iterrows():
        job_text = row["Job Description"]
        job_embedding = embed_model.encode([job_text])[0]
        job_skills = extract_skills(job_text, ner_model)
        job_category = row.get("Category", "").strip().upper()

        # 1. Semantic similarity
        semantic_score = cosine_similarity([resume_embedding], [job_embedding])[0][0]

        # 2. Skill similarity
        if job_skills:
            skill_score = len(resume_skills & job_skills) / len(job_skills)
        else:
            skill_score = 0

        # 3. Category match
        category_score = 1.0 if resume_category == job_category else 0.0

        # Final score
        final_score = (0.5 * category_score) + (0.3 * skill_score) + (0.2 * semantic_score)

        results.append({
            "Job Title": row["Job Title"],
            "Company": row["Company"],
            "Location": row.get("Location", ""),
            "Salary": row.get("Salary", ""),
            "Job Link": row.get("Job Link", ""),
            "Job Category": job_category,
            "Job Description": job_text,
            "Category Match": category_score,
            "Skill Match": skill_score,
            "Semantic Similarity": semantic_score,
            "Final Score": final_score
        })

    return pd.DataFrame(results).sort_values(by="Final Score", ascending=False).head(10)

# ------------------ MAIN UI ------------------
st.title("🧞‍♂️ JobGenie – Smart Resume Matcher")

uploaded_pdf = st.file_uploader("📄 Upload your resume (PDF)", type=["pdf"])

if uploaded_pdf:
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
        resume_category = predict_category(resume_text).upper()
        resume_skills = extract_skills(resume_text, ner_model)
        top_matches = match_jobs(resume_text, job_df, resume_category, resume_skills)

    st.success("✅ Done analyzing your resume.")

    # ---------- Resume Summary ----------
    st.subheader("📋 Resume Analysis Summary")
    st.markdown(f"**Predicted Resume Category:** `{resume_category}`")
    if resume_skills:
        st.markdown("**Extracted Skills:**")
        st.write(", ".join(sorted(resume_skills)))
    else:
        st.markdown("**Extracted Skills:** _None detected_")

    with st.expander("📄 View Extracted Resume Text"):
        st.text_area("Resume Content", resume_text, height=300)

    # ---------- Job Matches ----------
    st.subheader("🎯 Top Job Matches")

    for idx, row in top_matches.iterrows():
        st.markdown(f"### {row['Job Title']} at {row['Company']}")
        st.progress(min(row["Final Score"], 1.0))
        st.markdown(f"**Compatibility:** {round(row['Final Score'] * 100)}%")
        st.markdown(f"📍 **Location:** {row['Location']}")
        st.markdown(f"💵 **Salary:** {row['Salary']}")
        st.markdown(f"🧠 **Skill Match:** {round(row['Skill Match']*100)}%")
        st.markdown(f"🗂️ **Category Match:** {'✅' if row['Category Match'] == 1 else '❌'}")
        st.markdown(f"🧠 **Semantic Similarity:** {round(row['Semantic Similarity']*100)}%")
        st.markdown(f"🏷️ **Job Category:** `{row['Job Category']}`")
        st.markdown("📝 **Job Description Preview:**")
        st.text_area(label=f"Job Description {idx}", value=row["Job Description"][:800] + "...", height=180, label_visibility="collapsed")

        if row["Job Link"]:
            st.markdown(f"""
                <a href="{row['Job Link']}" target="_blank">
                    <button style='background-color: #4CAF50; color: white; padding: 8px 20px; border: none; border-radius: 5px; cursor: pointer; font-size: 16px;'>Apply Now</button>
                </a>
            """, unsafe_allow_html=True)

        st.markdown("---")
