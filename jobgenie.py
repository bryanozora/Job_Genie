# --- FILE: jobgenie.py ---
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