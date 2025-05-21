# 📚 Academic Context  
This project was created as a college assignment for the **Natural Language Processing** and **Applied AI** course at **Petra Christian University**. It applies various NLP and deep learning algorithms to intelligently match CVs/Resumes with relevant job descriptions.

---

# 💬 JobGenie – Smart Resume & Job Matcher  

**JobGenie** is an AI-powered web application that helps job seekers find suitable job opportunities based on the content of their resumes. It performs **resume classification**, **skill extraction**, and **semantic similarity matching** to recommend the top 10 most relevant job postings. This project was developed as a college assignment for the **Natural Language Processing** and **Applied AI** course at **Petra Christian University**, showcasing practical applications of NLP and BERT-based deep learning models.

---

## 📌 Features

- 🔐 **User Authentication**: Login and sign-up system with CSV-based storage.
- 📄 **Resume Upload**: Accepts resumes in PDF format and extracts textual content.
- 🧠 **Resume Classification**: Categorizes resumes using a fine-tuned BERT model (trained on 20+ resume categories).
- 🛠️ **Skill Extraction**: Uses spaCy NER with custom patterns to extract domain-relevant skills.
- 🤝 **Job Matching**: Compares resume content with job descriptions using Sentence Transformers for semantic similarity, skill overlap, and category match.
- 📊 **Top 10 Job Recommendations**: Displays job cards with relevance score, skill match %, category match, and semantic similarity.
- 🧾 **Detailed Analysis**: Shows resume summary, extracted skills, predicted category, and full extracted resume content.

---

## 🛠️ Technologies Used

- **Language**: Python  
- **Framework**: Streamlit  
- **Libraries**:
  - `transformers` (BERT for classification)
  - `sentence-transformers` (semantic similarity)
  - `spaCy` (custom skill extraction using NER patterns)
  - `pandas`, `scikit-learn`, `joblib`
  - `PyMuPDF` (`fitz`) for PDF parsing
  - `torch` for deep learning

---

## 📁 Files Overview

- `jobgenie.py` – Main Streamlit app with UI, resume analysis, matching logic, and user system.
- `resume_classifier.py` – Script to train and evaluate a BERT model for resume category classification.
- `bert_resume_model/` – Saved transformer model used in prediction (still need to run the resume_classifier.py since i cannot upload the model file because it is too large).
- `data\` - collection of resumes taken from resume dataset on kaggle (https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- `Resume\Resume.csv` - A csv file containing every resume content taken from resume dataset on kaggle (https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset)
- `label2id.pkl` – Mapping dictionary from category names to label IDs.
- `auth_utils.py` – Authentication utility module for handling login/signup and credential management.
- `designer.pdf`, `digital media.pdf`, `information tehcnology.pdf` - Resume used to test the application
- `jobs.csv` – Job database with fields like title, company, description, category, salary, and link.
- `jsearch_job.py` - a code used to retreive job list from jsearch using rapidAPI
- `users.csv` - saved user accounts
- `jz_skill_patterns.jsonl` – spaCy EntityRuler pattern file for skill recognition.

---

## 🚀 How It Works

1. **User Login**: Users log in or sign up through a basic credential system.
2. **Resume Upload**: Users upload their resume in PDF format.
3. **Resume Processing**:
   - Extract text using PyMuPDF
   - Clean and tokenize content
   - Run BERT classifier to predict resume category
   - Extract skills using spaCy NER with custom patterns
4. **Job Matching**:
   - Each job description is vectorized using `all-MiniLM-L6-v2`
   - Match is based on:
     - Category match (50% weight)
     - Skill overlap (30% weight)
     - Semantic similarity (20% weight)
   - Top 10 jobs are sorted and displayed
5. **Results Displayed**: Users see job titles, compatibility scores, skill overlap, category alignment, semantic similarity, and application links.

---

## ✅ Requirements

Install required packages via pip:

`pip install streamlit pandas torch transformers sentence-transformers spacy joblib pymupdf` 
`python -m spacy download en_core_web_sm`

run resume_classifier.py to train the bert model

run the jobgenie.py by using streamlit (streamlit run jobgenie.py)
