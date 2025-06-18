# 💬 JobGenie – Smart Resume & Job Matcher

**JobGenie** is an AI-powered web application that helps job seekers find suitable job opportunities based on their resumes. This project was developed as a university assignment for the **Natural Language Processing** and **Applied AI** courses at **Petra Christian University**, applying modern NLP techniques and BERT-based deep learning models.

---

## 📚 Academic Context

This project was created as a college assignment for the **Natural Language Processing** and **Applied AI** courses at **Petra Christian University**. It showcases how NLP and transformer-based models can be used to solve real-world problems in job matching.

---

## 📌 Features

- 🔐 **User Authentication**: Login/signup system for job seekers & businesses using CSV-based storage.
- 📄 **Resume Upload**: Accepts PDF files and extracts text using PyMuPDF.
- 🧠 **Resume Classification**: Classifies resume into categories using BERT/IndoBERT models (English & Indonesian).
- 🛠️ **Skill Extraction**: Extracts domain-specific skills using spaCy NER with custom pattern rules.
- 🤝 **Job Matching**:
  - Category match (50%)
  - Skill overlap (30%)
  - Semantic similarity using Sentence Transformers (20%)
- 📊 **Top 10 Recommendations**: Displays best-matched job cards with compatibility score, skills, category alignment, and job links.
- 📂 **Business Dashboard**: Businesses can view applications submitted to their job postings.

---

## 🛠️ Technologies Used

- **Language**: Python  
- **Framework**: Streamlit  
- **Libraries**:
  - `transformers`, `torch` – BERT and IndoBERT resume classification
  - `sentence-transformers` – Semantic similarity matching
  - `spaCy` – Skill extraction using NER patterns
  - `pandas`, `joblib`, `scikit-learn`
  - `PyMuPDF (fitz)` – PDF parsing
  - `langdetect` – Resume language detection
  - `beautifulsoup4`, `requests`, `tqdm` – Job scraping from jSearch API

---

## 📁 Files Overview

| File | Description |
|------|-------------|
| `jobgenie.py` | Main Streamlit app: UI, resume processing, job matching |
| `resume_classifier_eng.py` | Train and evaluate BERT model for English resume classification |
| `resume_classifier_ind.py` | Train and evaluate IndoBERT for Indonesian resume classification |
| `auth_utils.py` | Handles user/business authentication |
| `jsearch_job.py` | Job scraper using jSearch (RapidAPI) |
| `migrate_old_data.py` | Precomputes job embeddings & skills for matching |
| `jobs.csv` | Job postings dataset |
| `jz_skill_patterns.jsonl` | spaCy patterns for skill recognition |
| `users.csv`, `business_users.csv` | CSV database for user credentials |
| `label2id_eng.pkl`, `label2id_indobert.pkl` | Mapping dictionaries for category labels |

---

## 🚀 How to Run

### 1. Install Dependencies

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If you don't have a `requirements.txt`, use this:

```txt
streamlit
pandas
PyMuPDF==1.23.9
joblib
spacy
sentence-transformers
scikit-learn
transformers
torch
langdetect
beautifulsoup4
requests
tqdm
```

---

### 2. Job Scraping (optional)

Edit `jsearch_job.py` and insert your API key:

```python
API_KEY = "your_rapidapi_key_here"
```

Run:

```bash
python jsearch_job.py
```


---

### 3. Train Resume Classifier (if not already trained)

```bash
python resume_classifier_eng.py      # For English BERT model
python resume_classifier_ind.py      # For IndoBERT model
```

This will save:
- `bert_resume_model_eng/`
- `indobert_resume_model/`
- `label2id_eng.pkl`, `label2id_indobert.pkl`

---

### 3. Precompute Job Embeddings (optional)

If you are not using the jobs.csv i provided, but runs the jsearch_job.py, then, run:

```bash
python migrate_old_data.py
```

this will generate and save job description embeddings into jobs.csv

---

### 4. Launch the Web App

```bash
streamlit run jobgenie.py
```

Visit: `http://localhost:8501`

---

## 🧪 Testing & Demo

Upload sample resumes like `designer.pdf`, `digital media.pdf`, etc.  
The system will:
- Detect language (English / Indonesian)
- Predict resume category
- Extract relevant skills
- Display top 10 matched jobs

---

## 📄 Dataset Sources

- Resume Dataset: https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset  
