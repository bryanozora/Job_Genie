# precompute_existing_jobs.py
import pandas as pd, json, numpy as np
from jobgenie import embed_model, extract_skills, ner_model

df = pd.read_csv("jobs.csv")
df["Job Embedding"] = df["Job Description"].apply(lambda x: json.dumps(embed_model.encode([x])[0].tolist()))
df["Job Skills"] = df["Job Description"].apply(lambda x: json.dumps(list(extract_skills(x, ner_model))))
df.to_csv("jobs.csv", index=False)
