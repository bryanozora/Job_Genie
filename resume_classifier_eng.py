import torch
import pandas as pd
import re
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import joblib
from tqdm import tqdm

# ---------- CLEANING ----------
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- LOAD & PREPROCESS DATA ----------
df = pd.read_csv("Resume/Resume.csv").dropna()
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)

# Balance categories
max_count = df['Category'].value_counts().max()
balanced = []
for cat in df['Category'].unique():
    group = df[df['Category'] == cat]
    balanced.append(resample(group, replace=True, n_samples=max_count, random_state=42))
df = pd.concat(balanced)

# Label encoding
categories = sorted(df['Category'].unique())
label2id = {v: i for i, v in enumerate(categories)}
id2label = {i: v for v, i in label2id.items()}
df['label'] = df['Category'].map(label2id)

# Save label2id map
joblib.dump(label2id, "label2id_eng.pkl")

# ---------- TOKENIZATION ----------
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {'labels': torch.tensor(self.labels[idx])}

    def __len__(self):
        return len(self.labels)

train_texts, test_texts, train_labels, test_labels = train_test_split(df["Cleaned_Resume"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42)

train_dataset = ResumeDataset(train_texts, train_labels)
test_dataset = ResumeDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ---------- MODEL SETUP ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(categories))
model.to(device)

optimizer = AdamW(model.parameters(), lr=5e-5)

# ---------- TRAIN ----------
epochs = 20
model.train()
for epoch in range(epochs):
    print(f"Epoch {epoch+1}/{epochs}")
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

# ---------- EVALUATE ----------
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch['labels'].cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=categories))
print("Accuracy:", accuracy_score(all_labels, all_preds))

# ---------- SAVE ----------
model.save_pretrained("bert_resume_model_eng")
tokenizer.save_pretrained("bert_resume_model_eng")