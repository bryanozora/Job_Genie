import torch
import pandas as pd
import re
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_scheduler
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
import joblib
from tqdm import tqdm
import collections

# ---------- CLEANING ----------
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------- LOAD & PREPROCESS DATA ----------
df = pd.read_csv("/content/final_detailed_resume_dataset.csv").dropna()
df["Cleaned_Resume"] = df["Resume_str"].apply(clean_text)

# Balance categories
max_count = df['Category'].value_counts().max()
balanced = []
for cat in df['Category'].unique():
    group = df[df['Category'] == cat]
    balanced.append(resample(group, replace=True, n_samples=max_count, random_state=42))
df = pd.concat(balanced)

print("✅ Category distribution after balancing:")
print(df['Category'].value_counts())

# Label encoding
categories = sorted(df['Category'].unique())
label2id = {v: i for i, v in enumerate(categories)}
id2label = {i: v for v, i in label2id.items()}
df['label'] = df['Category'].map(label2id)

joblib.dump(label2id, "/content/label2id_indobert.pkl")
joblib.dump(id2label, "/content/id2label_indobert.pkl")

# ---------- TOKENIZATION ----------
model_name = "indobenchmark/indobert-base-p1"
tokenizer = AutoTokenizer.from_pretrained(model_name)

class ResumeDataset(Dataset):
    def __init__(self, texts, labels):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {
            'labels': torch.tensor(self.labels[idx])
        }

    def __len__(self):
        return len(self.labels)

# ---------- SPLIT ----------
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df["Cleaned_Resume"].tolist(),
    df["label"].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df["label"].tolist()
)

print("✅ Train label distribution:", collections.Counter(train_labels))
print("✅ Test label distribution:", collections.Counter(test_labels))

train_dataset = ResumeDataset(train_texts, train_labels)
test_dataset = ResumeDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8)

# ---------- MODEL SETUP ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(categories))
model.config.id2label = id2label
model.config.label2id = label2id
model.to(device)

# Freeze all layers except classifier and last encoder layer
for name, param in model.named_parameters():
    if not any(layer in name for layer in ["classifier", "encoder.layer.11", "pooler"]):
        param.requires_grad = False

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0,
                             num_training_steps=len(train_loader) * 10)

# ---------- DEBUG INPUT ----------
print("\n🔍 Sample tokenized input check:")
sample_text = df["Cleaned_Resume"].iloc[0]
tokens = tokenizer(sample_text, truncation=True, padding=True, max_length=256, return_tensors="pt")
print("Sample input length:", len(tokens['input_ids'][0]))
print("Sample tokens:", tokenizer.convert_ids_to_tokens(tokens['input_ids'][0][:20]))

# ---------- TRAIN WITH EARLY STOPPING ----------
epochs = 10
patience = 3
best_accuracy = 0
epochs_no_improve = 0

for epoch in range(epochs):
    model.train()
    print(f"\nEpoch {epoch+1}/{epochs}")
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        loop.set_description(f"Epoch {epoch+1}")
        loop.set_postfix(loss=loss.item())

    # ---------- VALIDATION ----------
    model.eval()
    val_preds, val_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            preds = torch.argmax(outputs.logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(batch['labels'].cpu().numpy())

    unique_labels = sorted(set(val_labels) | set(val_preds))
    target_names = [id2label[i] for i in unique_labels]
    accuracy = accuracy_score(val_labels, val_preds)

    print(f"\n[Validation Report - Epoch {epoch+1}]")
    print(classification_report(val_labels, val_preds, labels=unique_labels, target_names=target_names, zero_division=0))
    print("Accuracy:", accuracy)
    print("Predicted label counts:", collections.Counter(val_preds))

    # ---------- EARLY STOPPING ----------
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        epochs_no_improve = 0
        print("✅ New best accuracy. Saving model checkpoint.")
        model.save_pretrained("indobert_resume_model")
        tokenizer.save_pretrained("indobert_resume_model")
    else:
        epochs_no_improve += 1
        print(f"⚠️ No improvement for {epochs_no_improve} epoch(s).")

    if epochs_no_improve >= patience:
        print("\n⏹️ Early stopping triggered.")
        break

# ---------- FINAL ----------
print(f"\n✅ Best validation accuracy: {best_accuracy:.4f}")
print("✅ Final model saved to '/content/resume_model_xlm_roberta'")
