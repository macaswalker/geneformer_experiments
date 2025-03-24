import scanpy as sc
import numpy as np
import pandas as pd
import torch
import pickle
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel

# === 1. Load loom data ===
adata = sc.read_loom("RCC_data.loom")

# === 2. Filter for Tumour-normal and Metastasis ===
adata = adata[adata.obs["summaryDescription"].isin(["Tumour-normal", "Metastasis"])].copy()

# === 3. Assign classification labels ===
label_map = {"Tumour-normal": 0, "Metastasis": 1}
adata.obs["label"] = adata.obs["summaryDescription"].map(label_map)
labels = adata.obs["label"].astype(int).tolist()

# === 4. Preprocess gene IDs ===
# Strip version numbers like ".1", ".2" from Ensembl IDs
adata.var["ensembl_id"] = adata.var["ensembl_id"].astype(str).str.strip().str.replace(r"\..*$", "", regex=True)
adata = adata[:, adata.var["ensembl_id"].notna()].copy()

# Set Ensembl IDs as var_names (this is what Geneformer will use as input tokens)
adata.var_names = adata.var["ensembl_id"]

# === 5. Add required 'filter_pass' column ===
adata.var["filter_pass"] = True  # All genes are considered valid

# === 6. Split into train/test ===
train_idx, test_idx = train_test_split(
    np.arange(adata.n_obs),
    test_size=0.2,
    stratify=labels,
    random_state=42
)
adata_train = adata[train_idx].copy()
adata_test = adata[test_idx].copy()
labels_train = adata_train.obs["label"].tolist()
labels_test = adata_test.obs["label"].tolist()

# === 7. Set up Geneformer ===
device = "cuda" if torch.cuda.is_available() else "cpu"
config = GeneformerConfig(device=device, batch_size=10, model_name="gf-6L-30M-i2048")
model = GeneformerFineTuningModel(config, fine_tuning_head="classification", output_size=2)

# === 8. Process data using Ensembl IDs directly ===
train_ds = model.process_data(adata_train, gene_names="ensembl_id").add_column("label", labels_train)
test_ds = model.process_data(adata_test, gene_names="ensembl_id").add_column("label", labels_test)

# === 9. Train ===
model.train(
    train_dataset=train_ds.shuffle(),
    validation_dataset=test_ds,
    label="label",
    freeze_layers=0,
    epochs=2,
    optimizer_params={"lr": 1e-4},
    lr_scheduler_params={"name": "linear", "num_warmup_steps": 0, "num_training_steps": 2}
)

# === 9 and 3/4s: Save the fine-tuned model weights and configuration manually ===
os.makedirs("fine_tuned_model", exist_ok=True)
torch.save(model.state_dict(), "fine_tuned_model/pytorch_model.bin")
with open("fine_tuned_model/config.json", "w") as f:
    json.dump(model.config, f)

print("Model fine-tuned and saved.")

# === 10. Evaluate ===
outputs = model.get_outputs(test_ds)
preds = outputs.argmax(axis=1)
print("\nClassification Report:")
print(classification_report(labels_test, preds, target_names=["Tumour-normal", "Metastasis"]))

# === 11. Confusion Matrix ===
cm = confusion_matrix(labels_test, preds)
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(6, 6))
ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Tumour-normal", "Metastasis"]).plot(ax=ax)
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
