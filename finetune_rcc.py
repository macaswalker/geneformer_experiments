# finetune_rcc.py

import scanpy as sc
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel

# === Setup ===
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# === Load data ===
adata = sc.read_loom("RCC_data.loom")
adata = adata[adata.obs["summaryDescription"].isin(["Tumour-normal", "Metastasis"])].copy()

# === Encode labels ===
label_map = {"Tumour-normal": 0, "Metastasis": 1}
labels = adata.obs["summaryDescription"].map(label_map).astype(int).tolist()

# === Train-test split ===
train_idx, test_idx = train_test_split(
    np.arange(adata.n_obs),
    test_size=0.2,
    stratify=labels,
    random_state=42
)
adata_train = adata[train_idx].copy()
adata_test = adata[test_idx].copy()
labels_train = [label_map[s] for s in adata_train.obs["summaryDescription"]]
labels_test = [label_map[s] for s in adata_test.obs["summaryDescription"]]

# === Configure and process ===
geneformer_config = GeneformerConfig(
    device=device,
    batch_size=10,
    model_name="gf-6L-30M-i2048"
)

model = GeneformerFineTuningModel(
    geneformer_config=geneformer_config,
    fine_tuning_head="classification",
    output_size=2
)

train_ds = model.process_data(adata_train).add_column("progression", labels_train)
test_ds = model.process_data(adata_test).add_column("progression", labels_test)

# === Train ===
model.train(
    train_dataset=train_ds.shuffle(),
    validation_dataset=test_ds,
    label="progression",
    freeze_layers=0,
    epochs=2,
    optimizer_params={"lr": 1e-4},
    lr_scheduler_params={"name": "linear", "num_warmup_steps": 0, "num_training_steps": 2}
)

# === Evaluate ===
outputs = model.get_outputs(test_ds)
print("\nClassification Report:")
print(classification_report(labels_test, outputs.argmax(axis=1), target_names=["Tumour-normal", "Metastasis"]))

# === Plot confusion matrix ===
cm = confusion_matrix(labels_test, outputs.argmax(axis=1))
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=["Tumour-normal", "Metastasis"])
disp.plot(ax=ax, cmap='coolwarm', values_format=".2f")
plt.title("Normalized Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
