# mech_interp_geneformer.py

import torch
import json
import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap

from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
from sklearn.metrics import classification_report
from copy import deepcopy

# === Step 1: Load the fine-tuned model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("fine_tuned_model/config.json") as f:
    config_dict = json.load(f)

config = GeneformerConfig(**config_dict, device=device)
model = GeneformerFineTuningModel(config, fine_tuning_head="classification", output_size=2)
model.load_state_dict(torch.load("fine_tuned_model/pytorch_model.bin", map_location=device))
model.to(device).eval()

print("✅ Model loaded successfully.")

# === Step 2: Load and preprocess RCC dataset ===
adata = sc.read_loom("RCC_data.loom")
adata = adata[adata.obs["summaryDescription"].isin(["Tumour-normal", "Metastasis"])].copy()
label_map = {"Tumour-normal": 0, "Metastasis": 1}
adata.obs["label"] = adata.obs["summaryDescription"].map(label_map)

adata.var["ensembl_id"] = adata.var["ensembl_id"].astype(str).str.strip().str.replace(r"\..*$", "", regex=True)
adata.var_names = adata.var["ensembl_id"]
adata.var["filter_pass"] = True

labels = adata.obs["label"].tolist()
print("✅ Data loaded and processed.")

# === Step 3: Convert data for Geneformer ===
dataset = model.process_data(adata, gene_names="ensembl_id").add_column("label", labels)
print("✅ Data tokenized successfully.")

# === Step 4: Attention-based gene importance ===
print("⏳ Extracting attention weights...")
attention_weights = model.get_attention_weights(dataset)
avg_attention = attention_weights.mean(axis=(0, 1, 2))  # average over all cells and heads

# Get gene IDs and sort by attention
gene_ids = adata.var_names.tolist()
attention_df = pd.DataFrame({
    "ensembl_id": gene_ids,
    "attention": avg_attention
}).sort_values(by="attention", ascending=False)

top_genes_attention = attention_df.head(20)
print("✅ Top 20 genes based on attention:")
print(top_genes_attention)

# === Step 5: Perturbation-based gene importance ===
def perturb_and_measure_importance(model, dataset, gene_idx):
    perturbed_dataset = deepcopy(dataset)
    for i in range(len(perturbed_dataset)):
        if gene_idx in perturbed_dataset[i]["input_ids"]:
            perturbed_dataset[i]["input_ids"].remove(gene_idx)
    
    original_outputs = model.get_outputs(dataset).softmax(axis=1)
    perturbed_outputs = model.get_outputs(perturbed_dataset).softmax(axis=1)
    diff = np.abs(original_outputs - perturbed_outputs).mean()
    return diff

print("⏳ Computing perturbation-based importance...")
top_attention_gene_indices = [gene_ids.index(g) for g in top_genes_attention["ensembl_id"]]
perturbation_scores = []

for idx in top_attention_gene_indices:
    score = perturb_and_measure_importance(model, dataset, idx)
    perturbation_scores.append(score)
    print(f"Gene {gene_ids[idx]}: importance score = {score}")

perturbation_df = pd.DataFrame({
    "ensembl_id": [gene_ids[idx] for idx in top_attention_gene_indices],
    "perturbation_importance": perturbation_scores
}).sort_values("perturbation_importance", ascending=False)

print("✅ Perturbation-based gene importance:")
print(perturbation_df)

# === Step 6: Generate embeddings and visualize via UMAP ===
print("⏳ Generating embeddings...")
embeddings = model.get_embeddings(dataset)

reducer = umap.UMAP(random_state=42)
embedding_2d = reducer.fit_transform(embeddings)

umap_df = pd.DataFrame({
    "UMAP1": embedding_2d[:, 0],
    "UMAP2": embedding_2d[:, 1],
    "label": adata.obs["summaryDescription"].values
})

sns.scatterplot(x="UMAP1", y="UMAP2", hue="label", data=umap_df)
plt.title("UMAP Embeddings of RCC Cells")
plt.savefig("UMAP_RCC_cells.png", dpi=300)
plt.show()

# === Step 7: Save top-ranked genes for drug discovery ===
final_genes = perturbation_df.merge(top_genes_attention, on="ensembl_id")
final_genes.to_csv("top_ranked_genes.csv", index=False)
print("✅ Top-ranked genes saved to 'top_ranked_genes.csv'")
