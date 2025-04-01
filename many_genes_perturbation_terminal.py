# perturb_multiple_genes_simultaneously.py
import torch
import json
import scanpy as sc
import numpy as np
import pandas as pd
import sys
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
from copy import deepcopy

# Check for command-line arguments
if len(sys.argv) < 2:
    raise ValueError("Please provide at least one Ensembl gene ID to perturb as a command-line argument.")

perturb_genes = sys.argv[1:]

# === Step 1: Load the fine-tuned Geneformer model ===
device = "cuda" if torch.cuda.is_available() else "cpu"

with open("fine_tuned_model/config.json") as f:
    config_dict = json.load(f)

for key in ['device', 'input_size', 'special_token', 'embsize']:
    config_dict.pop(key, None)

config = GeneformerConfig(**config_dict, device=device)
model = GeneformerFineTuningModel(config, fine_tuning_head="classification", output_size=2)
model.load_state_dict(torch.load("fine_tuned_model/pytorch_model.bin", map_location=device))
model.model.to(device).eval()
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

# === Step 4: Perturbation analysis for all genes simultaneously ===
gene_ids = list(adata.var_names)
perturb_gene_indices = []

for gene in perturb_genes:
    if gene not in gene_ids:
        print(f"⚠️ Gene ({gene}) not found in dataset! Skipping.")
        continue
    perturb_gene_indices.append(gene_ids.index(gene))

# Define multi-gene perturbation function
def perturb_multiple_genes(dataset, gene_indices):
    perturbed_dataset = deepcopy(dataset)
    for example in perturbed_dataset:
        example["input_ids"] = [gene for gene in example["input_ids"] if gene not in gene_indices]
    return perturbed_dataset

original_outputs = model.get_outputs(dataset)
original_probs = torch.softmax(torch.tensor(original_outputs), dim=1)

perturbed_dataset = perturb_multiple_genes(dataset, perturb_gene_indices)
perturbed_outputs = model.get_outputs(perturbed_dataset)
perturbed_probs = torch.softmax(torch.tensor(perturbed_outputs), dim=1)

# Calculate average difference in probability for class "Metastasis"
diff_metastasis = (perturbed_probs[:, 1] - original_probs[:, 1]).mean().item()

# Create a clean filename using the provided gene names
gene_names_str = "_".join(perturb_genes)

# Save results
with open(f"{gene_names_str}_multi_gene_perturbation_results.txt", "w") as f:
    f.write(f"Perturbation effect for genes: {', '.join(perturb_genes)}\n")
    f.write(f"Average change in Metastasis probability after perturbation: {diff_metastasis:.4f}\n")

print(f"✅ Perturbation effect for genes ({', '.join(perturb_genes)}):")
print(f"Average change in Metastasis probability after perturbation: {diff_metastasis:.4f}")
print(f"✅ Results saved to '{gene_names_str}_multi_gene_perturbation_results.txt'")