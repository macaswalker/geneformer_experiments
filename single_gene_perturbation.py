# perturb_fh_gene.py

import torch
import json
import scanpy as sc
import numpy as np
import pandas as pd
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel
from copy import deepcopy

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

# === Step 4: Perturbation analysis for single gene ===
single_gene = "ENSG00000091483"
gene_ids = list(adata.var_names)

if single_gene not in gene_ids:
    raise ValueError(f"Gene ({single_gene}) not found in dataset!")

single_gene_idx = gene_ids.index(single_gene)

# Define perturbation function
def perturb_gene(dataset, gene_idx):
    perturbed_dataset = deepcopy(dataset)
    for example in perturbed_dataset:
        if gene_idx in example["input_ids"]:
            example["input_ids"].remove(gene_idx)
    return perturbed_dataset

original_outputs = model.get_outputs(dataset)
perturbed_dataset = perturb_gene(dataset, single_gene_idx)
perturbed_outputs = model.get_outputs(perturbed_dataset)

# Compute effect on classification probabilities
original_probs = torch.softmax(torch.tensor(original_outputs), dim=1)
perturbed_probs = torch.softmax(torch.tensor(perturbed_outputs), dim=1)

# Calculate average difference in probability for class "Metastasis"
diff_metastasis = (perturbed_probs[:, 1] - original_probs[:, 1]).mean().item()

print(f"✅ Perturbation effect for FH (ENSG00000091483):")
print(f"Average change in Metastasis probability after perturbation: {diff_metastasis:.4f}")

# Save result
with open("single_gene_perturbation_result.txt", "w") as f:
    f.write(f"FH Gene Perturbation Effect (ENSG00000091483):\n")
    f.write(f"Average change in Metastasis probability: {diff_metastasis:.4f}\n")

print("✅ Results saved to 'single_gene_perturbation_result.txt'")