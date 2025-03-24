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

# Remove existing 'device' key if it exists
config_dict.pop('device', None)
config_dict.pop('input_size', None)
config_dict.pop('special_token', None)
config_dict.pop('embsize', None)

# Now safely add 'device=device'
config = GeneformerConfig(**config_dict, device=device)


config = GeneformerConfig(**config_dict, device=device)
model = GeneformerFineTuningModel(config, fine_tuning_head="classification", output_size=2)
model.load_state_dict(torch.load("fine_tuned_model/pytorch_model.bin", map_location=device))

# Use underlying model for inference
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
print("✅ Data tokenized successfully.")

# === Step 4: Attention-based gene importance (no separate gene2idx) ===

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict

print("⏳ Extracting attention weights with batch-wise approach (using adata.var_names)...")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.model.eval().to(device)

###############################################################################
# 4.1) Make a list of your Ensembl IDs from the AnnData object
###############################################################################
gene_ids = list(adata.var_names)  
# We'll assume gene_ids[0] corresponds to token index 0, gene_ids[1] -> token 1, etc.

###############################################################################
# 4.2) Define a collate_fn to pad within each batch
###############################################################################
def collate_fn(batch):
    """
    Each 'example' in 'batch' is a dict from the Geneformer dataset,
    with example["input_ids"] = [0,1,2,...,N-1] (for N genes).
    We pad them so that within a batch, all samples match the longest length.
    """
    input_ids_list = [torch.tensor(example["input_ids"]) for example in batch]
    labels = [example["label"] for example in batch]

    input_ids_padded = pad_sequence(
        input_ids_list, batch_first=True, padding_value=0
    )

    return {
        "input_ids": input_ids_padded,
        "labels": torch.tensor(labels)
    }

###############################################################################
# 4.3) Create the DataLoader with a small batch_size if you get OOM
###############################################################################
dataloader = DataLoader(
    dataset, 
    batch_size=2,  # reduce to 1 if you still run out of GPU memory
    shuffle=False, 
    collate_fn=collate_fn
)

###############################################################################
# 4.4) Accumulate the per-gene "attention received" across all cells
###############################################################################
gene_attention_sums = defaultdict(float)
gene_counts = defaultdict(int)

with torch.no_grad():
    for batch_data in dataloader:
        input_ids = batch_data["input_ids"].to(device)
        # output_attentions=True -> returns attention matrices
        outputs = model.model(input_ids=input_ids, output_attentions=True)
        
        # outputs.attentions is a tuple of length = num_layers
        # each element has shape [batch_size, num_heads, seq_len, seq_len]
        attn_tensor = torch.stack(outputs.attentions, dim=0)
        # shape: [num_layers, batch_size, num_heads, seq_len, seq_len]

        # Average across layers and heads => shape [batch_size, seq_len, seq_len]
        attn_avg = attn_tensor.mean(dim=(0, 2))

        # Loop over each sample in the batch
        for i in range(attn_avg.size(0)):  # batch_size
            single_attn_matrix = attn_avg[i]  # shape: [seq_len, seq_len]

            # Identify valid positions (ignore padded zeros)
            valid_positions = (input_ids[i] != 0).nonzero(as_tuple=True)[0]

            for pos in valid_positions:
                # 'pos' is the same index as in gene_ids[pos]
                ensembl_id = gene_ids[pos]

                # "Attention received" by this gene => sum over the column
                received_attn = single_attn_matrix[:, pos].sum().item()

                gene_attention_sums[ensembl_id] += received_attn
                gene_counts[ensembl_id] += 1

###############################################################################
# 4.5) Build a DataFrame of average attention per gene
###############################################################################
scores = []
for gene, total_attn in gene_attention_sums.items():
    avg_attn = total_attn / gene_counts[gene]
    scores.append((gene, avg_attn))

attention_df = pd.DataFrame(scores, columns=["ensembl_id", "attention_score"])
attention_df.sort_values("attention_score", ascending=False, inplace=True)

top_genes_attention = attention_df.head(20)
print("✅ Top 20 genes based on aggregated attention:")
print(top_genes_attention)


# === Step 5: Perturbation-based gene importance ===
def perturb_and_measure_importance(model, dataset, gene_idx):
    perturbed_dataset = deepcopy(dataset)
    for i in range(len(perturbed_dataset)):
        if gene_idx in perturbed_dataset[i]["input_ids"]:
            perturbed_dataset[i]["input_ids"].remove(gene_idx)

    # model.get_outputs(...) => returns a NumPy array
    original_outputs_np = model.get_outputs(dataset)
    perturbed_outputs_np = model.get_outputs(perturbed_dataset)

    # Convert to torch tensors
    original_outputs_t = torch.from_numpy(original_outputs_np)
    perturbed_outputs_t = torch.from_numpy(perturbed_outputs_np)

    # Then apply PyTorch softmax
    original_probs = torch.softmax(original_outputs_t, dim=1)
    perturbed_probs = torch.softmax(perturbed_outputs_t, dim=1)

    # Compute absolute difference and mean over all samples
    diff = (original_probs - perturbed_probs).abs().mean().item()
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
