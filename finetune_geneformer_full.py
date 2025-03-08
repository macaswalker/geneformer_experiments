#!/usr/bin/env python
"""
finetune_geneformer_full.py

This script fine-tunes the Geneformer model on the full RCC_data.loom dataset.
It:
  - Loads the full loom file.
  - Strips version numbers from Ensembl IDs and sets adata.var_names accordingly.
  - Filters out any genes with missing Ensembl IDs.
  - Adds a "filter_pass" column to adata.var.
  - Creates dummy classification labels from the last part of "summaryDescription".
  - Processes the AnnData into a Hugging Face dataset (passing gene_names="ensembl_id").
  - Fine-tunes Geneformer for a classification task (e.g., Tumour vs Metastasis).
  - Generates outputs and prints the test accuracy.

Make sure the RCC_data.loom file is available in the same directory (or provide the correct path).
"""

import anndata as ad
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import Sequence, Value

# Import Helical classes
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel

def main():
    # === 1) Load the full loom file ===
    print("Loading RCC_data.loom ...")
    adata = ad.read_loom("RCC_data.loom")
    print(f"Original AnnData shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    # === 2) Set gene identifiers properly ===
    if "ensembl_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["ensembl_id"].apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
        # Remove genes with missing ensembl_id
        adata = adata[:, adata.var["ensembl_id"].notnull()].copy()
        adata.var_names = adata.var["ensembl_id"]
        print("Set adata.var_names to Ensembl IDs (versions stripped).")
    else:
        print("Warning: 'ensembl_id' column not found in adata.var.")
    
    # === 3) Create classification labels ===
    # Assume summaryDescription is of the form "PD43948_Tumour" or "PD43948_Metastasis".
    adata.obs["celltype"] = adata.obs["summaryDescription"].apply(
        lambda x: x.split("_")[-1] if isinstance(x, str) else x
    )
    label_map = {"Tumour": 0, "Metastasis": 1}
    adata = adata[adata.obs["celltype"].isin(label_map.keys())].copy()
    adata.obs["label_int"] = adata.obs["celltype"].map(label_map)
    print("Label distribution in adata.obs:")
    print(adata.obs["label_int"].value_counts())
    
    # === 4) Use the full dataset (do not subset) ===
    print(f"Using full dataset: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    # === 5) Ensure 'filter_pass' column exists in adata.var ===
    if "filter_pass" not in adata.var.columns:
        adata.var["filter_pass"] = True
        print("Added 'filter_pass' column to adata.var.")
    else:
        print("'filter_pass' column already exists in adata.var.")
    
    # === 6) Create GeneformerConfig and FineTuningModel ===
    print("Creating GeneformerConfig ...")
    config = GeneformerConfig(batch_size=16, model_name="gf-6L-30M-i2048")
    print("Creating FineTuningModel (classification head) ...")
    model = GeneformerFineTuningModel(geneformer_config=config, fine_tuning_head="classification", output_size=2)
    
    # === 7) Process AnnData into a Hugging Face dataset ===
    print("Processing AnnData using process_data() ...")
    dataset = model.process_data(adata, gene_names="ensembl_id")
    print("Processed dataset length:", len(dataset))
    
    # Inspect one sample from the dataset.
    sample = dataset[0]
    print("Keys in one processed sample:", sample.keys())
    print("Sample 'input_ids':", sample.get("input_ids", "Not found"))
    print("Sample 'length':", sample.get("length", "Not found"))
    
    # === 8) Add labels, shuffle, and split dataset ===
    labels_list = adata.obs["label_int"].values.tolist()
    dataset = dataset.add_column("labels", labels_list)
    dataset = dataset.shuffle(seed=42)
    split_index = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(split_index))
    test_dataset = dataset.select(range(split_index, len(dataset)))
    print(f"Train dataset size: {len(train_dataset)}; Test dataset size: {len(test_dataset)}")
    
    # === 9) Fine-tune the model ===
    # Here we fine-tune for a specified number of epochs. Adjust epochs as needed.
    epochs = 10  # Change this to your desired number of epochs.
    print(f"Training for {epochs} epochs ...")
    model.train(
        train_dataset=train_dataset,
        label="labels",
        freeze_layers=2,  # You can adjust the number of frozen layers.
        epochs=epochs,
        optimizer_params={"lr": 1e-4},
        lr_scheduler_params={"name": "linear", "num_warmup_steps": 0, "num_training_steps": epochs}
    )
    print("Training complete.")
    
    # === 10) Generate outputs and evaluate ===
    outputs = model.get_outputs(test_dataset)
    preds = outputs.argmax(axis=1)
    true_labels = np.array([ex["labels"] for ex in test_dataset])
    accuracy = (preds == true_labels).mean()
    print("Test Accuracy:", accuracy)
    
    # Optionally, plot a confusion matrix
    if len(np.unique(true_labels)) == 2:
        cm = confusion_matrix(true_labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tumour", "Metastasis"])
        fig, ax = plt.subplots(figsize=(6, 6))
        disp.plot(ax=ax, xticks_rotation='vertical', values_format='.2f', cmap='coolwarm')
        plt.title("Confusion Matrix")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
