#!/usr/bin/env python
"""
diagnostic_mapping.py

This script performs a step-by-step diagnostic of your dataset and Helical’s gene mapping/tokenization.
It will:
  1. Load your AnnData from the loom file.
  2. Print out the columns and first few rows of adata.var.
  3. Strip version numbers from the Ensembl IDs.
  4. Load Helical’s ensembl mapping dictionary and compute the intersection with your gene IDs.
  5. Set up a dummy classification label using the last part of summaryDescription.
  6. Create a GeneformerConfig and GeneformerFineTuningModel.
  7. Process the AnnData into a Hugging Face dataset using process_data().
  8. Print out keys and contents of one sample from the processed dataset.

This should help pinpoint why no tokens are being produced.
"""

import os
import pickle
import anndata as ad
import numpy as np
import torch
import matplotlib.pyplot as plt

# Import Value and Sequence for proper column casting if needed
from datasets import Value, Sequence

# Import Helical classes
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel

def diagnostic_mapping():
    # Path to Helical's mapping dictionary (adjust if necessary)
    mapping_path = os.path.expanduser("~/.cache/helical/models/geneformer/v1/ensembl_mapping_dict.pkl")
    if not os.path.exists(mapping_path):
        print("Mapping dictionary not found at:", mapping_path)
        return
    with open(mapping_path, 'rb') as f:
        mapping_dict = pickle.load(f)
    print("Number of keys in ensembl_mapping_dict:", len(mapping_dict))
    
    # Load AnnData
    print("\nLoading RCC_data.loom ...")
    adata = ad.read_loom("RCC_data.loom")
    print(f"Loaded AnnData with shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    # Strip version numbers from ensembl_id
    if "ensembl_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["ensembl_id"].apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
        gene_ids = adata.var["ensembl_id"].tolist()
        print("Total genes in dataset:", len(gene_ids))
        mapped_genes = [gene for gene in gene_ids if gene in mapping_dict]
        print("Number of mapped genes:", len(mapped_genes))
        if mapped_genes:
            print("First 10 mapped genes:", mapped_genes[:10])
        else:
            print("No genes mapped. Your gene IDs may not match the keys in the mapping dictionary.")
    else:
        print("Warning: 'ensembl_id' column not found in adata.var.")

def process_and_inspect():
    # Load AnnData from your loom file
    print("\nLoading RCC_data.loom for processing ...")
    adata = ad.read_loom("RCC_data.loom")
    print(f"AnnData shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    # Inspect adata.var
    print("\nAnnData.var columns:", adata.var.columns.tolist())
    print("First 5 rows of adata.var:")
    print(adata.var.head())
    
    # Ensure Ensembl IDs are stripped of version numbers
    if "ensembl_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["ensembl_id"].apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
    
    # Ensure a 'filter_pass' column exists
    if "filter_pass" not in adata.var.columns:
        print("\nAdding 'filter_pass' column to adata.var (all True).")
        adata.var["filter_pass"] = True
    else:
        print("\n'filter_pass' column exists in adata.var.")
    
    # For a dummy classification, create a new column in adata.obs.
    # Here we use the last part of summaryDescription (e.g., "PD43948_Tumour" -> "Tumour").
    print("\nCreating dummy classification labels from 'summaryDescription' ...")
    adata.obs["celltype_from_summary"] = adata.obs["summaryDescription"].apply(
        lambda x: x.split("_")[-1] if isinstance(x, str) else x
    )
    label_map = {"Tumour": 0, "Metastasis": 1}
    # Filter only cells that match our label_map keys.
    adata = adata[adata.obs["celltype_from_summary"].isin(label_map.keys())].copy()
    adata.obs["label_int"] = adata.obs["celltype_from_summary"].map(label_map)
    print("Label distribution in adata.obs:")
    print(adata.obs["label_int"].value_counts())
    
    # Set up the Geneformer fine-tuning model
    print("\nCreating GeneformerConfig and FineTuningModel ...")
    geneformer_config = GeneformerConfig(
        batch_size=8,
        model_name="gf-6L-30M-i2048"
    )
    fine_tune_model = GeneformerFineTuningModel(
        geneformer_config=geneformer_config,
        fine_tuning_head="classification",
        output_size=2
    )
    print("Created FineTuningModel.")
    
    # Process the AnnData into a Hugging Face dataset
    print("\nProcessing AnnData using process_data() ...")
    dataset = fine_tune_model.process_data(adata)
    print("Processed dataset length:", len(dataset))
    
    # Inspect one sample from the dataset
    sample = dataset[0]
    print("\nKeys in one processed sample:")
    print(sample.keys())
    print("Sample content:")
    print("input_ids:", sample.get("input_ids", "Not found"))
    print("length:", sample.get("length", "Not found"))
    if "input_ids" in sample and sample["input_ids"]:
        print("Type of first token in input_ids:", type(sample["input_ids"][0]))
    else:
        print("No tokens found in sample['input_ids'].")
    
if __name__ == "__main__":
    print("=== Running Mapping Diagnostics ===")
    diagnostic_mapping()
    
    print("\n=== Running Dataset Processing Diagnostics ===")
    process_and_inspect()
