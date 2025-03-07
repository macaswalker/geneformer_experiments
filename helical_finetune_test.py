import anndata as ad
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from datasets import Sequence, Value  # for column type casting, if needed

# Import Helical classes
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel

def main():
    # === 1) Load the loom file ===
    print("Loading RCC_data.loom ...")
    adata = ad.read_loom("RCC_data.loom")
    print(f"Original AnnData shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    # === 2) Set gene identifiers properly ===
    # If "ensembl_id" is in adata.var, strip version numbers and set as var_names.
    if "ensembl_id" in adata.var.columns:
        adata.var["ensembl_id"] = adata.var["ensembl_id"].apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
        # Remove genes with missing ensembl_id
        adata = adata[:, adata.var["ensembl_id"].notnull()].copy()
        # Set the var_names to ensembl_id so mapping uses these gene IDs.
        adata.var_names = adata.var["ensembl_id"]
        print("Set adata.var_names to Ensembl IDs (versions stripped).")
    else:
        print("Warning: 'ensembl_id' column not found in adata.var.")
    
    # === 3) Create dummy classification labels ===
    # Assume summaryDescription is of the form "PD43948_Tumour" or "PD43948_Metastasis".
    adata.obs["celltype"] = adata.obs["summaryDescription"].apply(
        lambda x: x.split("_")[-1] if isinstance(x, str) else x
    )
    # Define a simple label mapping
    label_map = {"Tumour": 0, "Metastasis": 1}
    # Filter cells that match our keys:
    adata = adata[adata.obs["celltype"].isin(label_map.keys())].copy()
    adata.obs["label_int"] = adata.obs["celltype"].map(label_map)
    print("Label distribution in adata.obs:")
    print(adata.obs["label_int"].value_counts())
    
    # === 4) Optionally, subset to a small number for quick testing ===
    adata = adata[:10, :].copy()
    print(f"Subset AnnData shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
    
    # === 5) Ensure 'filter_pass' column exists in adata.var ===
    if "filter_pass" not in adata.var.columns:
        # Create a column of True for all genes
        adata.var["filter_pass"] = True
        print("Added 'filter_pass' column to adata.var.")
    else:
        print("'filter_pass' column already exists in adata.var.")
    
    # === 6) Create GeneformerConfig and FineTuningModel ===
    print("Creating GeneformerConfig ...")
    config = GeneformerConfig(batch_size=8, model_name="gf-6L-30M-i2048")
    print("Creating FineTuningModel (classification head) ...")
    model = GeneformerFineTuningModel(geneformer_config=config, fine_tuning_head="classification", output_size=2)
    
    # === 7) Process AnnData into a Hugging Face dataset ===
    print("Processing AnnData using process_data() ...")
    # Pass gene_names explicitly as recommended.
    dataset = model.process_data(adata, gene_names="ensembl_id")
    print("Processed dataset length:", len(dataset))
    
    # Inspect one sample from the processed dataset.
    sample = dataset[0]
    print("Keys in one processed sample:", sample.keys())
    print("Sample 'input_ids':", sample.get("input_ids", "Not found"))
    print("Sample 'length':", sample.get("length", "Not found"))
    if "input_ids" in sample and sample["input_ids"]:
        print("Type of first token in input_ids:", type(sample["input_ids"][0]))
    else:
        print("No tokens found in sample['input_ids'].")
    
    # === 8) Add labels, shuffle, and split dataset ===
    labels_list = adata.obs["label_int"].values.tolist()
    dataset = dataset.add_column("labels", labels_list)
    dataset = dataset.shuffle(seed=42)
    split_index = int(0.8 * len(dataset))
    train_dataset = dataset.select(range(split_index))
    test_dataset = dataset.select(range(split_index, len(dataset)))
    print(f"Train dataset size: {len(train_dataset)}; Test dataset size: {len(test_dataset)}")
    
    # === 9) Fine-tune the model (run for 1 epoch for quick test) ===
    print("Training for 1 epoch (quick test) ...")
    model.train(
        train_dataset=train_dataset,
        label="labels",
        freeze_layers=0,
        epochs=1,
        optimizer_params={"lr": 1e-4},
        lr_scheduler_params={"name": "linear", "num_warmup_steps": 0, "num_training_steps": 1}
    )
    print("Training complete.")
    
    # === 10) Generate outputs and evaluate ===
    outputs = model.get_outputs(test_dataset)
    preds = outputs.argmax(axis=1)
    true_labels = np.array([ex["labels"] for ex in test_dataset])
    accuracy = (preds == true_labels).mean()
    print("Test Accuracy:", accuracy)
    
    if len(np.unique(true_labels)) == 2:
        cm = confusion_matrix(true_labels, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tumour", "Metastasis"])
        fig, ax = plt.subplots(figsize=(4, 4))
        disp.plot(ax=ax, xticks_rotation='vertical')
        plt.show()
    
if __name__ == "__main__":
    main()