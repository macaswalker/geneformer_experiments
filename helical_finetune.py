import anndata as ad
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Helical
from helical.models.geneformer import GeneformerConfig, GeneformerFineTuningModel

def main():
    # 1. Read loom
    adata = ad.read_loom("RCC_data.loom")

    adata_sub = adata[:10, :]  # 10 cells, all genes JUST FOR TESTING

    
    # ----------------------------------------------------------
    # 2. Parse out the cell type from the summaryDescription
    #
    # Assuming summaryDescription has the pattern "PD43948_Tumour" 
    # or "PD43948_Metastasis", etc. We split on "_" and take the 
    # last item as the cell type.
    # ----------------------------------------------------------
    adata.obs["celltype_from_summary"] = (
        adata.obs["summaryDescription"].apply(lambda x: x.split("_")[-1] if isinstance(x, str) else x)
    )
    
    # 3. Keep only cells labeled Tumour or Metastasis
    keep_labels = ["Tumour", "Metastasis"]
    adata = adata[adata.obs["celltype_from_summary"].isin(keep_labels)].copy()
    
    # 4. Map these labels to integer IDs
    label_map = {"Tumour": 0, "Metastasis": 1}
    adata.obs["label_int"] = adata.obs["celltype_from_summary"].map(label_map)
    
    # 5. Configure Geneformer
    config = GeneformerConfig(
        model_name="gf-6L-30M-i2048",  # or whichever model variant you want
        device="cuda" if torch.cuda.is_available() else "cpu",
        batch_size=8
    )
    
    # 6. Setup fine-tuning model
    geneformer_fine_tune = GeneformerFineTuningModel(
        geneformer_config=config,
        fine_tuning_head="classification",
        output_size=2  # two classes (Tumour, Metastasis)
    )

    # 7. Convert data to a huggingface dataset
    dataset = geneformer_fine_tune.process_data(adata)

    # 8. Add label column
    labels_list = adata.obs["label_int"].values.tolist()
    dataset = dataset.add_column("labels", labels_list)

    # 9. (Optional) Shuffle and split train/test
    dataset = dataset.shuffle(seed=42)
    train_dataset = dataset.select(range(int(0.8 * len(dataset))))
    test_dataset  = dataset.select(range(int(0.8 * len(dataset)), len(dataset)))

    # 10. Train
    geneformer_fine_tune.train(
        train_dataset=train_dataset,
        label="labels",
        freeze_layers=0,
        epochs=5,
        optimizer_params={"lr": 1e-4},
        lr_scheduler_params={
            "name": "linear",
            "num_warmup_steps": 0,
            "num_training_steps": 5
        }
    )

    # 11. Predict on test
    test_outputs = geneformer_fine_tune.get_outputs(test_dataset)
    preds = test_outputs.argmax(axis=1)
    true_labels = np.array([ex["labels"] for ex in test_dataset])

    # 12. Evaluate
    accuracy = (preds == true_labels).mean()
    print(f"Test Accuracy: {accuracy:.3f}")

    # 13. (Optional) Confusion matrix
    cm = confusion_matrix(true_labels, preds)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=["Tumour", "Metastasis"]
    )
    fig, ax = plt.subplots(figsize=(5, 5))
    disp.plot(ax=ax, xticks_rotation='vertical')
    plt.show()

if __name__ == "__main__":
    main()
