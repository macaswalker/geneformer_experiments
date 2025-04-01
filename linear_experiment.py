import scanpy as sc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

# === Step 1: Load loom data ===
adata = sc.read_loom("RCC_data.loom")

# Clean Ensembl IDs; remove version suffixes if present
adata.var["ensembl_id"] = (
    adata.var["ensembl_id"]
    .astype(str)
    .str.strip()
    .str.replace(r"\..*$", "", regex=True)
)

# Set var_names to Ensembl IDs
adata.var_names = adata.var["ensembl_id"].values

print("First few var_names after setting ensembl_id:")
print(adata.var_names[:10])

# === Step 2: Filter the obs (rows) to relevant samples, define label map ===
adata = adata[adata.obs["summaryDescription"].isin(["Tumour-normal", "Metastasis"])].copy()
label_map = {"Tumour-normal": 0, "Metastasis": 1}
adata.obs["label"] = adata.obs["summaryDescription"].map(label_map)

# === Step 3: Define your top 20 Ensembl IDs ===
# top_20_genes = [
#     "ENSG00000224051", "ENSG00000169972", "ENSG00000188290",
#     "ENSG00000242485", "ENSG00000131584", "ENSG00000221978",
#     "ENSG00000179403", "ENSG00000188157", "ENSG00000131591",
#     "ENSG00000187642", "ENSG00000162576", "ENSG00000078808",
#     "ENSG00000215915", "ENSG00000160087", "ENSG00000184163",
#     "ENSG00000127054", "ENSG00000175756", "ENSG00000187608",
#     "ENSG00000188976", "ENSG00000107404"
# ]

# === Step 4: Check which genes exist in var_names ===
# var_names_set = set(adata.var_names)
# common_genes = [g for g in top_20_genes if g in var_names_set]
# missing_genes = list(set(top_20_genes) - set(common_genes))

# print("\n=== Genes found in adata.var_names ===")
# print(common_genes)
# print("\n=== Genes missing in adata.var_names ===")
# print(missing_genes)

# if len(common_genes) == 0:
#     raise ValueError("None of the specified top_20_genes were found in adata.var_names.")

# # === Step 5: Subset the AnnData to only the found genes, extract X and y ===
# adata_sub = adata[:, common_genes].copy()

# # Convert to dense if needed for scikit-learn
# X = adata_sub.X.toarray() if not isinstance(adata_sub.X, np.ndarray) else adata_sub.X
# y = adata_sub.obs["label"].values

# === Step 5a: Linear Regression Using all Data
# Use ALL genes in your data
X = adata.X.toarray() if not isinstance(adata.X, np.ndarray) else adata.X
y = adata.obs["label"].values



# === Step 6: Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# === Step 7: Train logistic regression ===
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# === Step 8: Evaluation ===
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on test set: {acc}")
print("Classification report:")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------------------------------
#                               PLOTTING
# ---------------------------------------------------------------------------------

# 1) Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cmdisp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Tumour", "Metastasis"])
cmdisp.plot(values_format='d')
plt.title("Confusion Matrix (All Genes")
plt.show()

# 2) ROC Curve
# For LogisticRegression, you can get decision_function or predict_proba
try:
    y_score = clf.decision_function(X_test)
except AttributeError:
    # If the model doesn't have decision_function, use predict_proba
    y_score = clf.predict_proba(X_test)[:, 1]

fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label="Model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (AUC = {roc_auc:.3f})")
plt.legend()
plt.show()

# 3) Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_score)

# Compute approximate area under the precision-recall curve via numeric integration
# or you can use sklearn.metrics.average_precision_score(y_test, y_score)
pr_auc = np.trapz(precision[::-1], recall[::-1])  # simple numeric integration

plt.figure()
plt.plot(recall, precision, label="Model")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall Curve (Area = {pr_auc:.3f})")
plt.legend()
plt.show()
