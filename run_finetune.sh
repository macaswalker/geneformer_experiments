#!/bin/bash
#SBATCH --job-name=finetune_geneformer
#SBATCH --output=finetune_geneformer_%j.out
#SBATCH --error=finetune_geneformer_%j.err
#SBATCH -A HAN-SL3-GPU
#SBATCH --partition=ampere
#SBATCH --time=12:00:00           # 12 hours walltime; adjust if needed
#SBATCH --mem=64G                 # Adjust memory as needed
#SBATCH --cpus-per-task=4         # Number of CPU cores per task
#SBATCH --gres=gpu:4              # Request 4 GPUs (full node on ampere)

# Load necessary modules (adjust these to your HPC system's modules)
module load python/3.9
module load cuda/11.3

# Install required packages in your user space (if not pre-installed)
pip install --user -r requirements.txt

# Run the fine-tuning script
python finetune_geneformer_full.py
