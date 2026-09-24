#!/usr/bin/env python
"""
Script d'entraînement / évaluation FLOCAT-GENAGN.

Exemple d'utilisation :
    python run_flocat.py \
        --inlier_topic bottle \
        --embedding_type wide_resnet50_2 \
        --data_root /home/2017025/ayouce01/FLOCAT-GENAGN/What-if-Flocat-were-truly-generic-and-agnostic/data/embeddings \
        --n_runs 5 \
        --output_file results/results.txt
"""

import argparse
import time
import os

import torch
import numpy as np

import sys
sys.path.append('./FLOCAT-GENAGN/What-if-Flocat-were-truly-generic-and-agnostic')
from GENAGN.flocat import flocatTrainer, flocat


# ----------------------------------------------------------------------
# Parsing des arguments
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraîne et évalue FLOCAT-GENAGN sur des embeddings pré-calculés."
    )

    # Données
    parser.add_argument("--data_root", type=str, required=True,
                         help="Racine du dossier contenant les embeddings.")
    parser.add_argument("--embedding_type", type=str, default="wide_resnet50_2",
                         help="Sous-dossier / type d'embedding (ex: wide_resnet50_2, sentence_bert...).")
    parser.add_argument("--inlier_topic", type=str, required=True,
                         help="Classe inlier (ex: bottle, World...).")
    parser.add_argument("--dataset_name", type=str, default=None,
                         help="Nom du dataset pour le log (par défaut = embedding_type).")

    # Hyperparamètres du modèle
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--freq_embed_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lambda_love", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr_epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--coef_var", type=float, default=1e-3)
    parser.add_argument("--target", type=str, default="gaussian-neigh")

    # Evaluation
    parser.add_argument("--eval_type", type=str, default="norm-centroid")
    parser.add_argument("--n_steps", type=int, default=10)

    # Répétitions / device / sortie
    parser.add_argument("--diy_patch_size", type=int, default=64)
    parser.add_argument("--n_runs", type=int, default=3,
                         help="Nombre de runs pour calculer moyenne ± std.")
    parser.add_argument("--seed", type=int, default=None,
                         help="Seed initiale (incrémentée à chaque run). Si None: pas de seed fixée.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_file", type=str, default="results.txt",
                         help="Fichier texte dans lequel les résultats sont ajoutés (append).")

    return parser.parse_args()

def load_embeddings(path):
    return torch.load(path, map_location="cpu")


def load_data(args):
    base = os.path.join(args.data_root, args.embedding_type, args.inlier_topic)

    if args.data_root.split("/")[-1] == 'diypatch_level':
        base = os.path.join(args.data_root, args.embedding_type+f"_manual_p{args.diy_patch_size}", args.inlier_topic)


    data_train = load_embeddings(os.path.join(base, "train.pt"))
    X_train = data_train["embeddings"]

    data_test = load_embeddings(os.path.join(base, "test.pt"))
    X_test = data_test["embeddings"]
    y_test = data_test["labels"]

    return X_train, X_test, y_test


def run_once(args, X_train, X_test, y_test):

    device = args.device

    # ------------------------------------------------------------------
    # Détection du format des embeddings :
    #   - X_train.dim() == 3 -> [N, num_patches, C] (patch tokens, ResNet/ViT)
    #       -> on construit un attention mask "plein" (tout à 1), puisqu'il
    #          n'y a pas de padding (toutes les images produisent le même
    #          nombre de patchs).
    #   - X_train.dim() == 2 -> [N, C] (embedding CLS global par image)
    #       -> pas de notion de "patchs" ici : on ne construit pas de mask,
    #          on laisse flocat le gérer en interne (attentions_mask=None).
    #          Idem pour trainer.test(), on ne passe pas de mask du tout,
    #          il se met à None par défaut.
    # ------------------------------------------------------------------
    is_patch_embeddings = X_train.dim() == 3
    latent_dim = X_train.shape[-1]
    
    if is_patch_embeddings:
        num_patches = X_train.shape[1]
        attentions_train_mask = torch.ones((X_train.shape[0], num_patches), dtype=torch.long)
    else:
        attentions_train_mask = None

    flocat_config = {
        "latent_dim": latent_dim,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "n_heads": args.n_heads,
        "freq_embed_size": args.freq_embed_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "lambda_love": args.lambda_love,
        "epochs": args.epochs,
        "lr_epochs": args.lr_epochs,
        "batch_size": args.batch_size,
        "coef_var": args.coef_var,
        "target": args.target,
        "source": X_train,
        "attentions_mask": attentions_train_mask,
        "device": device,
    }

    flow_model = flocat(
        latent_dim=flocat_config["latent_dim"],
        hidden_dim=flocat_config["hidden_dim"],
        depth=flocat_config["depth"],
        n_heads=flocat_config["n_heads"],
    ).to(device)

    trainer = flocatTrainer(flow_model, flocat_config)

    t0 = time.time()
    trainer.train(True)
    t1 = time.time()
    train_time = t1 - t0

    if is_patch_embeddings:
        num_patches_test = X_test.shape[1]
        attentions_test_mask = torch.ones((X_test.shape[0], num_patches_test), dtype=torch.long)
        auc, fpr, ap = trainer.test(
            X_test, y_test, attentions_test_mask,
            type=args.eval_type, n_steps=args.n_steps,
        )
    else:
        # Pas de mask passé -> flocat le gère en interne (défaut None)
        auc, fpr, ap = trainer.test(
            X_test, y_test,
            type=args.eval_type, n_steps=args.n_steps,
        )

    return {"auc": auc, "fpr95": fpr, "ap": ap, "train_time": train_time}


def write_results(args, metrics_list):
    aucs = np.array([m["auc"] for m in metrics_list])
    aps = np.array([m["ap"] for m in metrics_list])
    fprs = np.array([m["fpr95"] for m in metrics_list])
    times = np.array([m["train_time"] for m in metrics_list])

    dataset_name = args.dataset_name or args.embedding_type

    block = (
        "========================================\n"
        f"Dataset:        {dataset_name}\n"
        f"Inlier class:   {args.inlier_topic}\n"
        f"Embedding type: {args.embedding_type}\n"
        f"AD model:       flocat-genagn\n"
        f"Training time:  {times.mean():.2f} sec\n"
        "----------------------------------------\n"
        f"AUC:            {aucs.mean():.4f} ± {aucs.std():.4f}\n"
        f"Avg Precision:  {aps.mean():.4f} ± {aps.std():.4f}\n"
        f"FPR@95:         {fprs.mean():.4f} ± {fprs.std():.4f}\n"
        "========================================\n\n"
    )

    os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
    with open(args.output_file, "a") as f:
        f.write(block)

    print(block)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()

    X_train, X_test, y_test = load_data(args)

    all_metrics = []

    for n_run in range(args.n_runs):

        print(f"--------- n_run {n_run+1} -----------------\n")

        metrics = run_once(args, X_train, X_test, y_test)
        print(
            f"AUC: {metrics['auc']:.4f} | "
            f"FPR@95: {metrics['fpr95']:.4f} | "
            f"AP: {metrics['ap']:.4f} | "
            f"time: {metrics['train_time']:.2f}s\n"
        )

        all_metrics.append(metrics)


    write_results(args, all_metrics)


if __name__ == "__main__":
    main()