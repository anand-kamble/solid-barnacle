import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch

from models.embeddings.account_encoder import (
    AccountEncoder,
    LorentzSGD,
    lorentz_to_poincare,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
LOGGER = logging.getLogger("train_account_embeddings")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage 1 Lorentz embedding pre-training")
    parser.add_argument("--accounts-csv", type=str, required=True, help="Path to ledger account CSV")
    parser.add_argument(
        "--id-column",
        type=str,
        default="ledger_account_id",
        help="Column containing unique account identifiers",
    )
    parser.add_argument(
        "--parent-column",
        type=str,
        default="parent_ledger_account_id",
        help="Column containing parent identifiers",
    )
    parser.add_argument("--name-column", type=str, default="name", help="Optional column for display names")
    parser.add_argument("--dim", type=int, default=32, help="Embedding dimension (Lorentz space has dim+1 coords)")
    parser.add_argument("--edge-length", type=float, default=1.0, help="Geodesic step between parent and child when initializing tree")
    parser.add_argument("--epochs", type=int, default=400, help="Number of full-batch epochs")
    parser.add_argument("--lr", type=float, default=5e-3, help="LorentzSGD learning rate")
    parser.add_argument("--kappa", type=float, default=0.95, help="Cone aperture scaling constant")
    parser.add_argument("--neg-multiplier", type=int, default=2, help="Negative samples per positive edge")
    parser.add_argument("--rank-k", type=int, default=4, help="Candidates per anchor for ranking loss (≥2)")
    parser.add_argument("--lambda-neg", type=float, default=1.0, help="Weight on negative-cone loss")
    parser.add_argument("--lambda-rank", type=float, default=0.1, help="Weight on Lorentz ranking loss")
    parser.add_argument("--lambda-reg", type=float, default=0.01, help="Weight on boundary regularizer")
    parser.add_argument("--margin", type=float, default=0.25, help="Margin for negative cone loss")
    parser.add_argument("--boundary-eps", type=float, default=1e-3, help="Allowed slack from Poincaré ball boundary")
    parser.add_argument("--seed", type=int, default=17, help="Pseudo-random seed")
    parser.add_argument("--output", type=str, default="artifacts/account_embeddings.pt", help="Path to save checkpoint (.pt)")
    parser.add_argument("--export-csv", type=str, help="Optional CSV export of embeddings in Lorentz and Poincaré charts")
    parser.add_argument("--metadata-json", type=str, help="Optional JSON file to store training metadata")
    parser.add_argument("--log-every", type=int, default=25, help="Logging frequency (epochs)")
    return parser.parse_args()


def load_hierarchy(
    csv_path: Path,
    id_column: str,
    parent_column: str,
    name_column: str,
) -> Tuple[pd.DataFrame, List[Tuple[str, str]], List[str], str]:
    df = pd.read_csv(csv_path)
    if id_column not in df.columns or parent_column not in df.columns:
        raise ValueError(f"CSV must contain {id_column} and {parent_column}")
    df[id_column] = df[id_column].astype(str)
    if name_column in df.columns:
        df[name_column] = df[name_column].fillna(df[id_column]).astype(str)
    else:
        df[name_column] = df[id_column]
    parent_series = df[parent_column].where(df[parent_column].notnull(), "")
    parent_series = parent_series.astype(str)
    df[parent_column] = parent_series
    edges: List[Tuple[str, str]] = []
    root_candidates: List[str] = []
    nodes = df[id_column].tolist()
    for idx, row in df.iterrows():
        node_id = row[id_column]
        parent_id = row[parent_column]
        if parent_id:
            edges.append((parent_id, node_id))
        else:
            root_candidates.append(node_id)
    synthetic_root = "__chart_root__"
    nodes_with_root = [synthetic_root] + nodes
    for root_child in root_candidates:
        edges.append((synthetic_root, root_child))
    return df, edges, nodes_with_root, synthetic_root


def sample_negative_pairs(
    nodes: Sequence[str],
    true_parent: Dict[str, str],
    count: int,
    root_id: str,
) -> List[Tuple[str, str]]:
    negatives: List[Tuple[str, str]] = []
    available = [n for n in nodes if n != root_id]
    while len(negatives) < count:
        parent = random.choice(nodes)
        child = random.choice(available)
        if parent == child:
            continue
        if true_parent.get(child) == parent:
            continue
        negatives.append((parent, child))
    return negatives


def build_ranking_candidates(
    edges: List[Tuple[str, str]],
    nodes: Sequence[str],
    rank_k: int,
) -> Tuple[List[str], List[List[str]]]:
    anchors: List[str] = []
    candidates: List[List[str]] = []
    candidate_pool = list(nodes)
    for parent, child in edges:
        if child == parent:
            continue
        anchors.append(child)
        row = [parent]
        while len(row) < rank_k:
            candidate = random.choice(candidate_pool)
            if candidate == parent or candidate == child:
                continue
            if candidate in row:
                continue
            row.append(candidate)
        candidates.append(row)
    return anchors, candidates


def compute_losses(
    encoder: AccountEncoder,
    edges: List[Tuple[str, str]],
    nodes: List[str],
    root_id: str,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    metrics: Dict[str, float] = {}
    parents = [parent for parent, _ in edges]
    children = [child for _, child in edges]
    pos_loss = encoder.cone_energy(parents, children).mean()
    loss = pos_loss
    metrics["pos"] = float(pos_loss.detach().cpu())
    if args.neg_multiplier > 0:
        parent_map = {child: parent for parent, child in edges}
        neg_pairs = sample_negative_pairs(nodes, parent_map, args.neg_multiplier * len(edges), root_id)
        neg_parents = [p for p, _ in neg_pairs]
        neg_children = [c for _, c in neg_pairs]
        neg_energy = torch.relu(
            args.margin - encoder.cone_energy(neg_parents, neg_children)
        ).mean()
        loss = loss + args.lambda_neg * neg_energy
        metrics["neg"] = float(neg_energy.detach().cpu())
    if args.rank_k > 1 and args.lambda_rank > 0:
        anchors, candidate_rows = build_ranking_candidates(edges, nodes, args.rank_k)
        rank_loss = encoder.ranking_loss(anchors, candidate_rows)
        loss = loss + args.lambda_rank * rank_loss
        metrics["rank"] = float(rank_loss.detach().cpu())
    if args.lambda_reg > 0:
        poincare = lorentz_to_poincare(encoder.get_all("lorentz"))
        norms = torch.norm(poincare, dim=-1)
        boundary = torch.relu(norms - (1.0 - args.boundary_eps)).mean()
        loss = loss + args.lambda_reg * boundary
        metrics["boundary"] = float(boundary.detach().cpu())
    metrics["total"] = float(loss.detach().cpu())
    return loss, metrics


def export_embeddings(
    encoder: AccountEncoder,
    dataframe: pd.DataFrame,
    path: Path,
    id_column: str,
    name_column: str,
) -> None:
    lorentz = encoder.get_all("lorentz").detach().cpu()
    poincare = encoder.get_all("poincare").detach().cpu()
    name_lookup = dataframe.set_index(id_column)[name_column].to_dict()
    records: List[Dict[str, object]] = []
    for idx, account_id in enumerate(encoder.index_to_id):
        row: Dict[str, object] = {
            id_column: account_id,
            "name": name_lookup.get(account_id, account_id),
        }
        row.update({f"lorentz_{d}": float(val) for d, val in enumerate(lorentz[idx].tolist())})
        row.update({f"poincare_{d}": float(val) for d, val in enumerate(poincare[idx].tolist())})
        records.append(row)
    export_df = pd.DataFrame(records)
    export_df.to_csv(path, index=False)
    LOGGER.info("Saved CSV embeddings to %s", path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    csv_path = Path(args.accounts_csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    df, edges, nodes, root_id = load_hierarchy(
        csv_path=csv_path,
        id_column=args.id_column,
        parent_column=args.parent_column,
        name_column=args.name_column,
    )
    encoder = AccountEncoder.from_tree(
        dim=args.dim,
        edges=edges,
        node_ids=nodes,
        root=root_id,
        edge_length=args.edge_length,
        seed=args.seed,
        freeze=False,
        default_representation="lorentz",
        kappa=args.kappa,
    )
    optimizer = LorentzSGD([encoder.weight], lr=args.lr)
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad()
        loss, metrics = compute_losses(encoder, edges, nodes, root_id, args)
        loss.backward()
        optimizer.step()
        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            LOGGER.info("Epoch %d | loss=%.4f pos=%.4f neg=%.4f rank=%.4f boundary=%.4f",
                        epoch,
                        metrics.get("total", 0.0),
                        metrics.get("pos", 0.0),
                        metrics.get("neg", 0.0),
                        metrics.get("rank", 0.0),
                        metrics.get("boundary", 0.0))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": encoder.state_dict(),
            "node_ids": encoder.index_to_id,
            "kappa": encoder.kappa,
            "dim": encoder.dim,
            "args": vars(args),
        },
        output_path,
    )
    LOGGER.info("Saved checkpoint to %s", output_path)
    if args.export_csv:
        export_embeddings(
            encoder=encoder,
            dataframe=df,
            path=Path(args.export_csv),
            id_column=args.id_column,
            name_column=args.name_column,
        )
    if args.metadata_json:
        meta = {
            "num_nodes": len(nodes),
            "num_edges": len(edges),
            "root_id": root_id,
            "args": vars(args),
        }
        metadata_path = Path(args.metadata_json)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(meta, indent=2))
        LOGGER.info("Saved metadata to %s", metadata_path)


if __name__ == "__main__":
    main()

