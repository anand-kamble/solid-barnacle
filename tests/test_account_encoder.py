import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.embeddings.account_encoder import (
    AccountEncoder,
    lorentz_inner,
    lorentz_to_poincare,
    poincare_to_lorentz,
    lorentz_distance,
    LorentzSGD,
)


def build_encoder():
    edges = [
        ("root", "assets"),
        ("root", "liabilities"),
        ("assets", "cash"),
        ("assets", "revenue"),
        ("cash", "bank"),
        ("liabilities", "payable"),
    ]
    nodes = ["root", "assets", "liabilities", "cash", "revenue", "bank", "payable"]
    return AccountEncoder.from_tree(
        dim=4,
        edges=edges,
        node_ids=nodes,
        root="root",
        edge_length=0.8,
        seed=17,
        freeze=True,
        default_representation="lorentz",
    )


def test_lorentz_embeddings_normalized():
    enc = build_encoder()
    emb = enc.get_all("lorentz")
    norms = lorentz_inner(emb, emb).squeeze(-1)
    target = torch.full_like(norms, -1.0)
    assert torch.allclose(norms, target, atol=1e-6)


def test_representation_switches_are_consistent():
    enc = build_encoder()
    node = ["cash"]
    lorentz_emb = enc.forward(node, "lorentz")
    poincare_emb = enc.forward(node, "poincare")
    tangent_emb = enc.forward(node, "tangent")
    assert poincare_emb.shape[-1] == enc.dim
    assert torch.allclose(poincare_emb, lorentz_to_poincare(lorentz_emb))
    assert tangent_emb.shape[-1] == enc.dim + 1


def test_lookup_and_indices_roundtrip():
    enc = build_encoder()
    ids = ["cash", "bank", "payable"]
    idx = enc.ids_to_indices(ids)
    back = enc.indices_to_ids(idx)
    assert back == ids


def test_cone_membership_matches_hierarchy():
    enc = build_encoder()
    pairs = [
        ("root", "assets"),
        ("root", "liabilities"),
        ("assets", "cash"),
        ("cash", "bank"),
    ]
    parents = [p for p, _ in pairs]
    children = [c for _, c in pairs]
    mask = enc.in_cone(parents, children)
    print(mask)
    assert torch.all(mask)


def test_cone_energy_detects_out_of_cone_points():
    enc = build_encoder()
    inside = enc.cone_energy("cash", "bank")
    assert inside.item() < 1e-6
    parent = enc.forward("cash", "lorentz")
    parent_ball = lorentz_to_poincare(parent)[0]
    outside_ball = (-parent_ball * 0.95).unsqueeze(0)
    outside = poincare_to_lorentz(outside_ball)
    energy = enc.cone_energy("cash", child_embeddings=outside)
    assert energy.item() > 1e-4


def test_embed_new_account_updates_state():
    enc = build_encoder()
    before = enc.weight.shape[0]
    point = enc.embed_new_account("petty", "cash", radial_step=0.3, seed=11)
    assert enc.weight.shape[0] == before + 1
    assert "petty" in enc.id_to_index
    assert "petty" in enc.children_by_id["cash"]
    mask = enc.in_cone("cash", child_embeddings=point.unsqueeze(0))
    assert bool(mask.item())


def test_refresh_cone_apertures_overrides_values():
    enc = build_encoder()
    manual = torch.full_like(enc.cone_apertures, 0.2)
    enc.refresh_cone_apertures(values=manual)
    assert torch.allclose(enc.cone_apertures, manual)
    enc.refresh_cone_apertures(kappa=0.5)
    assert not torch.allclose(enc.cone_apertures, manual)


def test_get_all_respects_default_representation():
    enc = build_encoder()
    enc.default_representation = "poincare"
    vec = enc.forward("revenue")
    assert vec.shape[-1] == enc.dim


def test_lorentz_distance_is_zero_for_identical_points():
    enc = build_encoder()
    root = enc.forward("root", "lorentz")
    dist = lorentz_distance(root, root).squeeze(-1)
    assert torch.allclose(dist, torch.zeros_like(dist), atol=1e-6)


def test_lorentz_sgd_preserves_manifold_and_reduces_distance():
    enc = AccountEncoder.from_tree(
        dim=3,
        edges=[("root", "a"), ("a", "b")],
        node_ids=["root", "a", "b"],
        root="root",
        edge_length=0.6,
        seed=7,
        freeze=False,
        default_representation="lorentz",
    )
    optimizer = LorentzSGD([enc.weight], lr=0.05)
    target = enc.forward("root", "lorentz").detach()
    baseline = lorentz_distance(enc.forward("a", "lorentz"), target).item()
    for _ in range(10):
        optimizer.zero_grad()
        current = enc.forward("a", "lorentz")
        loss = lorentz_distance(current, target).pow(2).sum()
        loss.backward()
        optimizer.step()
    updated = enc.forward("a", "lorentz")
    inner = lorentz_inner(updated, updated).item()
    after = lorentz_distance(updated, target).item()
    assert abs(inner + 1.0) < 1e-6
    assert after < baseline


def test_ranking_loss_prefers_closer_nodes():
    enc = build_encoder()
    anchors = ["assets"]
    candidates = [["cash", "payable"]]
    loss_close = enc.ranking_loss(anchors, candidates)
    targets = torch.tensor([1])
    loss_far = enc.ranking_loss(anchors, candidates, targets)
    assert loss_close < loss_far


if __name__ == "__main__":
    tests = [
        test_lorentz_embeddings_normalized,
        test_representation_switches_are_consistent,
        test_lookup_and_indices_roundtrip,
        test_cone_membership_matches_hierarchy,
        test_cone_energy_detects_out_of_cone_points,
        test_embed_new_account_updates_state,
        test_refresh_cone_apertures_overrides_values,
        test_get_all_respects_default_representation,
        test_lorentz_distance_is_zero_for_identical_points,
        test_lorentz_sgd_preserves_manifold_and_reduces_distance,
        test_ranking_loss_prefers_closer_nodes,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print(f"✓ {test.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{passed} passed, {failed} failed")
    if failed > 0:
        exit(1)
