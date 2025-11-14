import torch
from torch import nn
import torch.nn.functional as F
from typing import Callable, Dict, Iterable, Sequence, Mapping, Optional, Hashable, Tuple, List, Union, cast

DTYPE = torch.float64
DEVICE = torch.device("cpu")
EPS = 1e-12


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    spatial = (x[..., 1:] * y[..., 1:]).sum(dim=-1, keepdim=True)
    return -x[..., :1] * y[..., :1] + spatial


def lorentz_norm(u: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    return torch.sqrt(torch.clamp(lorentz_inner(u, u), min=eps))


def lorentz_distance(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    value = torch.clamp(-lorentz_inner(x, y), min=1.0 + eps)
    return torch.arccosh(value)


def project_tangent(base: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    base, vec = torch.broadcast_tensors(base, vec)
    inner = lorentz_inner(base, vec)
    return vec + inner * base


def normalize_lorentz(points: torch.Tensor) -> torch.Tensor:
    spatial = points[..., 1:]
    x0 = torch.sqrt(torch.clamp(1.0 + (spatial * spatial).sum(dim=-1, keepdim=True), min=1.0 + EPS))
    return torch.cat([x0, spatial], dim=-1)


def expmap(base: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    v = project_tangent(base, vec)
    norm = lorentz_norm(v)
    coef1 = torch.cosh(norm)
    coef2 = torch.sinh(norm) / torch.clamp(norm, min=EPS)
    return coef1 * base + coef2 * v


def lorentz_log_map(base: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    inner = lorentz_inner(base, point)
    alpha = torch.clamp(-inner, min=1.0 + EPS)
    omega = torch.arccosh(alpha)
    direction = point + inner * base
    tangent = project_tangent(base, direction)
    denom = torch.sqrt(torch.clamp(alpha * alpha - 1.0, min=EPS))
    scale = omega / denom
    return scale * tangent


def lorentz_origin(dim: int, device=DEVICE, dtype=DTYPE) -> torch.Tensor:
    vec = torch.zeros(dim + 1, device=device, dtype=dtype)
    vec[..., 0] = 1.0
    return vec


def lorentz_to_poincare(x: torch.Tensor) -> torch.Tensor:
    denom = torch.clamp(x[..., :1] + 1.0, min=EPS)
    return x[..., 1:] / denom


def poincare_to_lorentz(x: torch.Tensor) -> torch.Tensor:
    r2 = (x * x).sum(dim=-1, keepdim=True)
    denom = torch.clamp(1.0 - r2, min=EPS)
    x0 = (1.0 + r2) / denom
    spatial = 2.0 * x / denom
    return torch.cat([x0, spatial], dim=-1)


def mobius_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    xy = (x * y).sum(dim=-1, keepdim=True)
    x2 = (x * x).sum(dim=-1, keepdim=True)
    y2 = (y * y).sum(dim=-1, keepdim=True)
    num = (1.0 + 2.0 * xy + y2) * x + (1.0 - x2) * y
    den = torch.clamp(1.0 + 2.0 * xy + x2 * y2, min=EPS)
    return num / den


def mobius_sub(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return mobius_add(x, -y)


def poincare_log_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    diff = mobius_sub(y, x)
    norm_diff = torch.norm(diff, dim=-1, keepdim=True)
    lam = 2.0 / torch.clamp(1.0 - (x * x).sum(dim=-1, keepdim=True), min=EPS)
    scale = 2.0 / lam
    atanh_arg = torch.clamp(lam * norm_diff, max=1.0 - 1e-6)
    factor = torch.atanh(atanh_arg)
    direction = diff / torch.clamp(norm_diff, min=EPS)
    return scale * factor * direction


def cone_half_aperture_ball(x: torch.Tensor, kappa: float = 0.95) -> torch.Tensor:
    norm = torch.norm(x, dim=-1)
    num = kappa * (1.0 - norm * norm)
    den = norm + 1e-6
    value = num / den
    value = torch.clamp(value, min=-1.0 + 1e-6, max=1.0 - 1e-6)
    return torch.arcsin(value)


def geodesic_angle_ball(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    axis = poincare_log_map(u, torch.zeros_like(u))
    direction = poincare_log_map(u, v)
    axis_norm = torch.norm(axis, dim=-1)
    dir_norm = torch.norm(direction, dim=-1)
    inner = (axis * direction).sum(dim=-1)
    denom = torch.clamp(axis_norm * dir_norm, min=EPS)
    cosine = torch.clamp(inner / denom, min=-1.0 + 1e-6, max=1.0 - 1e-6)
    angle = torch.arccos(cosine)
    mask = (axis_norm < 1e-9) | (dir_norm < 1e-9)
    return torch.where(mask, torch.zeros_like(angle), angle)


def entailment_energy_lorentz(u: torch.Tensor, v: torch.Tensor, kappa: float = 0.95) -> torch.Tensor:
    u_ball = lorentz_to_poincare(u)
    v_ball = lorentz_to_poincare(v)
    phi = geodesic_angle_ball(u_ball, v_ball)
    psi = cone_half_aperture_ball(u_ball, kappa)
    return torch.relu(phi - psi).unsqueeze(-1)


def sample_tangent_directions(base: torch.Tensor, count: int, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    if base.dim() == 1:
        base = base.unsqueeze(0)
    d = base.shape[-1]
    noise = torch.randn(count, d, device=base.device, dtype=base.dtype, generator=generator)
    base_expanded = base.expand(count, -1)
    projected = project_tangent(base_expanded, noise)
    norms = lorentz_norm(projected)
    return projected / torch.clamp(norms, min=EPS)


def build_children(edges: Iterable[Tuple[Hashable, Hashable]]) -> Dict[Hashable, List[Hashable]]:
    children: Dict[Hashable, List[Hashable]] = {}
    for parent, child in edges:
        children.setdefault(parent, []).append(child)
        children.setdefault(child, [])
    return children


def build_parent_map(edges: Iterable[Tuple[Hashable, Hashable]]) -> Dict[Hashable, Optional[Hashable]]:
    parents: Dict[Hashable, Optional[Hashable]] = {}
    for parent, child in edges:
        parents.setdefault(parent, None)
        parents[child] = parent
    return parents


class LorentzSGD(torch.optim.Optimizer):
    def __init__(self, params, lr: float = 1e-2):
        if lr <= 0:
            raise ValueError("lr must be positive")
        defaults = dict(lr=lr)
        super().__init__(params, defaults)

    @staticmethod
    def _apply_metric(grad: torch.Tensor) -> torch.Tensor:
        updated = grad.clone()
        updated[..., :1] = -updated[..., :1]
        return updated

    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                if param.data.shape[-1] < 2:
                    raise ValueError("Lorentz parameters require dim>=2")
                grad = self._apply_metric(param.grad.data)
                with torch.no_grad():
                    projected = project_tangent(param.data, grad)
                    update = -lr * projected
                    next_point = expmap(param.data, update)
                    param.data.copy_(next_point)
        return None


class AccountEncoder(nn.Module):
    def __init__(
        self,
        embeddings: torch.Tensor,
        node_ids: Sequence[Hashable],
        parents: Optional[Mapping[Hashable, Optional[Hashable]]] = None,
        cone_apertures: Optional[Union[Sequence[float], Mapping[Hashable, float], torch.Tensor]] = None,
        freeze: bool = True,
        default_representation: str = "lorentz",
        kappa: float = 0.95,
        device=DEVICE,
        dtype=DTYPE,
    ):
        super().__init__()
        tensor = torch.as_tensor(embeddings, dtype=dtype, device=device)
        if tensor.dim() != 2:
            raise ValueError("embeddings must be 2D")
        if tensor.shape[1] < 2:
            raise ValueError("embedding dimension must be at least 2")
        normalized = normalize_lorentz(tensor)
        self.weight = nn.Parameter(normalized, requires_grad=not freeze)
        self.dim = normalized.shape[1] - 1
        ids = list(node_ids)
        if len(ids) != normalized.shape[0]:
            raise ValueError("node_ids length mismatch")
        self.index_to_id = list(ids)
        self.id_to_index = {nid: i for i, nid in enumerate(ids)}
        self.kappa = float(kappa)
        self.default_representation = default_representation.lower()
        parent_map: Dict[Hashable, Optional[Hashable]] = {}
        if parents is None:
            for nid in ids:
                parent_map[nid] = None
        else:
            for nid in ids:
                parent_map[nid] = parents.get(nid)
                if parent_map[nid] is not None and parent_map[nid] not in self.id_to_index:
                    raise ValueError(f"parent {parent_map[nid]} missing from node_ids")
        self.parent_by_id = parent_map
        children: Dict[Hashable, List[Hashable]] = {nid: [] for nid in ids}
        for child, parent in parent_map.items():
            if parent is not None:
                children.setdefault(parent, []).append(child)
        self.children_by_id = children
        self.depths = self._compute_depths()
        origin = lorentz_origin(self.dim, device=self.weight.device, dtype=self.weight.dtype)
        self.register_buffer("origin_vec", origin)
        self._init_cones(cone_apertures)

    def _compute_depths(self) -> Dict[Hashable, int]:
        depths: Dict[Hashable, int] = {}
        roots = [nid for nid, parent in self.parent_by_id.items() if parent is None]
        if not roots:
            root = self.index_to_id[0]
            self.parent_by_id[root] = None
            roots = [root]
        queue = list(roots)
        for nid in roots:
            depths[nid] = 0
        while queue:
            node = queue.pop(0)
            for child in self.children_by_id.get(node, []):
                depths[child] = depths[node] + 1
                queue.append(child)
        return depths

    def _init_cones(self, cone_apertures: Optional[Union[Sequence[float], Mapping[Hashable, float], torch.Tensor]]) -> None:
        if cone_apertures is None:
            values = cone_half_aperture_ball(lorentz_to_poincare(self.weight.data), self.kappa)
        elif isinstance(cone_apertures, Mapping):
            values = torch.tensor(
                [float(cone_apertures[nid]) for nid in self.index_to_id],
                device=self.weight.device,
                dtype=self.weight.dtype,
            )
        else:
            values = torch.as_tensor(cone_apertures, device=self.weight.device, dtype=self.weight.dtype)
            if values.shape[0] != len(self.index_to_id):
                raise ValueError("cone_apertures length mismatch")
        self.register_buffer("cone_apertures", values.reshape(-1))

    def _coerce_indices(self, value: Union[int, Sequence[int], torch.Tensor, Hashable, Sequence[Hashable]]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            idx = value.to(device=self.weight.device)
            if idx.dtype != torch.long:
                idx = idx.long()
            return idx
        if isinstance(value, Hashable) and value in self.id_to_index:
            return torch.tensor([self.id_to_index[value]], dtype=torch.long, device=self.weight.device)
        if isinstance(value, int):
            return torch.tensor([value], dtype=torch.long, device=self.weight.device)
        if isinstance(value, (list, tuple)):
            if not value:
                return torch.empty(0, dtype=torch.long, device=self.weight.device)
            data: List[int] = []
            if all(isinstance(v, Hashable) and v in self.id_to_index for v in value):
                data = [self.id_to_index[cast(Hashable, v)] for v in value]
            elif all(isinstance(v, int) for v in value):
                data = [int(cast(int, v)) for v in value]
            else:
                raise TypeError("unable to interpret indices")
            return torch.tensor(data, dtype=torch.long, device=self.weight.device)
        raise TypeError("unable to interpret indices")

    def _coerce_index_matrix(self, value: Union[torch.Tensor, Sequence[Sequence[Union[int, Hashable]]]]) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            if value.dim() != 2:
                raise ValueError("candidate tensor must be 2D")
            return value.to(device=self.weight.device, dtype=torch.long)
        if isinstance(value, (list, tuple)):
            if not value:
                raise ValueError("candidate container empty")
            rows: List[List[int]] = []
            for row in value:
                if not isinstance(row, (list, tuple)):
                    raise TypeError("candidate rows must be sequences")
                if not row:
                    raise ValueError("candidate rows cannot be empty")
                if all(isinstance(v, Hashable) and v in self.id_to_index for v in row):
                    rows.append([self.id_to_index[cast(Hashable, v)] for v in row])
                    continue
                if all(isinstance(v, int) for v in row):
                    rows.append([int(cast(int, v)) for v in row])
                    continue
                raise TypeError("candidate rows must use uniform types")
            return torch.tensor(rows, dtype=torch.long, device=self.weight.device)
        raise TypeError("invalid candidate container")

    def ids_to_indices(self, ids: Union[Hashable, Sequence[Hashable], torch.Tensor]) -> torch.Tensor:
        if isinstance(ids, torch.Tensor):
            return ids.to(device=self.weight.device, dtype=torch.long)
        if isinstance(ids, Hashable):
            return torch.tensor([self.id_to_index[ids]], dtype=torch.long, device=self.weight.device)
        if isinstance(ids, (list, tuple)):
            return torch.tensor([self.id_to_index[i] for i in ids], dtype=torch.long, device=self.weight.device)
        raise TypeError("invalid id container")

    def indices_to_ids(self, indices: Union[int, Sequence[int], torch.Tensor]) -> List[Hashable]:
        if isinstance(indices, torch.Tensor):
            data = indices.detach().cpu().long().tolist()
        elif isinstance(indices, int):
            data = [indices]
        elif isinstance(indices, (list, tuple)):
            data = list(indices)
        else:
            raise TypeError("invalid indices container")
        return [self.index_to_id[int(i)] for i in data]

    def _convert(self, tensor: torch.Tensor, representation: Optional[str]) -> torch.Tensor:
        mode = (representation or self.default_representation).lower()
        if mode == "lorentz":
            return tensor
        if mode == "poincare":
            return lorentz_to_poincare(tensor)
        if mode == "euclidean":
            return tensor[..., 1:]
        if mode == "tangent":
            base = self.origin_vec.unsqueeze(0).expand_as(tensor)
            return lorentz_log_map(base, tensor)
        raise ValueError(f"unsupported representation {mode}")

    def forward(self, indices: Union[torch.Tensor, Sequence[int], int, Hashable, Sequence[Hashable]], representation: Optional[str] = None) -> torch.Tensor:
        idx = self._coerce_indices(indices)
        emb = self.weight.index_select(0, idx)
        return self._convert(emb, representation)

    def lookup(self, ids: Union[Hashable, Sequence[Hashable]], representation: Optional[str] = None) -> torch.Tensor:
        idx = self.ids_to_indices(ids)
        return self.forward(idx, representation)

    def get_all(self, representation: Optional[str] = None) -> torch.Tensor:
        return self._convert(self.weight, representation)

    def pairwise_distance(
        self,
        ids_a: Union[torch.Tensor, Sequence[int], Sequence[Hashable]],
        ids_b: Union[torch.Tensor, Sequence[int], Sequence[Hashable]],
    ) -> torch.Tensor:
        a = self.forward(ids_a, representation="lorentz")
        b = self.forward(ids_b, representation="lorentz")
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        return lorentz_distance(a, b).squeeze(-1)

    def ranking_loss(
        self,
        anchor_ids: Union[torch.Tensor, Sequence[int], Sequence[Hashable]],
        candidate_ids: Union[torch.Tensor, Sequence[Sequence[Union[int, Hashable]]]],
        target_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        anchors = self.forward(anchor_ids, representation="lorentz")
        candidate_idx = self._coerce_index_matrix(candidate_ids)
        if anchors.shape[0] != candidate_idx.shape[0]:
            raise ValueError("anchor and candidate batch must align")
        flat = candidate_idx.reshape(-1)
        candidates = self.weight.index_select(0, flat).reshape(
            candidate_idx.shape[0], candidate_idx.shape[1], -1
        )
        expanded = anchors.unsqueeze(1).expand_as(candidates)
        logits = -lorentz_distance(expanded, candidates).squeeze(-1)
        if target_positions is None:
            targets = torch.zeros(candidate_idx.shape[0], dtype=torch.long, device=self.weight.device)
        else:
            targets = target_positions.to(device=self.weight.device, dtype=torch.long)
            if targets.shape[0] != candidate_idx.shape[0]:
                raise ValueError("target positions must align with anchors")
        return F.cross_entropy(logits, targets)

    def cone_half_aperture(self, ids: Optional[Union[Hashable, Sequence[Hashable]]] = None) -> torch.Tensor:
        if ids is None:
            return self.cone_apertures
        idx = self.ids_to_indices(ids)
        return self.cone_apertures.index_select(0, idx)

    def refresh_cone_apertures(self, values: Optional[torch.Tensor] = None, kappa: Optional[float] = None) -> None:
        if values is not None:
            tensor = torch.as_tensor(values, device=self.weight.device, dtype=self.weight.dtype)
            if tensor.shape[0] != len(self.index_to_id):
                raise ValueError("cone_apertures length mismatch")
            self.cone_apertures = tensor
            return
        if kappa is not None:
            self.kappa = float(kappa)
        updated = cone_half_aperture_ball(lorentz_to_poincare(self.weight.data), self.kappa)
        self.cone_apertures = updated

    def cone_energy(
        self,
        parent_ids: Union[torch.Tensor, Sequence[int], Hashable, Sequence[Hashable]],
        child_ids: Optional[Union[torch.Tensor, Sequence[int], Hashable, Sequence[Hashable]]] = None,
        child_embeddings: Optional[torch.Tensor] = None,
        kappa: Optional[float] = None,
    ) -> torch.Tensor:
        parents = self.forward(parent_ids, representation="lorentz")
        if child_embeddings is None:
            if child_ids is None:
                raise ValueError("child identifiers required")
            children = self.forward(child_ids, representation="lorentz")
        else:
            children = torch.as_tensor(child_embeddings, device=self.weight.device, dtype=self.weight.dtype)
        if parents.shape != children.shape:
            raise ValueError("parent and child batches must align")
        energy = entailment_energy_lorentz(parents, children, kappa if kappa is not None else self.kappa)
        return energy.squeeze(-1)

    def in_cone(
        self,
        parent_ids: Union[torch.Tensor, Sequence[int], Hashable, Sequence[Hashable]],
        child_ids: Optional[Union[torch.Tensor, Sequence[int], Hashable, Sequence[Hashable]]] = None,
        child_embeddings: Optional[torch.Tensor] = None,
        kappa: Optional[float] = None,
        tol: float = 1e-6,
    ) -> torch.Tensor:
        energy = self.cone_energy(parent_ids, child_ids, child_embeddings, kappa)
        return energy <= tol

    def normalize(self) -> None:
        with torch.no_grad():
            normalized = normalize_lorentz(self.weight.data)
            self.weight.data.copy_(normalized)

    def embed_new_account(
        self,
        account_id: Hashable,
        parent_id: Hashable,
        radial_step: float = 1.0,
        noise_scale: float = 0.15,
        seed: Optional[int] = None,
        register: bool = True,
        aperture: Optional[float] = None,
    ) -> torch.Tensor:
        if parent_id not in self.id_to_index:
            raise KeyError(parent_id)
        parent_idx = self.id_to_index[parent_id]
        parent = self.weight[parent_idx : parent_idx + 1]
        generator = torch.Generator(device=self.weight.device)
        if seed is not None:
            generator.manual_seed(seed)
        axis = -lorentz_log_map(parent, self.origin_vec.unsqueeze(0))
        axis_norm = lorentz_norm(axis)
        if axis_norm.item() < EPS:
            axis = sample_tangent_directions(parent, 1, generator=generator)
        else:
            axis = axis / axis_norm
        noise = sample_tangent_directions(parent, 1, generator=generator)[0]
        direction = project_tangent(parent, axis + noise_scale * noise)
        direction = direction / torch.clamp(lorentz_norm(direction), min=EPS)
        tangent = radial_step * direction
        child = expmap(parent, tangent)
        energy = entailment_energy_lorentz(parent, child, self.kappa).squeeze(-1)
        if energy.item() > 0:
            child = expmap(parent, radial_step * axis)
        point = child[0]
        if register:
            self._register_new_account(account_id, point, parent_id, aperture)
        return point

    def _register_new_account(
        self,
        account_id: Hashable,
        embedding: torch.Tensor,
        parent_id: Optional[Hashable],
        aperture: Optional[float],
    ) -> None:
        if account_id in self.id_to_index:
            raise ValueError(f"{account_id} already registered")
        emb = normalize_lorentz(embedding.unsqueeze(0))
        new_weight = torch.cat([self.weight.data, emb], dim=0)
        requires_grad = self.weight.requires_grad
        self.weight = nn.Parameter(new_weight, requires_grad=requires_grad)
        self.index_to_id.append(account_id)
        idx = len(self.index_to_id) - 1
        self.id_to_index[account_id] = idx
        self.parent_by_id[account_id] = parent_id
        if parent_id is not None:
            self.children_by_id.setdefault(parent_id, []).append(account_id)
        self.children_by_id.setdefault(account_id, [])
        parent_depth = self.depths.get(parent_id, -1)
        self.depths[account_id] = parent_depth + 1
        if aperture is None:
            aperture_tensor = cone_half_aperture_ball(lorentz_to_poincare(emb), self.kappa)[0]
        else:
            aperture_tensor = torch.tensor(float(aperture), device=self.weight.device, dtype=self.weight.dtype)
        new_cones = torch.cat([self.cone_apertures, aperture_tensor.reshape(1)], dim=0)
        self.cone_apertures = new_cones

    @classmethod
    def from_tree(
        cls,
        dim: int,
        edges: Iterable[Tuple[Hashable, Hashable]],
        node_ids: Optional[Sequence[Hashable]] = None,
        root: Hashable = "root",
        edge_length: float = 1.0,
        seed: int = 0,
        freeze: bool = True,
        default_representation: str = "lorentz",
        kappa: float = 0.95,
        device=DEVICE,
        dtype=DTYPE,
    ) -> "AccountEncoder":
        if dim < 2:
            raise ValueError("dim must be >=2")
        children = build_children(edges)
        parents = build_parent_map(edges)
        parents.setdefault(root, None)
        children.setdefault(root, [])
        emb_map = cls._encode_from_tree(children, dim, edge_length, seed, device, dtype, root)
        if node_ids is None:
            nodes = list(emb_map.keys())
        else:
            nodes = list(node_ids)
        missing = [n for n in nodes if n not in emb_map]
        if missing:
            raise ValueError(f"missing embeddings for {missing[:5]}")
        embeddings = torch.stack([emb_map[n] for n in nodes], dim=0)
        parent_subset = {nid: parents.get(nid) for nid in nodes}
        return cls(
            embeddings=embeddings,
            node_ids=nodes,
            parents=parent_subset,
            freeze=freeze,
            default_representation=default_representation,
            kappa=kappa,
            device=device,
            dtype=dtype,
        )

    @staticmethod
    def _encode_from_tree(
        children: Mapping[Hashable, Iterable[Hashable]],
        dim: int,
        edge_length: float,
        seed: int,
        device,
        dtype,
        root: Hashable,
    ) -> Dict[Hashable, torch.Tensor]:
        from collections import deque

        emb: Dict[Hashable, torch.Tensor] = {}
        emb[root] = lorentz_origin(dim, device=device, dtype=dtype)
        queue = deque([root])
        while queue:
            parent = queue.popleft()
            kids = list(children.get(parent, []))
            if not kids:
                continue
            base = emb[parent].unsqueeze(0)
            generator = torch.Generator(device=device)
            generator.manual_seed((seed + hash(parent)) % (2**31 - 1))
            dirs = sample_tangent_directions(base, len(kids), generator=generator)
            for child, direction in zip(kids, dirs):
                tangent = direction * edge_length
                point = expmap(base, tangent.unsqueeze(0))[0]
                emb[child] = point
                queue.append(child)
        return emb
