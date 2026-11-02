"""Table 1: Bi-HYCO (physical and synthetic) + XPINN; three seeds per noise level.

Run this complete file in Jupyter or use the equivalent .ipynb notebook.
Dependencies: numpy, scipy, pandas, torch. No previous files are needed.
The final iterate is always evaluated. Each execution recomputes everything.
"""

# In Jupyter, if needed: %pip install numpy scipy pandas torch
from pathlib import Path
from copy import deepcopy
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import scipy
import torch
from torch import nn
from torch.nn import functional as F


# 1. Complete experiment configuration.
BASE_SEED = 1240
NOISE_LEVELS = [0.00, 0.05, 0.10, 0.15, 0.20]
SEEDS_PER_LEVEL = 3                  # Also three when eta = 0.
A_TRUE = np.array([2.0, 1.0])
A_INIT = np.array([0.5, 0.5])
COEFFICIENT_FLOOR = 1e-4

HYCO_ROUNDS = 20000
HYCO_LOCAL_STEPS = 2
HYCO_WIDTH, HYCO_DEPTH = 96, 4
HYCO_COEFFICIENT_LR, HYCO_NETWORK_LR = 5e-3, 8e-4
ALPHA_PHY, ALPHA_SYN = 0.3, 0.2
OMEGA_PHY, OMEGA_SYN = 0.5, 0.5
UNIT_WEIGHTS = (0.5, 0.5)
BETA_PHY, BETA_SYN = 1e-8, 1e-8
LAMBDA_INTERACTION = 1.0
HYCO_CLIP_NORM = 5.0
LR_DECAY, LR_DECAY_EVERY = 0.9995, 100
N_INTERACTION = 1000
NEAR_INTERFACE_FRACTION = 0.70
INTERFACE_HALF_WIDTH = 0.04          # Explicit choice: radial band +/-0.04.

XPINN_EPOCHS = 30000
XPINN_WIDTH, XPINN_DEPTH = 80, 4
XPINN_LR = 1e-3
POOL_INTERIOR, POOL_BOUNDARY, POOL_INTERFACE = 8192, 1024, 1024
BATCH_INTERIOR, BATCH_BOUNDARY, BATCH_INTERFACE = 512, 256, 256
PRINT_EVERY = 2000

RESULTS_DIR = Path("Elliptic_Table1_3seeds")
REFERENCE_FILE = Path("uref-elliptic.npz")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float32


def tensor(x):
    return torch.as_tensor(np.asarray(x), dtype=DTYPE, device=DEVICE)


def seed_training(seed):
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def synchronize_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


class MLP(nn.Module):
    def __init__(self, width, depth):
        super().__init__()
        layers = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth-1):
            layers.extend([nn.Linear(width, width), nn.Tanh()])
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x):
        return self.net(x)


# 2. Geometry, fixed finite elements, and fast differentiable solves.
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
from scipy.linalg import eigh
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator


def outer_radius(theta):
    return 1.0 + 0.12*np.cos(3.0*theta) + 0.08*np.sin(5.0*theta)


def interface_radius(theta):
    return 0.42*(1.0 + 0.22*np.cos(5.0*theta))


def inside_outer_np(pts, tol=1e-12):
    pts = np.asarray(pts)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    return np.hypot(pts[:, 0], pts[:, 1]) <= outer_radius(theta) + tol


def inside_inner_np(pts, tol=1e-12):
    pts = np.asarray(pts)
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    return np.hypot(pts[:, 0], pts[:, 1]) <= interface_radius(theta) + tol


def outer_boundary_points(n):
    theta = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    radius = outer_radius(theta)
    return np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))


def interface_points_np(n):
    theta = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    radius = interface_radius(theta)
    return np.column_stack((radius*np.cos(theta), radius*np.sin(theta)))


def interface_points_and_normals_np(n, eps=1e-5):
    theta = np.linspace(0.0, 2.0*np.pi, n, endpoint=False)
    def gamma(t):
        radius = interface_radius(t)
        return np.column_stack((radius*np.cos(t), radius*np.sin(t)))
    tangent = (gamma(theta+eps)-gamma(theta-eps))/(2.0*eps)
    normal = np.column_stack((tangent[:, 1], -tangent[:, 0]))
    normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-14
    pts = gamma(theta)
    normal[np.sum(normal*pts, axis=1) < 0] *= -1.0
    return pts, normal


def sample_uniform_in_outer(n, seed=None):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        cand = rng.uniform(-1.25, 1.25, size=(4*n, 2))
        out.extend(list(cand[inside_outer_np(cand)]))
    return np.asarray(out[:n])


def sample_uniform_in_inner(n, seed=None):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        cand = rng.uniform(-0.65, 0.65, size=(4*n, 2))
        out.extend(list(cand[inside_inner_np(cand)]))
    return np.asarray(out[:n])


def sample_uniform_in_outer_shell(n, seed=None):
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        cand = rng.uniform(-1.25, 1.25, size=(5*n, 2))
        selected = inside_outer_np(cand) & ~inside_inner_np(cand)
        out.extend(list(cand[selected]))
    return np.asarray(out[:n])


def build_unstructured_mesh(n_interior, n_outer_boundary, n_interface, seed):
    outer = outer_boundary_points(n_outer_boundary)
    iface = interface_points_np(n_interface)
    interior = sample_uniform_in_outer(n_interior, seed=seed+10)
    pts = np.unique(np.round(np.vstack((outer, iface, interior)), 12), axis=0)
    raw = Delaunay(pts).simplices.copy()
    centroids = pts[raw].mean(axis=1)
    p0, p1, p2 = (pts[raw[:, j]] for j in (0, 1, 2))
    area = 0.5*np.abs((p1[:, 0]-p0[:, 0])*(p2[:, 1]-p0[:, 1])
                      - (p1[:, 1]-p0[:, 1])*(p2[:, 0]-p0[:, 0]))
    triangles = raw[inside_outer_np(centroids) & (area > 1e-8)]
    theta = np.arctan2(pts[:, 1], pts[:, 0])
    boundary = np.flatnonzero(np.abs(np.hypot(pts[:, 0], pts[:, 1])
                                    - outer_radius(theta)) < 5e-6)
    return pts, triangles, boundary


def source_np(x, y):
    return 1.0 + 0.5*np.sin(np.pi*x)*np.cos(np.pi*y)


def make_interpolator(pts, values):
    # Preserve the supplied notebook's interpolation, including its fallback.
    linear = LinearNDInterpolator(pts, values, fill_value=np.nan)
    nearest = NearestNDInterpolator(pts, values)
    def interpolate(xy):
        xy = np.asarray(xy)
        values = np.asarray(linear(xy))
        missing = np.isnan(values)
        if np.any(missing):
            values[missing] = nearest(xy[missing])
        return values
    return interpolate


class FixedFEM:
    def __init__(self, points, triangles, boundary):
        self.points, self.triangles, self.boundary = points, triangles, boundary
        self.free = np.setdiff1d(np.arange(len(points)), boundary)
        p = points[triangles]
        x1, y1 = p[:, 0, 0], p[:, 0, 1]
        x2, y2 = p[:, 1, 0], p[:, 1, 1]
        x3, y3 = p[:, 2, 0], p[:, 2, 1]
        area = 0.5*np.abs((x2-x1)*(y3-y1)-(x3-x1)*(y2-y1))
        b = np.stack((y2-y3, y3-y1, y1-y2), axis=1)/(2*area[:, None])
        c = np.stack((x3-x2, x1-x3, x2-x1), axis=1)/(2*area[:, None])
        local = area[:, None, None]*(b[:, :, None]*b[:, None, :]
                                    + c[:, :, None]*c[:, None, :])
        centroids = p.mean(axis=1)
        inner = inside_inner_np(centroids)
        rows = np.repeat(triangles, 3, axis=1).ravel()
        cols = np.tile(triangles, (1, 3)).ravel()
        shape = (len(points), len(points))
        def assemble(mask):
            values = (local*mask[:, None, None]).ravel()
            return coo_matrix((values, (rows, cols)), shape=shape).tocsr()
        self.K1, self.K2 = assemble(inner), assemble(~inner)
        self.rhs, self.mass = np.zeros(len(points)), np.zeros(len(points))
        load = source_np(centroids[:, 0], centroids[:, 1])*area/3
        for j in range(3):
            np.add.at(self.rhs, triangles[:, j], load)
            np.add.at(self.mass, triangles[:, j], area/3)
        self.K1_free = self.K1[self.free][:, self.free]
        self.K2_free = self.K2[self.free][:, self.free]
        self._spectral = None

    def solve_numpy(self, coefficients):
        a1, a2 = coefficients
        state = np.zeros(len(self.points))
        matrix = a1*self.K1_free + a2*self.K2_free
        state[self.free] = spsolve(matrix, self.rhs[self.free])
        return state

    def prepare_torch(self, device):
        # Full basis: an algebraic diagonalization, with no truncation.
        if self._spectral is None:
            k1 = self.K1_free.toarray()
            k0 = k1 + self.K2_free.toarray()
            eigenvalues, vectors = eigh(k1, k0, check_finite=False)
            self._spectral = eigenvalues, vectors, vectors.T @ self.rhs[self.free]
        return SpectralFEM(len(self.points), self.free, *self._spectral, device)


class SpectralFEM(nn.Module):
    def __init__(self, n_nodes, free, eigenvalues, vectors, load, device):
        super().__init__()
        self.n_nodes = n_nodes
        self.register_buffer("free", torch.as_tensor(free, dtype=torch.long, device=device))
        for name, value in (("d", eigenvalues), ("V", vectors), ("load", load)):
            self.register_buffer(name, torch.as_tensor(value, dtype=torch.float32, device=device))

    def forward(self, coefficients):
        a1, a2 = coefficients.unbind()
        free_state = self.V @ (self.load/(a2 + (a1-a2)*self.d))
        return free_state.new_zeros(self.n_nodes).index_copy(0, self.free, free_state)


class NodalInterpolator:
    def __init__(self, points):
        # As LinearNDInterpolator, use the full point-cloud Delaunay tessellation.
        self.triangulation = Delaunay(points)
        self.nearest = cKDTree(points)

    def indices_weights(self, xy):
        xy = np.atleast_2d(np.asarray(xy))
        tri = self.triangulation
        simplex = tri.find_simplex(xy)
        valid = simplex >= 0
        indices = np.empty((len(xy), 3), dtype=np.int64)
        weights = np.zeros((len(xy), 3))
        transforms = tri.transform[simplex[valid]]
        bary = np.einsum("nij,nj->ni", transforms[:, :2],
                         xy[valid]-transforms[:, 2])
        indices[valid] = tri.simplices[simplex[valid]]
        weights[valid, :2] = bary
        weights[valid, 2] = 1.0-bary.sum(axis=1)
        if np.any(~valid):
            nearest = self.nearest.query(xy[~valid])[1]
            indices[~valid] = nearest[:, None]
            weights[~valid, 0] = 1.0
        return indices, weights

    def torch_map(self, xy, device):
        return TorchNodalMap(*self.indices_weights(xy), device)


class TorchNodalMap(nn.Module):
    def __init__(self, indices, weights, device):
        super().__init__()
        self.register_buffer("indices", torch.as_tensor(indices, dtype=torch.long, device=device))
        self.register_buffer("weights", torch.as_tensor(weights, dtype=torch.float32, device=device))

    def forward(self, nodal_values):
        return (nodal_values[self.indices]*self.weights).sum(dim=-1)


def relative_l2_error_on_mesh(u_pred, u_ref, mass):
    return float(np.sqrt(np.sum(mass*(u_pred-u_ref)**2))
                 /(np.sqrt(np.sum(mass*u_ref**2))+1e-14))


def state_errors(u_pred, u_ref, fem):
    inner = inside_inner_np(fem.points)
    masks = (np.ones(len(inner), dtype=bool), inner, ~inner)
    names = ("global_error", "inner_error", "outer_error")
    return {name: relative_l2_error_on_mesh(u_pred[mask], u_ref[mask], fem.mass[mask])
            for name, mask in zip(names, masks)}


# 3. Bi-HYCO: both units update their local coefficients and network.
class BiHYCO(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = MLP(HYCO_WIDTH, HYCO_DEPTH)
        self.raw_coefficients = nn.Parameter(
            torch.log(torch.expm1(torch.tensor(A_INIT-COEFFICIENT_FLOOR,
                                             dtype=DTYPE))))

    def coefficients(self):
        return F.softplus(self.raw_coefficients) + COEFFICIENT_FLOOR


def interaction_points(rng):
    n_near = int(NEAR_INTERFACE_FRACTION*N_INTERACTION)
    angle = rng.uniform(0.0, 2*np.pi, n_near)
    radius = interface_radius(angle) + rng.uniform(
        -INTERFACE_HALF_WIDTH, INTERFACE_HALF_WIDTH, n_near)
    near = np.column_stack((radius*np.cos(angle), radius*np.sin(angle)))
    uniform = sample_uniform_in_outer(N_INTERACTION-n_near, seed=rng)
    return np.vstack((near, uniform))


def train_bihyco(problem, y_phy, y_syn, seed):
    seed_training(seed)
    local = [BiHYCO().to(DEVICE)]
    local.append(deepcopy(local[0]))
    optimizers = [torch.optim.Adam([
        {"params": [unit.raw_coefficients], "lr": HYCO_COEFFICIENT_LR},
        {"params": unit.net.parameters(), "lr": HYCO_NETWORK_LR},
    ]) for unit in local]
    # Adam moments persist PER UNIT; only parameters are aggregated.
    rng = np.random.default_rng(seed+22001)
    y1, y2 = tensor(y_phy), tensor(y_syn)
    w1, w2 = UNIT_WEIGHTS
    synchronize_device()
    start = time.perf_counter()

    for round_number in range(1, HYCO_ROUNDS+1):
        z = interaction_points(rng)
        z_t = tensor(z)
        interaction_map = problem["interpolator"].torch_map(z, DEVICE)
        for unit_index, (unit, optimizer) in enumerate(zip(local, optimizers)):
            for _ in range(HYCO_LOCAL_STEPS):
                optimizer.zero_grad(set_to_none=True)
                a = unit.coefficients()
                u = problem["physical"](a)
                agreement = (interaction_map(u)-unit.net(z_t).flatten()).square().mean()
                if unit_index == 0:
                    data = (problem["observation_map"](u)-y1).square().mean()
                    regularizer = BETA_PHY*a.square().sum()
                    loss = OMEGA_PHY*ALPHA_PHY*(data+regularizer)
                else:
                    data = (unit.net(problem["X_syn_t"]).flatten()-y2).square().mean()
                    regularizer = 0.5*BETA_SYN*sum(
                        p.square().sum() for p in unit.net.parameters())
                    loss = OMEGA_SYN*ALPHA_SYN*(data+regularizer)
                loss = loss + LAMBDA_INTERACTION*agreement
                loss.backward()                # Both blocks, evaluated at the same local iterate.
                torch.nn.utils.clip_grad_norm_(unit.net.parameters(), HYCO_CLIP_NORM)
                optimizer.step()

        # Average physical a1,a2, then return to the unconstrained softplus coordinates.
        with torch.no_grad():
            averaged_a = w1*local[0].coefficients()+w2*local[1].coefficients()
            positive = averaged_a-COEFFICIENT_FLOOR
            averaged_raw = positive + torch.log(-torch.expm1(-positive))
            for unit in local:
                unit.raw_coefficients.copy_(averaged_raw)
            for p1, p2 in zip(local[0].net.parameters(), local[1].net.parameters()):
                averaged = w1*p1+w2*p2
                p1.copy_(averaged)
                p2.copy_(averaged)

        if round_number % LR_DECAY_EVERY == 0:
            for optimizer in optimizers:
                for group in optimizer.param_groups:
                    group["lr"] *= LR_DECAY
        if round_number % PRINT_EVERY == 0 or round_number == HYCO_ROUNDS:
            a_print = local[0].coefficients().detach().cpu().numpy()
            print(f"  Bi-HYCO {round_number:5d}/{HYCO_ROUNDS}: "
                  f"a=({a_print[0]:.6f}, {a_print[1]:.6f})", flush=True)

    synchronize_device()
    elapsed = time.perf_counter()-start
    return local[0], elapsed


# 4. XPINN: two local data holders and shared positive coefficients.
# This is a single-process simulation of local computations and coordination.
class XPINN(nn.Module):
    def __init__(self):
        super().__init__()
        self.inner = MLP(width=XPINN_WIDTH, depth=XPINN_DEPTH)
        self.outer = MLP(width=XPINN_WIDTH, depth=XPINN_DEPTH)
        target = torch.as_tensor(np.asarray(A_INIT)-1e-4, dtype=DTYPE)
        self.raw_coefficients = nn.Parameter(torch.log(torch.expm1(target)))

    def coefficients(self):
        return F.softplus(self.raw_coefficients) + 1e-4


def laplacian_of_network(network, x):
    x = x.detach().requires_grad_(True)
    u = network(x)
    grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(grad[:, 0].sum(), x, create_graph=True)[0][:, 0:1]
    u_yy = torch.autograd.grad(grad[:, 1].sum(), x, create_graph=True)[0][:, 1:2]
    return u_xx + u_yy


def normal_gradient(network, x, normals):
    x = x.detach().requires_grad_(True)
    u = network(x)
    grad = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    return u, (grad*normals).sum(dim=1, keepdim=True)


def source_torch(x):
    return 1.0 + 0.5*torch.sin(np.pi*x[:, 0:1])*torch.cos(np.pi*x[:, 1:2])


class XPINNLocalUnit:
    """A unit stores only its own observations and computes its own gradients."""
    def __init__(self, network, coordinates, values, interior_pool, boundary_pool=None):
        self.network = network
        self.coordinates = coordinates
        self.values = tensor(np.asarray(values).reshape(-1, 1))
        self.interior_pool = interior_pool
        self.boundary_pool = boundary_pool

    def backward_local(self, coefficient, interior_indices, boundary_indices=None):
        data_loss = (self.network(self.coordinates)-self.values).square().mean()
        x = self.interior_pool[interior_indices]
        residual = -coefficient*laplacian_of_network(self.network, x)-source_torch(x)
        # Each unit contributes half of the original two-subdomain data/PDE loss.
        loss = 5.0*data_loss + 0.5*residual.square().mean()
        if self.boundary_pool is not None:
            boundary = self.network(self.boundary_pool[boundary_indices])
            loss = loss + 10.0*boundary.square().mean()
        loss.backward()
        return loss.detach()


def train_xpinn(problem, y_phy, y_syn, seed):
    seed_training(seed)
    inner_pool = tensor(sample_uniform_in_inner(POOL_INTERIOR, seed=seed+11001))
    outer_pool = tensor(sample_uniform_in_outer_shell(POOL_INTERIOR, seed=seed+11002))
    boundary_pool = tensor(outer_boundary_points(POOL_BOUNDARY))
    iface, normal = interface_points_and_normals_np(POOL_INTERFACE)
    interface_pool, normal_pool = tensor(iface), tensor(normal)
    model = XPINN().to(device=DEVICE, dtype=DTYPE)
    unit1 = XPINNLocalUnit(model.inner, problem['X_phy_t'], y_phy, inner_pool)
    unit2 = XPINNLocalUnit(model.outer, problem['X_syn_t'], y_syn, outer_pool, boundary_pool)
    optimizer = torch.optim.Adam(model.parameters(), lr=XPINN_LR)
    synchronize_device()
    start = time.perf_counter()

    for epoch in range(1, XPINN_EPOCHS+1):
        optimizer.zero_grad(set_to_none=True)
        # Preserve the original random draw order for all four minibatches.
        idx_inner = torch.randint(len(inner_pool), (BATCH_INTERIOR,), device=DEVICE)
        idx_outer = torch.randint(len(outer_pool), (BATCH_INTERIOR,), device=DEVICE)
        idx_boundary = torch.randint(len(boundary_pool), (BATCH_BOUNDARY,), device=DEVICE)
        idx_interface = torch.randint(len(interface_pool), (BATCH_INTERFACE,), device=DEVICE)

        # No raw observation pooling: each unit backpropagates only its local loss.
        # Fresh coefficient graphs allow independent local backward passes.
        local1 = unit1.backward_local(model.coefficients()[0], idx_inner)
        local2 = unit2.backward_local(model.coefficients()[1], idx_outer, idx_boundary)

        # The coordinator couples unlabeled interface predictions and their fluxes.
        xf, nf = interface_pool[idx_interface], normal_pool[idx_interface]
        u1, dn1 = normal_gradient(model.inner, xf, nf)
        u2, dn2 = normal_gradient(model.outer, xf, nf)
        a1, a2 = model.coefficients().unbind()
        state_loss = (u1-u2).square().mean()
        flux_loss = (a1*dn1-a2*dn2).square().mean()
        interface_loss = 10.0*state_loss + 2.0*flux_loss
        interface_loss.backward()
        # One shared Adam step uses the sum of all contributions, as in the source.
        optimizer.step()

        if epoch % PRINT_EVERY == 0 or epoch == XPINN_EPOCHS:
            a = model.coefficients().detach().cpu().numpy()
            total_loss = local1 + local2 + interface_loss.detach()
            print(f'  XPINN {epoch}/{XPINN_EPOCHS}: '
                  f'a=({a[0]:.6f}, {a[1]:.6f}), loss={total_loss.item():.3e}')

    synchronize_device()
    elapsed = time.perf_counter()-start
    return model, elapsed


def predict_xpinn(model, xy):
    xy = np.asarray(xy)
    inner = inside_inner_np(xy)
    result = np.empty(len(xy), dtype=np.float64)
    model.eval()
    with torch.no_grad():
        result[inner] = model.inner(tensor(xy[inner])).cpu().numpy().reshape(-1)
        result[~inner] = model.outer(tensor(xy[~inner])).cpu().numpy().reshape(-1)
    return result



# Imported routines are based on the supplied original elliptic notebook.
