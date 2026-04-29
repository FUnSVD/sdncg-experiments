#!/usr/bin/env python3
from __future__ import annotations

"""Clean final single-file experiment runner for the rebuilt S-DNCG project.

Scope of this file
------------------
- dynamic stochastic congestion-game environment
- risk-sensitive effective cost: mean + 0.5 * alpha * variance
- shared information-state encoder
- unified empirical CE evaluator
- shared risk-stratified DeepCFR-style teacher
- PPO baseline, teacher-only DeepCFR, DeepCFR-guided PPO, and KL ablation
- paper-ready figures
- minimal logging, CSV/JSON saving, and visualization

This final runner keeps one centralized parameter section (FINAL_DEFAULTS)
and removes legacy tabular/regret-PG/older main runners.
"""

import argparse
import csv
import json
import math
import os
import random
from bisect import bisect_right
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplconfig")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Reduce CPU overhead from multithreaded BLAS on tiny policy/value forwards.
torch.set_num_threads(1)
if hasattr(torch, "set_num_interop_threads"):
    torch.set_num_interop_threads(1)
# -----------------------------------------------------------------------------
# Final experiment defaults
# -----------------------------------------------------------------------------
# All default experiment settings are centralized here.  The dataclass defaults
# above are only safe fallbacks; the executable runner below builds all configs
# from this dictionary through the CLI parser.
FINAL_DEFAULTS = {
    # output / sweep
    "output_dir": "./N5_N20_N50_sizes_5000_stabilized_355455_outputs",
    "seeds": [42, 123, 256],
    "modes": "ppo,deepcfr,deepcfr_rl_no_distill,deepcfr_rl_distill",
    "graph_sizes": "small,medium,large",
    "small_widths": "3,3,3",
    "medium_widths": "4,4,4",
    "large_widths": "5,5,5",
    "topology": "main_path",
    "base_capacity": 5,
    "num_agents": 20,
    "agent_counts": "5,20,50",
    "horizon": 5,
    "risk_alphas": "0.0,0.2,0.6,1.0",
    "reward_mode": "effective",

    # PPO student
    "actor_lr": 3e-4,
    "critic_lr": 1e-3,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_ratio": 0.2,
    "entropy_coef": 0.01,
    "value_coef": 0.5,
    "max_grad_norm": 0.5,
    "hidden_dim": 128,
    "node_embed_dim": 16,
    "ppo_epochs": 4,
    "minibatch_size": 128,
    "episodes_per_update": 1,
    "total_updates": 5000,
    "eval_every": 500,
    "eval_episodes": 3,

    # Budgeted shared DeepCFR-style teacher.
    # This slightly stronger budget is a compromise for MacBook runs:
    # stronger than the 4/6/batch=48 diagnostic setting, but still lighter
    # than the previous 6/8/lookahead=2 heavy teacher. The teacher is trained
    # once per graph/N/seed and reused by DeepCFR, guided PPO, and KL ablation.
    "teacher_eval_policy": "average",
    "deep_teacher_hidden_dim": 128,
    "deep_teacher_node_embed_dim": 16,
    "deep_teacher_lr": 8e-4,
    "deep_teacher_buffer_capacity": 50000,
    "deep_teacher_train_steps": 5,
    "deep_teacher_strategy_train_steps": 7,
    "deep_teacher_batch_size": 64,
    "deep_teacher_min_buffer": 32,
    "deep_teacher_retrain_from_scratch": False,
    "deep_teacher_target_lookahead_steps": 1,
    "deep_teacher_target_policy_mode": "current",
    "deep_teacher_target_future_load_floor": 1.0,

    # Teacher-student guidance and KL ablation.
    "hybrid_teacher_warmup_updates": 0,
    "hybrid_teacher_policy_mode": "average",
    "hybrid_beta_start": 0.65,
    "hybrid_beta_end": 0.00,
    "hybrid_beta_floor": 0.03,
    "hybrid_mix_conf_threshold": 0.03,
    "hybrid_mix_conf_scale": 0.20,
    "hybrid_mix_js_threshold": 0.001,
    "hybrid_mix_js_scale": 0.02,
    "hybrid_mix_min_js_gate": 0.25,
    "distill_lambda_max": 0.02,
    "distill_temperature": 2.5,
    "distill_conf_threshold": 0.05,
    "distill_conf_scale": 0.20,
    "distill_gap_threshold": 0.002,
    "distill_gap_scale": 0.02,
    "distill_cf_advantage_threshold": 0.0,
    "distill_cf_advantage_scale": 0.005,
    "distill_ramp_updates": 600,
    "distill_min_branching_actions": 2,
    "adaptive_teacher_snapshot": True,
    "teacher_freeze_min_updates": 1000,
    "teacher_freeze_max_updates": 5000,
    "teacher_ready_ce_ucb_threshold": 0.0068,
    "teacher_ready_stability_window": 3,
    "teacher_ready_stability_delta": 0.0010,
    "teacher_ready_min_confidence": 0.03,
    "distill_decay_start_fraction": 0.30,

    # Evaluation / reproducibility
    "epsilon_ce": 0.01,
    "stability_threshold": 0.005,
    "trailing_window": 200,
    "device": "cpu",
}


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RunLogger:
    def __init__(self, output_dir: Path, verbose: bool = False) -> None:
        self.output_dir = output_dir
        self.verbose = verbose
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        if self.verbose:
            print(message)

    def summary(self, message: str) -> None:
        print(message)

    def write_json(self, relative_path: str, payload: Mapping) -> Path:
        path = self.output_dir / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return path


# -----------------------------------------------------------------------------
# Core data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeSpec:
    edge_id: int
    src: str
    dst: str
    capacity: int
    base_cost: float
    congestion_coef: float
    congestion_quad: float = 0.0
    variance_base: float = 0.05
    variance_coef: float = 0.02
    incident_prob: float = 0.0
    incident_delay: float = 0.0
    failure_mode: str = "normal"  # {"normal", "stay", "fallback"}
    fallback_edge_id: int | None = None

    def mean_cost(self, load: int) -> float:
        return float(self.base_cost + self.congestion_coef * load + self.congestion_quad * load * load)

    def variance(self, load: int) -> float:
        return float(max(1e-8, self.variance_base + self.variance_coef * load))


@dataclass(frozen=True)
class AgentSpec:
    agent_id: int
    risk_alpha: float = 0.0
    type_name: str = "default"


@dataclass(frozen=True)
class PublicStateSnapshot:
    time_step: int
    node_counts: Mapping[str, int]
    last_edge_loads: Mapping[int, int]
    last_incidents: Mapping[int, bool]

    def last_load_ratio(self, edge_id: int, capacity: int) -> float:
        return float(self.last_edge_loads.get(edge_id, 0) / max(1, capacity))


@dataclass(frozen=True)
class LocalObservation:
    agent_id: int
    time_step: int
    current_node: str
    risk_alpha: float
    feasible_action_ids: Tuple[int, ...]
    edge_feature_map: Mapping[int, np.ndarray]
    public_snapshot: PublicStateSnapshot
    done: bool = False

    def to_feature_matrix(self) -> Tuple[Tuple[int, ...], np.ndarray]:
        if not self.feasible_action_ids:
            return (), np.zeros((0, 0), dtype=np.float32)
        mats = [np.asarray(self.edge_feature_map[eid], dtype=np.float32) for eid in self.feasible_action_ids]
        return self.feasible_action_ids, np.stack(mats, axis=0)


@dataclass(frozen=True)
class InfoStateKey:
    time_bin: int
    node_id: str
    risk_bin: int
    occupancy_bin: int
    recent_load_bin: int
    incident_bin: int

    def as_tuple(self) -> Tuple[int, str, int, int, int, int]:
        return (
            self.time_bin,
            self.node_id,
            self.risk_bin,
            self.occupancy_bin,
            self.recent_load_bin,
            self.incident_bin,
        )


@dataclass(frozen=True)
class StepAgentOutcome:
    agent_id: int
    chosen_edge_id: int | None
    current_node: str
    next_node: str
    load_on_chosen_edge: int
    mean_cost: float
    variance_cost: float
    effective_cost: float
    realized_cost: float
    incident_triggered: bool
    terminated: bool


@dataclass(frozen=True)
class EnvStepResult:
    observations: Mapping[int, LocalObservation]
    rewards: Mapping[int, float]
    dones: Mapping[int, bool]
    outcomes: Mapping[int, StepAgentOutcome]
    public_snapshot_before: PublicStateSnapshot
    positions_before: Mapping[int, str]
    joint_actions: Mapping[int, int]


@dataclass(frozen=True)
class DecisionRecord:
    agent_id: int
    time_step: int
    current_node: str
    action_id: int | None
    feasible_action_ids: Tuple[int, ...]
    risk_alpha: float
    info_state_key: InfoStateKey | None
    public_snapshot: PublicStateSnapshot
    positions_before: Mapping[int, str]
    joint_actions: Mapping[int, int]
    mean_cost: float
    variance_cost: float
    effective_cost: float
    realized_cost: float
    load_on_action: int


@dataclass
class PPOSample:
    current_node_idx: int
    feasible_action_ids: Tuple[int, ...]
    edge_features: np.ndarray
    action_index: int
    old_logprob: float
    value: float
    reward: float
    done: bool
    agent_id: int
    time_step: int
    info_state_key: InfoStateKey | None
    advantage: float = 0.0
    return_: float = 0.0


# -----------------------------------------------------------------------------
# Configs
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class EnvConfig:
    num_agents: int = 20
    horizon: int = 5
    cost_clip: float = 100.0
    reward_mode: str = "effective"  # {"effective", "realized"}
    shared_edge_incident: bool = True
    normal_std_floor: float = 1e-3
    random_seed: int = 42


@dataclass(frozen=True)
class EncoderConfig:
    risk_bin_edges: Sequence[float] = (0.0, 0.2, 0.6, 1.0)
    occupancy_bin_edges: Sequence[float] = (0.2, 0.5, 0.8)
    recent_load_bin_edges: Sequence[float] = (0.2, 0.5, 0.8)
    time_bin_size: int = 1


@dataclass(frozen=True)
class PPOConfig:
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dim: int = 128
    node_embed_dim: int = 16
    ppo_epochs: int = 4
    minibatch_size: int = 128
    episodes_per_update: int = 8
    total_updates: int = 120
    eval_every: int = 10
    eval_episodes: int = 8


@dataclass(frozen=True)
class EvalConfig:
    epsilon: float = 0.01
    stability_threshold: float = 0.005
    trailing_window: int = 200


# -----------------------------------------------------------------------------
# Graph builder and info-state encoder
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GraphSpec:
    source: str
    target: str
    nodes: Tuple[str, ...]
    layer_nodes: Tuple[Tuple[str, ...], ...]
    edges: Mapping[int, EdgeSpec]
    outgoing: Mapping[str, Tuple[int, ...]]


class LayeredGraphBuilder:
    @staticmethod
    def build(widths: Sequence[int], topology: str = "fully_connected", base_capacity: int = 6) -> GraphSpec:
        if not widths:
            raise ValueError("widths must contain at least one intermediate layer")
        if topology not in {"fully_connected", "adjacent_only", "main_path"}:
            raise ValueError(f"Unsupported topology: {topology}")

        source = "src"
        target = "tgt"
        layers: List[Tuple[str, ...]] = []
        for layer_idx, width in enumerate(widths):
            layers.append(tuple(f"L{layer_idx}_N{j}" for j in range(width)))

        all_nodes = (source,) + tuple(node for layer in layers for node in layer) + (target,)
        outgoing: Dict[str, List[int]] = {node: [] for node in all_nodes}
        edges: Dict[int, EdgeSpec] = {}
        edge_id = 0

        def connect_pairs(layer_a: Sequence[str], layer_b: Sequence[str]) -> Iterable[Tuple[int, int]]:
            width_b = len(layer_b)
            center_b = width_b // 2
            for i in range(len(layer_a)):
                if topology == "fully_connected":
                    js = range(width_b)
                elif topology == "adjacent_only":
                    js = [j for j in range(width_b) if abs(j - min(i, width_b - 1)) <= 1]
                else:
                    neighbors = {min(i, width_b - 1), center_b}
                    if i > 0:
                        neighbors.add(min(i - 1, width_b - 1))
                    if i + 1 < width_b:
                        neighbors.add(i + 1)
                    js = sorted(neighbors)
                for j in js:
                    yield i, j

        def edge_params(src_idx: int, dst_idx: int, layer_idx: int, width_b: int) -> Dict[str, float | int | str | None]:
            center = width_b // 2
            dist = abs(dst_idx - center)
            fast_risky = dist == 0
            safe_stable = dist == max(center, width_b - 1 - center)

            base_cost = 1.0 + 0.25 * layer_idx + 0.10 * dist
            congestion_coef = 0.35 + 0.08 * ((src_idx + dst_idx + layer_idx) % 3)
            variance_base = 0.04 + 0.02 * dist
            variance_coef = 0.02 + 0.01 * ((src_idx + 2 * dst_idx) % 2)
            incident_prob = 0.02 + 0.01 * dist
            incident_delay = 0.6 + 0.2 * dist
            failure_mode = "normal"
            fallback_edge_id = None

            if fast_risky:
                base_cost -= 0.12
                variance_base += 0.10
                variance_coef += 0.05
                incident_prob += 0.10
                incident_delay += 0.30
                failure_mode = "stay"
            if safe_stable:
                base_cost += 0.12
                variance_base = max(0.01, variance_base - 0.02)
                variance_coef = max(0.01, variance_coef - 0.005)
                incident_prob = max(0.0, incident_prob - 0.015)

            return {
                "capacity": base_capacity,
                "base_cost": round(base_cost, 4),
                "congestion_coef": round(congestion_coef, 4),
                "variance_base": round(variance_base, 4),
                "variance_coef": round(variance_coef, 4),
                "incident_prob": round(incident_prob, 4),
                "incident_delay": round(incident_delay, 4),
                "failure_mode": failure_mode,
                "fallback_edge_id": fallback_edge_id,
            }

        layers_with_terminal: List[Tuple[str, ...]] = [(source,)] + layers + [(target,)]
        for layer_idx in range(len(layers_with_terminal) - 1):
            current_layer = layers_with_terminal[layer_idx]
            next_layer = layers_with_terminal[layer_idx + 1]
            for src_idx, dst_idx in connect_pairs(current_layer, next_layer):
                src = current_layer[src_idx]
                dst = next_layer[dst_idx]
                params = edge_params(src_idx, dst_idx, layer_idx, len(next_layer))
                edges[edge_id] = EdgeSpec(edge_id=edge_id, src=src, dst=dst, **params)
                outgoing[src].append(edge_id)
                edge_id += 1

        return GraphSpec(
            source=source,
            target=target,
            nodes=tuple(all_nodes),
            layer_nodes=tuple(layers),
            edges=edges,
            outgoing={node: tuple(ids) for node, ids in outgoing.items()},
        )


class InfoStateEncoder:
    def __init__(self, config: EncoderConfig) -> None:
        self.config = config

    @staticmethod
    def _bin(value: float, edges: Sequence[float]) -> int:
        return int(bisect_right(list(edges), value))

    def encode(self, obs: LocalObservation) -> InfoStateKey:
        node_count = obs.public_snapshot.node_counts.get(obs.current_node, 0)
        occupancy_ratio = node_count / max(1, sum(obs.public_snapshot.node_counts.values()))

        if obs.feasible_action_ids:
            last_ratios = []
            any_incident = False
            for edge_id in obs.feasible_action_ids:
                feat = obs.edge_feature_map[edge_id]
                last_ratios.append(float(feat[6]))
                any_incident = any_incident or bool(round(float(feat[7])))
            recent_load_ratio = max(last_ratios) if last_ratios else 0.0
        else:
            recent_load_ratio = 0.0
            any_incident = False

        return InfoStateKey(
            time_bin=obs.time_step // max(1, self.config.time_bin_size),
            node_id=obs.current_node,
            risk_bin=self._bin(obs.risk_alpha, self.config.risk_bin_edges),
            occupancy_bin=self._bin(occupancy_ratio, self.config.occupancy_bin_edges),
            recent_load_bin=self._bin(recent_load_ratio, self.config.recent_load_bin_edges),
            incident_bin=int(any_incident),
        )


# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------


class SDNCGEnv:
    FEATURE_ORDER = (
        "base_cost",
        "congestion_coef",
        "variance_base",
        "variance_coef",
        "incident_prob",
        "node_occupancy_ratio",
        "last_edge_load_ratio",
        "last_incident_flag",
        "time_ratio",
        "risk_alpha",
        # Explicit risk-interaction features. These make it easier for the
        # teacher and PPO student to separate risk-neutral and risk-averse
        # behavior instead of asking the network to infer alpha x variance
        # interactions implicitly from raw features.
        "risk_x_variance_base",
        "risk_x_variance_coef",
        "risk_x_incident_prob",
        "risk_adjusted_base_cost",
        "risk_adjusted_recent_cost",
    )

    def __init__(
        self,
        graph: GraphSpec,
        agents: Sequence[AgentSpec],
        config: EnvConfig,
        info_state_encoder: InfoStateEncoder | None = None,
    ) -> None:
        self.graph = graph
        self.agents = tuple(agents)
        self.agent_specs = {agent.agent_id: agent for agent in self.agents}
        self.config = config
        self.info_state_encoder = info_state_encoder
        self.rng = np.random.default_rng(config.random_seed)
        self.time_step = 0
        self.positions: Dict[int, str] = {}
        self.last_edge_loads: Dict[int, int] = {eid: 0 for eid in self.graph.edges}
        self.last_incidents: Dict[int, bool] = {eid: False for eid in self.graph.edges}
        self.terminated: Dict[int, bool] = {}
        self.reset(config.random_seed)

    @property
    def feature_dim(self) -> int:
        return len(self.FEATURE_ORDER)

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def reset(self, seed: int | None = None) -> Mapping[int, LocalObservation]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.time_step = 0
        self.positions = {agent.agent_id: self.graph.source for agent in self.agents}
        self.terminated = {agent.agent_id: False for agent in self.agents}
        self.last_edge_loads = {eid: 0 for eid in self.graph.edges}
        self.last_incidents = {eid: False for eid in self.graph.edges}
        return self.get_observations()

    def get_public_snapshot(self) -> PublicStateSnapshot:
        node_counts = {node: 0 for node in self.graph.nodes}
        for pos in self.positions.values():
            node_counts[pos] += 1
        return PublicStateSnapshot(
            time_step=self.time_step,
            node_counts=node_counts,
            last_edge_loads=dict(self.last_edge_loads),
            last_incidents=dict(self.last_incidents),
        )

    def feasible_actions(self, agent_id: int) -> Tuple[int, ...]:
        if self.terminated.get(agent_id, True):
            return ()
        node = self.positions[agent_id]
        if node == self.graph.target:
            return ()
        return self.graph.outgoing.get(node, ())

    def _edge_feature_vector(self, edge_id: int, snapshot: PublicStateSnapshot, current_node: str, risk_alpha: float) -> np.ndarray:
        edge = self.graph.edges[edge_id]
        node_count = snapshot.node_counts.get(current_node, 0)
        last_load_ratio = snapshot.last_load_ratio(edge_id, edge.capacity)
        recent_load = max(1, int(round(last_load_ratio * max(1, edge.capacity))))
        recent_mean, recent_var = self.expected_cost_statistics(edge_id, recent_load)
        return np.asarray(
            [
                edge.base_cost,
                edge.congestion_coef,
                edge.variance_base,
                edge.variance_coef,
                edge.incident_prob,
                node_count / max(1, self.num_agents),
                last_load_ratio,
                1.0 if snapshot.last_incidents.get(edge_id, False) else 0.0,
                self.time_step / max(1, self.config.horizon),
                risk_alpha,
                risk_alpha * edge.variance_base,
                risk_alpha * edge.variance_coef,
                risk_alpha * edge.incident_prob,
                edge.base_cost + 0.5 * risk_alpha * edge.variance_base,
                recent_mean + 0.5 * risk_alpha * recent_var,
            ],
            dtype=np.float32,
        )

    def build_observation(self, agent_id: int, snapshot: PublicStateSnapshot | None = None) -> LocalObservation:
        snapshot = snapshot if snapshot is not None else self.get_public_snapshot()
        node = self.positions[agent_id]
        feasible = self.feasible_actions(agent_id)
        risk_alpha = self.agent_specs[agent_id].risk_alpha
        done = self.terminated.get(agent_id, True) or node == self.graph.target or self.time_step >= self.config.horizon
        edge_feature_map = {eid: self._edge_feature_vector(eid, snapshot, node, risk_alpha) for eid in feasible}
        return LocalObservation(
            agent_id=agent_id,
            time_step=self.time_step,
            current_node=node,
            risk_alpha=risk_alpha,
            feasible_action_ids=tuple(feasible),
            edge_feature_map=edge_feature_map,
            public_snapshot=snapshot,
            done=done,
        )

    def get_observations(self) -> Mapping[int, LocalObservation]:
        snapshot = self.get_public_snapshot()
        return {agent.agent_id: self.build_observation(agent.agent_id, snapshot) for agent in self.agents}

    def expected_cost_statistics(self, edge_id: int, load: int) -> Tuple[float, float]:
        edge = self.graph.edges[edge_id]
        base_mean = edge.mean_cost(load)
        base_var = edge.variance(load)
        incident_mean = edge.incident_prob * edge.incident_delay
        incident_var = edge.incident_prob * (1.0 - edge.incident_prob) * (edge.incident_delay ** 2)
        return base_mean + incident_mean, base_var + incident_var

    def effective_cost(self, edge_id: int, load: int, risk_alpha: float) -> float:
        mean_cost, variance_cost = self.expected_cost_statistics(edge_id, load)
        return float(mean_cost + 0.5 * risk_alpha * variance_cost)

    def realized_cost_sample(self, edge_id: int, load: int, incident_triggered: bool) -> float:
        edge = self.graph.edges[edge_id]
        base_mean = edge.mean_cost(load)
        base_var = edge.variance(load)
        sampled = float(self.rng.normal(base_mean, max(self.config.normal_std_floor, math.sqrt(base_var))))
        sampled = max(0.0, sampled)
        if incident_triggered:
            sampled += edge.incident_delay
        return float(min(self.config.cost_clip, sampled))

    def _shared_incident_map(self, loads: Mapping[int, int]) -> Dict[int, bool]:
        incidents = {eid: False for eid in self.graph.edges}
        for edge_id, load in loads.items():
            if load <= 0:
                continue
            edge = self.graph.edges[edge_id]
            incidents[edge_id] = bool(self.rng.random() < edge.incident_prob)
        return incidents

    def _next_node(self, current_node: str, edge_id: int, incident_triggered: bool) -> str:
        edge = self.graph.edges[edge_id]
        if not incident_triggered:
            return edge.dst
        if edge.failure_mode == "stay":
            return current_node
        if edge.failure_mode == "fallback" and edge.fallback_edge_id is not None:
            return self.graph.edges[edge.fallback_edge_id].dst
        return edge.dst

    def estimate_cost_normalizer(self) -> float:
        max_alpha = max(agent.risk_alpha for agent in self.agents) if self.agents else 0.0
        worst_step = 0.0
        for edge in self.graph.edges.values():
            mean_cost, variance_cost = self.expected_cost_statistics(edge.edge_id, self.num_agents)
            worst_step = max(worst_step, mean_cost + 0.5 * max_alpha * variance_cost)
        return max(1.0, worst_step * max(1, self.config.horizon))

    def step(self, actions: Mapping[int, int]) -> EnvStepResult:
        snapshot_before = self.get_public_snapshot()
        positions_before = dict(self.positions)
        active_agents = [aid for aid in self.positions if not self.terminated[aid] and self.positions[aid] != self.graph.target]

        for aid in active_agents:
            feasible = self.feasible_actions(aid)
            if aid not in actions:
                raise KeyError(f"Missing action for active agent {aid}")
            if actions[aid] not in feasible:
                raise ValueError(f"Invalid action {actions[aid]} for agent {aid}; feasible={feasible}")

        loads: Dict[int, int] = {}
        for aid in active_agents:
            edge_id = actions[aid]
            loads[edge_id] = loads.get(edge_id, 0) + 1

        shared_incidents = self._shared_incident_map(loads) if self.config.shared_edge_incident else {eid: False for eid in self.graph.edges}
        rewards: Dict[int, float] = {}
        dones: Dict[int, bool] = {}
        outcomes: Dict[int, StepAgentOutcome] = {}
        new_positions = dict(self.positions)

        for aid in self.positions:
            if aid not in active_agents:
                rewards[aid] = 0.0
                dones[aid] = True
                outcomes[aid] = StepAgentOutcome(
                    agent_id=aid,
                    chosen_edge_id=None,
                    current_node=self.positions[aid],
                    next_node=self.positions[aid],
                    load_on_chosen_edge=0,
                    mean_cost=0.0,
                    variance_cost=0.0,
                    effective_cost=0.0,
                    realized_cost=0.0,
                    incident_triggered=False,
                    terminated=True,
                )
                continue

            current_node = self.positions[aid]
            chosen_edge = actions[aid]
            load = loads[chosen_edge]
            edge = self.graph.edges[chosen_edge]
            incident = shared_incidents[chosen_edge] if self.config.shared_edge_incident else bool(self.rng.random() < edge.incident_prob)
            mean_cost, variance_cost = self.expected_cost_statistics(chosen_edge, load)
            effective_cost = self.effective_cost(chosen_edge, load, self.agent_specs[aid].risk_alpha)
            realized_cost = self.realized_cost_sample(chosen_edge, load, incident)
            next_node = self._next_node(current_node, chosen_edge, incident)
            terminated = next_node == self.graph.target or (self.time_step + 1) >= self.config.horizon

            new_positions[aid] = next_node
            self.terminated[aid] = terminated

            reward_cost = effective_cost if self.config.reward_mode == "effective" else realized_cost
            rewards[aid] = -float(min(self.config.cost_clip, reward_cost))
            dones[aid] = terminated
            outcomes[aid] = StepAgentOutcome(
                agent_id=aid,
                chosen_edge_id=chosen_edge,
                current_node=current_node,
                next_node=next_node,
                load_on_chosen_edge=load,
                mean_cost=float(min(self.config.cost_clip, mean_cost)),
                variance_cost=float(variance_cost),
                effective_cost=float(min(self.config.cost_clip, effective_cost)),
                realized_cost=float(min(self.config.cost_clip, realized_cost)),
                incident_triggered=incident,
                terminated=terminated,
            )

        self.positions = new_positions
        self.last_edge_loads = {eid: loads.get(eid, 0) for eid in self.graph.edges}
        self.last_incidents = {eid: shared_incidents.get(eid, False) for eid in self.graph.edges}
        self.time_step += 1
        return EnvStepResult(
            observations=self.get_observations(),
            rewards=rewards,
            dones=dones,
            outcomes=outcomes,
            public_snapshot_before=snapshot_before,
            positions_before=positions_before,
            joint_actions=dict(actions),
        )

    def build_decision_records(self, step_result: EnvStepResult) -> Tuple[DecisionRecord, ...]:
        records: List[DecisionRecord] = []
        snapshot = step_result.public_snapshot_before
        for aid, chosen_edge in step_result.joint_actions.items():
            risk_alpha = self.agent_specs[aid].risk_alpha
            current_node = step_result.positions_before[aid]
            feasible = self.graph.outgoing.get(current_node, ())
            obs = LocalObservation(
                agent_id=aid,
                time_step=snapshot.time_step,
                current_node=current_node,
                risk_alpha=risk_alpha,
                feasible_action_ids=feasible,
                edge_feature_map={eid: self._edge_feature_vector(eid, snapshot, current_node, risk_alpha) for eid in feasible},
                public_snapshot=snapshot,
                done=False,
            )
            info_state = self.info_state_encoder.encode(obs) if self.info_state_encoder is not None else None
            outcome = step_result.outcomes[aid]
            records.append(
                DecisionRecord(
                    agent_id=aid,
                    time_step=snapshot.time_step,
                    current_node=current_node,
                    action_id=chosen_edge,
                    feasible_action_ids=tuple(feasible),
                    risk_alpha=risk_alpha,
                    info_state_key=info_state,
                    public_snapshot=snapshot,
                    positions_before=dict(step_result.positions_before),
                    joint_actions=dict(step_result.joint_actions),
                    mean_cost=outcome.mean_cost,
                    variance_cost=outcome.variance_cost,
                    effective_cost=outcome.effective_cost,
                    realized_cost=outcome.realized_cost,
                    load_on_action=outcome.load_on_chosen_edge,
                )
            )
        return tuple(records)

    def counterfactual_effective_cost(self, agent_id: int, candidate_edge_id: int, positions_before: Mapping[int, str], joint_actions: Mapping[int, int]) -> float:
        candidate_joint = dict(joint_actions)
        candidate_joint[agent_id] = candidate_edge_id
        loads: Dict[int, int] = {}
        for aid, edge_id in candidate_joint.items():
            if positions_before[aid] == self.graph.target:
                continue
            loads[edge_id] = loads.get(edge_id, 0) + 1
        load = loads.get(candidate_edge_id, 0)
        return self.effective_cost(candidate_edge_id, load, self.agent_specs[agent_id].risk_alpha)


# -----------------------------------------------------------------------------
# Unified empirical CE evaluator
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class CEDiagnostic:
    num_decisions: int
    mean_positive_regret: float
    max_agent_regret: float
    ucb95: float
    stability_band: float
    epsilon_ce_ex_post: bool
    per_agent_mean_regret: Mapping[int, float]
    per_info_state_mean_regret: Mapping[str, float]


class CEEvaluator:
    def __init__(self, env: SDNCGEnv, epsilon: float = 0.01, stability_threshold: float = 0.005, trailing_window: int = 200) -> None:
        self.env = env
        self.epsilon = epsilon
        self.stability_threshold = stability_threshold
        self.trailing_window = trailing_window
        self.normalizer = env.estimate_cost_normalizer()

    def _record_positive_regret(self, record: DecisionRecord) -> float:
        if record.action_id is None or not record.feasible_action_ids:
            return 0.0
        chosen_cost = self.env.counterfactual_effective_cost(record.agent_id, record.action_id, record.positions_before, record.joint_actions)
        best_improvement = 0.0
        for alt_action in record.feasible_action_ids:
            alt_cost = self.env.counterfactual_effective_cost(record.agent_id, alt_action, record.positions_before, record.joint_actions)
            best_improvement = max(best_improvement, chosen_cost - alt_cost)
        return max(0.0, best_improvement) / self.normalizer

    def evaluate(self, decision_records: Sequence[DecisionRecord]) -> CEDiagnostic:
        if not decision_records:
            return CEDiagnostic(0, 0.0, 0.0, 0.0, 0.0, True, {}, {})

        regrets = np.asarray([self._record_positive_regret(r) for r in decision_records], dtype=np.float64)
        mean_regret = float(np.mean(regrets))
        stderr = float(np.std(regrets, ddof=1) / math.sqrt(regrets.size)) if regrets.size > 1 else 0.0
        ucb95 = mean_regret + 1.96 * stderr
        tail = regrets[-min(self.trailing_window, regrets.size):]
        stability_band = float(np.max(tail) - np.min(tail)) if tail.size else 0.0

        per_agent: Dict[int, List[float]] = {}
        per_info: Dict[str, List[float]] = {}
        for record, regret in zip(decision_records, regrets):
            per_agent.setdefault(record.agent_id, []).append(float(regret))
            key = str(record.info_state_key.as_tuple()) if record.info_state_key is not None else "None"
            per_info.setdefault(key, []).append(float(regret))

        per_agent_mean = {k: float(np.mean(v)) for k, v in per_agent.items()}
        per_info_mean = {k: float(np.mean(v)) for k, v in per_info.items()}
        max_agent_regret = max(per_agent_mean.values()) if per_agent_mean else 0.0

        return CEDiagnostic(
            num_decisions=len(decision_records),
            mean_positive_regret=mean_regret,
            max_agent_regret=max_agent_regret,
            ucb95=ucb95,
            stability_band=stability_band,
            epsilon_ce_ex_post=(ucb95 < self.epsilon and stability_band < self.stability_threshold),
            per_agent_mean_regret=per_agent_mean,
            per_info_state_mean_regret=per_info_mean,
        )


# -----------------------------------------------------------------------------
# Risk-stratified diagnostics
# -----------------------------------------------------------------------------

def _risk_label(alpha: float) -> str:
    return f"{float(alpha):.2f}".replace("-", "m").replace(".", "p")

def _known_risk_values(env: SDNCGEnv) -> Tuple[float, ...]:
    return tuple(sorted({float(agent.risk_alpha) for agent in env.agents}))

def _empty_risk_episode_maps(env: SDNCGEnv) -> Tuple[Dict[str, float], Dict[str, float]]:
    labels = [_risk_label(alpha) for alpha in _known_risk_values(env)]
    return ({label: 0.0 for label in labels}, {label: 0.0 for label in labels})

def _risk_diagnostic_fields(
    env: SDNCGEnv,
    cfg: EvalConfig,
    decision_records: Sequence[DecisionRecord],
    episode_realized_by_risk: Mapping[str, Sequence[float]],
    episode_effective_by_risk: Mapping[str, Sequence[float]],
) -> Dict[str, float]:
    fields: Dict[str, float] = {}
    evaluator = CEEvaluator(env, cfg.epsilon, cfg.stability_threshold, cfg.trailing_window)
    for alpha in _known_risk_values(env):
        label = _risk_label(alpha)
        subset = [r for r in decision_records if abs(float(r.risk_alpha) - float(alpha)) < 1e-9]
        diag = evaluator.evaluate(subset)
        realized_vals = list(episode_realized_by_risk.get(label, []))
        effective_vals = list(episode_effective_by_risk.get(label, []))
        prefix = f"risk_alpha_{label}_"
        fields[prefix + "ce_mean_positive_regret"] = float(diag.mean_positive_regret)
        fields[prefix + "ce_ucb95"] = float(diag.ucb95)
        fields[prefix + "ce_max_agent_regret"] = float(diag.max_agent_regret)
        fields[prefix + "num_eval_decisions"] = float(diag.num_decisions)
        fields[prefix + "eval_realized_cost_mean"] = float(np.mean(realized_vals)) if realized_vals else 0.0
        fields[prefix + "eval_effective_cost_mean"] = float(np.mean(effective_vals)) if effective_vals else 0.0
    return fields


# -----------------------------------------------------------------------------
# PPO network and agent
# -----------------------------------------------------------------------------


class MaskedActorCritic(nn.Module):
    def __init__(self, num_nodes: int, feature_dim: int, hidden_dim: int, node_embed_dim: int) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
        self.actor = nn.Sequential(
            nn.Linear(feature_dim + node_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(feature_dim * 2 + node_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def action_logits(self, edge_features: torch.Tensor, node_idx: int) -> torch.Tensor:
        node_id = torch.tensor(node_idx, dtype=torch.long, device=edge_features.device)
        node_emb = self.node_embedding(node_id).unsqueeze(0).expand(edge_features.shape[0], -1)
        x = torch.cat([edge_features, node_emb], dim=-1)
        return self.actor(x).squeeze(-1)

    def state_value(self, edge_features: torch.Tensor, node_idx: int) -> torch.Tensor:
        node_id = torch.tensor(node_idx, dtype=torch.long, device=edge_features.device)
        node_emb = self.node_embedding(node_id)
        mean_feat = edge_features.mean(dim=0)
        max_feat = edge_features.max(dim=0).values
        x = torch.cat([mean_feat, max_feat, node_emb], dim=-1)
        return self.critic(x).squeeze(-1)


class PPOAgent:
    def __init__(self, env: SDNCGEnv, cfg: PPOConfig, device: torch.device) -> None:
        self.env = env
        self.cfg = cfg
        self.device = device
        self.node_to_idx = {node: idx for idx, node in enumerate(env.graph.nodes)}
        self.model = MaskedActorCritic(len(self.node_to_idx), env.feature_dim, cfg.hidden_dim, cfg.node_embed_dim).to(device)
        self.optimizer = optim.Adam(
            [
                {"params": self.model.node_embedding.parameters(), "lr": cfg.actor_lr},
                {"params": self.model.actor.parameters(), "lr": cfg.actor_lr},
                {"params": self.model.critic.parameters(), "lr": cfg.critic_lr},
            ]
        )

    def obs_to_arrays(self, obs: LocalObservation) -> Tuple[int, np.ndarray]:
        _, edge_mat = obs.to_feature_matrix()
        return self.node_to_idx[obs.current_node], edge_mat.astype(np.float32)

    @torch.no_grad()
    def act(self, obs: LocalObservation, deterministic: bool = False) -> Tuple[int, int, float, float]:
        node_idx, edge_mat = self.obs_to_arrays(obs)
        edge_tensor = torch.tensor(edge_mat, dtype=torch.float32, device=self.device)
        logits = self.model.action_logits(edge_tensor, node_idx)
        probs = torch.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        action_index = int(torch.argmax(probs).item()) if deterministic else int(dist.sample().item())
        action_id = int(obs.feasible_action_ids[action_index])
        logprob = float(dist.log_prob(torch.tensor(action_index, dtype=torch.long, device=self.device)).item())
        value = float(self.model.state_value(edge_tensor, node_idx).item())
        return action_id, action_index, logprob, value

    @torch.no_grad()
    def value_of(self, obs: LocalObservation) -> float:
        if obs.done or not obs.feasible_action_ids:
            return 0.0
        node_idx, edge_mat = self.obs_to_arrays(obs)
        edge_tensor = torch.tensor(edge_mat, dtype=torch.float32, device=self.device)
        return float(self.model.state_value(edge_tensor, node_idx).item())

    def evaluate_sample(self, sample: PPOSample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_tensor = torch.tensor(sample.edge_features, dtype=torch.float32, device=self.device)
        logits = self.model.action_logits(edge_tensor, sample.current_node_idx)
        probs = torch.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        action_idx = torch.tensor(sample.action_index, dtype=torch.long, device=self.device)
        logprob = dist.log_prob(action_idx)
        entropy = dist.entropy()
        value = self.model.state_value(edge_tensor, sample.current_node_idx)
        return logprob, entropy, value

    def update(self, samples: Sequence[PPOSample]) -> Dict[str, float]:
        if not samples:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}

        advantages = np.asarray([s.advantage for s in samples], dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = np.asarray([s.return_ for s in samples], dtype=np.float32)
        old_logprobs = np.asarray([s.old_logprob for s in samples], dtype=np.float32)
        indices = np.arange(len(samples))

        meters = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0}
        num_steps = 0

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.cfg.minibatch_size):
                mb_idx = indices[start:start + self.cfg.minibatch_size]
                if len(mb_idx) == 0:
                    continue
                policy_terms = []
                value_terms = []
                entropy_terms = []
                for idx in mb_idx:
                    sample = samples[idx]
                    new_logprob, entropy, value = self.evaluate_sample(sample)
                    ratio = torch.exp(new_logprob - torch.tensor(old_logprobs[idx], dtype=torch.float32, device=self.device))
                    adv = torch.tensor(advantages[idx], dtype=torch.float32, device=self.device)
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv
                    policy_terms.append(-torch.min(unclipped, clipped))
                    target_return = torch.tensor(returns[idx], dtype=torch.float32, device=self.device)
                    value_terms.append((value - target_return) ** 2)
                    entropy_terms.append(entropy)

                policy_loss = torch.stack(policy_terms).mean()
                value_loss = torch.stack(value_terms).mean()
                entropy = torch.stack(entropy_terms).mean()
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                meters["loss"] += float(loss.item())
                meters["policy_loss"] += float(policy_loss.item())
                meters["value_loss"] += float(value_loss.item())
                meters["entropy"] += float(entropy.item())
                num_steps += 1

        if num_steps > 0:
            for key in meters:
                meters[key] /= num_steps
        return meters


# -----------------------------------------------------------------------------
# Rollout, training, evaluation
# -----------------------------------------------------------------------------



def build_agents(num_agents: int, risk_alphas: Sequence[float]) -> List[AgentSpec]:
    if not risk_alphas:
        risk_alphas = [0.0]
    agents: List[AgentSpec] = []
    for i in range(num_agents):
        alpha = float(risk_alphas[i % len(risk_alphas)])
        agents.append(AgentSpec(agent_id=i, risk_alpha=alpha, type_name=f"alpha_{alpha:.2f}"))
    return agents


def summarize_population(num_agents: int, risk_alphas: Sequence[float]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for agent in build_agents(num_agents, risk_alphas):
        key = f"{agent.risk_alpha:.2f}"
        summary[key] = summary.get(key, 0) + 1
    return summary


def build_env(
    seed: int,
    widths: Sequence[int],
    topology: str,
    base_capacity: int,
    env_cfg: EnvConfig,
    encoder_cfg: EncoderConfig,
    risk_alphas: Sequence[float],
    num_agents: int | None = None,
) -> SDNCGEnv:
    graph = LayeredGraphBuilder.build(widths=widths, topology=topology, base_capacity=base_capacity)
    actual_num_agents = env_cfg.num_agents if num_agents is None else int(num_agents)
    agents = build_agents(actual_num_agents, risk_alphas)
    encoder = InfoStateEncoder(encoder_cfg)
    return SDNCGEnv(
        graph=graph,
        agents=agents,
        config=EnvConfig(
            num_agents=actual_num_agents,
            horizon=env_cfg.horizon,
            cost_clip=env_cfg.cost_clip,
            reward_mode=env_cfg.reward_mode,
            shared_edge_incident=env_cfg.shared_edge_incident,
            normal_std_floor=env_cfg.normal_std_floor,
            random_seed=seed,
        ),
        info_state_encoder=encoder,
    )


def compute_gae(trajectory: List[PPOSample], gamma: float, gae_lambda: float) -> None:
    gae = 0.0
    next_value = 0.0
    for sample in reversed(trajectory):
        mask = 0.0 if sample.done else 1.0
        delta = sample.reward + gamma * next_value * mask - sample.value
        gae = delta + gamma * gae_lambda * mask * gae
        sample.advantage = float(gae)
        sample.return_ = float(sample.value + sample.advantage)
        next_value = sample.value


def collect_training_batch(env: SDNCGEnv, agent: PPOAgent, cfg: PPOConfig, seed_offset: int) -> Tuple[List[PPOSample], Dict[str, float]]:
    samples: List[PPOSample] = []
    stats = {"episodes": 0.0, "train_realized_cost": 0.0, "train_effective_cost": 0.0, "train_decisions": 0.0}
    for ep in range(cfg.episodes_per_update):
        episode_seed = seed_offset + ep
        env.reset(episode_seed)
        per_agent_traj: Dict[int, List[PPOSample]] = {aid: [] for aid in env.positions.keys()}
        total_realized = 0.0
        total_effective = 0.0
        total_decisions = 0
        while env.time_step < env.config.horizon and not all(env.terminated.values()):
            observations = env.get_observations()
            action_map: Dict[int, int] = {}
            meta: Dict[int, Tuple[int, float, float, np.ndarray, int, InfoStateKey | None]] = {}
            for aid, obs in observations.items():
                if obs.done or not obs.feasible_action_ids:
                    continue
                action_id, action_index, logprob, value = agent.act(obs, deterministic=False)
                node_idx, edge_mat = agent.obs_to_arrays(obs)
                info_state = env.info_state_encoder.encode(obs) if env.info_state_encoder is not None else None
                action_map[aid] = action_id
                meta[aid] = (action_index, logprob, value, edge_mat.copy(), node_idx, info_state)
            if not action_map:
                break
            step_result = env.step(action_map)
            total_realized += sum(out.realized_cost for out in step_result.outcomes.values())
            total_effective += sum(out.effective_cost for out in step_result.outcomes.values())
            total_decisions += len(step_result.joint_actions)
            for aid in step_result.joint_actions.keys():
                action_index, logprob, value, edge_mat, node_idx, info_state = meta[aid]
                per_agent_traj[aid].append(
                    PPOSample(
                        current_node_idx=node_idx,
                        feasible_action_ids=tuple(observations[aid].feasible_action_ids),
                        edge_features=edge_mat,
                        action_index=action_index,
                        old_logprob=float(logprob),
                        value=float(value),
                        reward=float(step_result.rewards[aid]),
                        done=bool(step_result.dones[aid]),
                        agent_id=aid,
                        time_step=observations[aid].time_step,
                        info_state_key=info_state,
                    )
                )
        for traj in per_agent_traj.values():
            if traj:
                compute_gae(traj, gamma=cfg.gamma, gae_lambda=cfg.gae_lambda)
                samples.extend(traj)
        stats["episodes"] += 1.0
        stats["train_realized_cost"] += total_realized
        stats["train_effective_cost"] += total_effective
        stats["train_decisions"] += total_decisions
    if stats["episodes"] > 0:
        stats["train_realized_cost"] /= stats["episodes"]
        stats["train_effective_cost"] /= stats["episodes"]
        stats["train_decisions"] /= stats["episodes"]
    return samples, stats


@torch.no_grad()
def evaluate_policy(env: SDNCGEnv, agent: PPOAgent, cfg: EvalConfig, eval_episodes: int, seed_offset: int, deterministic: bool = False) -> Dict[str, float]:
    episode_realized: List[float] = []
    episode_effective: List[float] = []
    decision_records: List[DecisionRecord] = []
    risk_values = _known_risk_values(env)
    episode_realized_by_risk: Dict[str, List[float]] = {_risk_label(alpha): [] for alpha in risk_values}
    episode_effective_by_risk: Dict[str, List[float]] = {_risk_label(alpha): [] for alpha in risk_values}
    for ep in range(eval_episodes):
        env.reset(seed_offset + ep)
        total_realized = 0.0
        total_effective = 0.0
        ep_realized_by_risk, ep_effective_by_risk = _empty_risk_episode_maps(env)
        while env.time_step < env.config.horizon and not all(env.terminated.values()):
            observations = env.get_observations()
            action_map: Dict[int, int] = {}
            for aid, obs in observations.items():
                if obs.done or not obs.feasible_action_ids:
                    continue
                action_id, _, _, _ = agent.act(obs, deterministic=deterministic)
                action_map[aid] = action_id
            if not action_map:
                break
            step_result = env.step(action_map)
            decision_records.extend(env.build_decision_records(step_result))
            total_realized += sum(out.realized_cost for out in step_result.outcomes.values())
            total_effective += sum(out.effective_cost for out in step_result.outcomes.values())
            for aid, out in step_result.outcomes.items():
                if out.chosen_edge_id is None:
                    continue
                label = _risk_label(env.agent_specs[aid].risk_alpha)
                ep_realized_by_risk[label] = ep_realized_by_risk.get(label, 0.0) + float(out.realized_cost)
                ep_effective_by_risk[label] = ep_effective_by_risk.get(label, 0.0) + float(out.effective_cost)
        episode_realized.append(total_realized)
        episode_effective.append(total_effective)
        for label in episode_realized_by_risk:
            episode_realized_by_risk[label].append(float(ep_realized_by_risk.get(label, 0.0)))
            episode_effective_by_risk[label].append(float(ep_effective_by_risk.get(label, 0.0)))
    ce_diag = CEEvaluator(env, cfg.epsilon, cfg.stability_threshold, cfg.trailing_window).evaluate(decision_records)
    result = {
        "eval_realized_cost_mean": float(np.mean(episode_realized)) if episode_realized else 0.0,
        "eval_realized_cost_std": float(np.std(episode_realized, ddof=1)) if len(episode_realized) > 1 else 0.0,
        "eval_effective_cost_mean": float(np.mean(episode_effective)) if episode_effective else 0.0,
        "eval_effective_cost_std": float(np.std(episode_effective, ddof=1)) if len(episode_effective) > 1 else 0.0,
        "ce_mean_positive_regret": ce_diag.mean_positive_regret,
        "ce_max_agent_regret": ce_diag.max_agent_regret,
        "ce_ucb95": ce_diag.ucb95,
        "ce_stability_band": ce_diag.stability_band,
        "epsilon_ce_ex_post": float(ce_diag.epsilon_ce_ex_post),
        "num_eval_decisions": float(ce_diag.num_decisions),
    }
    result.update(_risk_diagnostic_fields(env, cfg, decision_records, episode_realized_by_risk, episode_effective_by_risk))
    return result


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    stderr = std / math.sqrt(arr.size) if arr.size > 1 else 0.0
    ci95 = 1.96 * stderr
    return mean, std, stderr, ci95


def plot_seed_curves(rows: List[Dict[str, float]], output_path: Path, title: str) -> None:
    if not rows:
        return
    updates = [int(r["update"]) for r in rows]
    regrets = [r["ce_mean_positive_regret"] for r in rows]
    costs = [r["eval_realized_cost_mean"] for r in rows]
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(updates, regrets, marker="o", label="CE regret")
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Mean positive regret")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(updates, costs, linestyle="--", marker="s", label="Eval realized cost")
    ax2.set_ylabel("Eval realized cost")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_aggregate_curves(rows: List[Dict[str, float]], output_path: Path, title: str) -> None:
    if not rows:
        return
    grouped: Dict[int, List[Dict[str, float]]] = {}
    for row in rows:
        grouped.setdefault(int(row["update"]), []).append(row)
    updates = sorted(grouped.keys())
    ce_stats = [_mean_std_ci([r["ce_mean_positive_regret"] for r in grouped[u]]) for u in updates]
    ce_mean = [s[0] for s in ce_stats]
    ce_ci = [s[3] for s in ce_stats]
    cost_stats = [_mean_std_ci([r["eval_realized_cost_mean"] for r in grouped[u]]) for u in updates]
    cost_mean = [s[0] for s in cost_stats]
    cost_ci = [s[3] for s in cost_stats]
    fig = plt.figure(figsize=(8, 5))
    ax1 = fig.add_subplot(111)
    ax1.plot(updates, ce_mean, marker="o", label="CE regret mean")
    ax1.fill_between(updates, np.array(ce_mean) - np.array(ce_ci), np.array(ce_mean) + np.array(ce_ci), alpha=0.2)
    ax1.set_xlabel("Update")
    ax1.set_ylabel("Mean positive regret")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(updates, cost_mean, linestyle="--", marker="s", label="Eval realized cost mean")
    ax2.fill_between(updates, np.array(cost_mean) - np.array(cost_ci), np.array(cost_mean) + np.array(cost_ci), alpha=0.15)
    ax2.set_ylabel("Eval realized cost")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_by_agent_count(rows: List[Dict[str, float]], metric_key: str, output_path: Path, title: str, ylabel: str) -> None:
    if not rows:
        return
    grouped: Dict[int, Dict[int, List[float]]] = {}
    for row in rows:
        n = int(row["agent_count"])
        u = int(row["update"])
        grouped.setdefault(n, {}).setdefault(u, []).append(float(row[metric_key]))
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    for agent_count in sorted(grouped.keys()):
        updates = sorted(grouped[agent_count].keys())
        stats = [_mean_std_ci(grouped[agent_count][u]) for u in updates]
        means = [s[0] for s in stats]
        cis = [s[3] for s in stats]
        ax.plot(updates, means, marker="o", label=f"N={agent_count}")
        ax.fill_between(updates, np.array(means) - np.array(cis), np.array(means) + np.array(cis), alpha=0.15)
    ax.set_xlabel("Update")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)



# -----------------------------------------------------------------------------
# Experiment runner
# -----------------------------------------------------------------------------


def parse_int_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(x.strip()) for x in text.split(",") if x.strip())


def parse_float_tuple(text: str) -> Tuple[float, ...]:
    return tuple(float(x.strip()) for x in text.split(",") if x.strip())


def summarize_final_rows(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {
            "num_seeds": 0,
            "final_update": 0,
            "ce_mean_positive_regret_mean": 0.0,
            "ce_mean_positive_regret_std": 0.0,
            "ce_ucb95_mean": 0.0,
            "eval_realized_cost_mean": 0.0,
            "eval_realized_cost_std": 0.0,
            "fraction_ex_post_epsilon_ce": 0.0,
        }
    final_update = max(int(r["update"]) for r in rows)
    final_rows = [r for r in rows if int(r["update"]) == final_update]
    return {
        "num_seeds": len({int(r["seed"]) for r in final_rows}),
        "final_update": final_update,
        "ce_mean_positive_regret_mean": float(np.mean([r["ce_mean_positive_regret"] for r in final_rows])),
        "ce_mean_positive_regret_std": float(np.std([r["ce_mean_positive_regret"] for r in final_rows], ddof=1)) if len(final_rows) > 1 else 0.0,
        "ce_ucb95_mean": float(np.mean([r["ce_ucb95"] for r in final_rows])),
        "eval_realized_cost_mean": float(np.mean([r["eval_realized_cost_mean"] for r in final_rows])),
        "eval_realized_cost_std": float(np.std([r["eval_realized_cost_mean"] for r in final_rows], ddof=1)) if len(final_rows) > 1 else 0.0,
        "fraction_ex_post_epsilon_ce": float(np.mean([r["epsilon_ce_ex_post"] for r in final_rows])),
    }


def extract_final_rows_by_mode_count_seed(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    for row in rows:
        key = (str(row.get("mode", "unknown")), int(row.get("agent_count", 0)), int(row.get("seed", 0)))
        prev = grouped.get(key)
        if prev is None or float(row.get("update", 0.0)) >= float(prev.get("update", -1.0)):
            grouped[key] = dict(row)
    return [grouped[k] for k in sorted(grouped.keys())]


def build_final_summary_rows(final_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, int], List[Dict[str, float]]] = {}
    for row in final_rows:
        key = (str(row.get("mode", "unknown")), int(row.get("agent_count", 0)))
        grouped.setdefault(key, []).append(dict(row))
    summary_rows: List[Dict[str, float]] = []
    for (mode, agent_count), rows_ in sorted(grouped.items(), key=lambda x: (x[0][1], x[0][0])):
        ce_vals = [float(r.get("ce_mean_positive_regret", 0.0)) for r in rows_]
        ce_ucb_vals = [float(r.get("ce_ucb95", 0.0)) for r in rows_]
        cost_vals = [float(r.get("eval_realized_cost_mean", 0.0)) for r in rows_]
        eps_vals = [float(r.get("epsilon_ce_ex_post", 0.0)) for r in rows_]
        summary_rows.append({
            "mode": mode,
            "agent_count": agent_count,
            "num_seeds": len(rows_),
            "final_update": int(max(float(r.get("update", 0.0)) for r in rows_)) if rows_ else 0,
            "ce_mean_positive_regret_mean": float(np.mean(ce_vals)) if ce_vals else 0.0,
            "ce_mean_positive_regret_std": float(np.std(ce_vals, ddof=1)) if len(ce_vals) > 1 else 0.0,
            "ce_ucb95_mean": float(np.mean(ce_ucb_vals)) if ce_ucb_vals else 0.0,
            "eval_realized_cost_mean": float(np.mean(cost_vals)) if cost_vals else 0.0,
            "eval_realized_cost_std": float(np.std(cost_vals, ddof=1)) if len(cost_vals) > 1 else 0.0,
            "fraction_ex_post_epsilon_ce": float(np.mean(eps_vals)) if eps_vals else 0.0,
        })
    return summary_rows


def run_single_seed(
    seed: int,
    agent_count: int,
    widths: Sequence[int],
    topology: str,
    base_capacity: int,
    risk_alphas: Sequence[float],
    env_cfg: EnvConfig,
    encoder_cfg: EncoderConfig,
    ppo_cfg: PPOConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
    logger: RunLogger,
) -> List[Dict[str, float]]:
    set_global_seed(seed)
    env_cfg_local = EnvConfig(
        num_agents=agent_count,
        horizon=env_cfg.horizon,
        cost_clip=env_cfg.cost_clip,
        reward_mode=env_cfg.reward_mode,
        shared_edge_incident=env_cfg.shared_edge_incident,
        normal_std_floor=env_cfg.normal_std_floor,
        random_seed=seed,
    )
    env = build_env(seed, widths, topology, base_capacity, env_cfg_local, encoder_cfg, risk_alphas, num_agents=agent_count)
    agent = PPOAgent(env, ppo_cfg, device)
    rows: List[Dict[str, float]] = []
    risk_cycle_label = "|".join(f"{x:.2f}" for x in risk_alphas)
    progress_log_every = max(1, ppo_cfg.total_updates // 4)
    for update_idx in range(1, ppo_cfg.total_updates + 1):
        samples, train_stats = collect_training_batch(env, agent, ppo_cfg, seed_offset=seed * 100_000 + update_idx * 1_000)
        opt_stats = agent.update(samples)
        if update_idx == 1 or update_idx % ppo_cfg.eval_every == 0 or update_idx == ppo_cfg.total_updates:
            eval_stats = evaluate_policy(
                env,
                agent,
                eval_cfg,
                eval_episodes=ppo_cfg.eval_episodes,
                seed_offset=seed * 1_000_000 + update_idx * 10_000,
                deterministic=False,
            )
            row = {
                "seed": float(seed),
                "agent_count": float(agent_count),
                "update": float(update_idx),
                "risk_cycle_label": risk_cycle_label,
                **train_stats,
                **opt_stats,
                **eval_stats,
            }
            rows.append(row)
            if update_idx == 1 or update_idx == ppo_cfg.total_updates or update_idx % progress_log_every == 0:
                logger.summary(
                    "N={n:>2d} seed={s} update={u:4d} ce={ce:.4f} ucb95={ucb:.4f} cost={cost:.2f} ent={ent:.3f}".format(
                        n=agent_count,
                        s=seed,
                        u=update_idx,
                        ce=row["ce_mean_positive_regret"],
                        ucb=row["ce_ucb95"],
                        cost=row["eval_realized_cost_mean"],
                        ent=row["entropy"],
                    )
                )
    return rows


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def make_seed_tag(seeds: Sequence[int]) -> str:
    return "-".join(str(int(s)) for s in seeds) if seeds else "none"


def collect_plot_manifest(output_dir: Path) -> List[Dict[str, str]]:
    manifest: List[Dict[str, str]] = []
    for path in sorted(output_dir.rglob("*.png")):
        manifest.append({
            "relative_path": str(path.relative_to(output_dir)),
            "filename": path.name,
            "plot_kind": path.stem.split("__", 1)[0],
        })
    return manifest




# Preserve the PPO-only runner for the shared-teacher final modes.
run_single_seed_ppo = run_single_seed
def parse_str_tuple(text: str) -> Tuple[str, ...]:
    return tuple(x.strip() for x in text.split(",") if x.strip())


def save_rows_csv(rows: List[Dict[str, float]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def filter_rows(rows: Sequence[Dict[str, float]], mode: str | None = None, agent_count: int | None = None) -> List[Dict[str, float]]:
    out: List[Dict[str, float]] = []
    for row in rows:
        if mode is not None and str(row.get("mode")) != mode:
            continue
        if agent_count is not None and int(row.get("agent_count", -1)) != agent_count:
            continue
        out.append(dict(row))
    return out


def plot_metric_by_mode_for_agent_count(
    rows: Sequence[Dict[str, float]],
    agent_count: int,
    metric_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    rows = [dict(r) for r in rows if int(r.get("agent_count", -1)) == agent_count]
    if not rows:
        return
    grouped: Dict[str, Dict[int, List[float]]] = {}
    for row in rows:
        mode = str(row["mode"])
        update = int(row["update"])
        grouped.setdefault(mode, {}).setdefault(update, []).append(float(row[metric_key]))
    fig = plt.figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    for mode in sorted(grouped.keys()):
        updates = sorted(grouped[mode].keys())
        stats = [_mean_std_ci(grouped[mode][u]) for u in updates]
        means = [s[0] for s in stats]
        cis = [s[3] for s in stats]
        ax.plot(updates, means, marker="o", label=mode)
        ax.fill_between(updates, np.array(means) - np.array(cis), np.array(means) + np.array(cis), alpha=0.15)
    ax.set_xlabel("Update")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_metric_by_agent_count_for_mode(
    rows: Sequence[Dict[str, float]],
    mode: str,
    metric_key: str,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    filtered = [dict(r) for r in rows if str(r.get("mode")) == mode]
    if not filtered:
        return
    plot_metric_by_agent_count(filtered, metric_key=metric_key, output_path=output_path, title=title, ylabel=ylabel)



# =============================================================================
# Stage 4-5 extension: DeepCFR-style teacher and four final modes
# =============================================================================


@dataclass(frozen=True)
class DeepTeacherConfig:
    episodes_per_update: int = 8
    total_updates: int = 120
    eval_every: int = 10
    eval_episodes: int = 8
    eval_policy: str = "average"  # {"average", "current"}
    hidden_dim: int = 128
    node_embed_dim: int = 16
    lr: float = 1e-3
    buffer_capacity: int = 50_000
    train_steps: int = 60
    strategy_train_steps: int = 60
    batch_size: int = 64
    min_buffer_before_train: int = 32
    retrain_from_scratch: bool = False
    target_lookahead_steps: int = 2
    target_policy_mode: str = "current"  # {"current", "average"}
    target_future_load_floor: float = 1.0


@dataclass
class TeacherRegretSample:
    current_node_idx: int
    feasible_action_ids: Tuple[int, ...]
    edge_features: np.ndarray
    regret_targets: np.ndarray
    iter_weight: float
    risk_bin: int = 0


@dataclass
class TeacherStrategySample:
    current_node_idx: int
    feasible_action_ids: Tuple[int, ...]
    edge_features: np.ndarray
    policy_targets: np.ndarray
    iter_weight: float
    risk_bin: int = 0


class ReservoirBuffer:
    def __init__(self, capacity: int, seed: int = 0) -> None:
        self.capacity = int(capacity)
        self.rng = np.random.default_rng(seed)
        self.data: List[object] = []
        self.num_seen = 0

    def __len__(self) -> int:
        return len(self.data)

    def add(self, item: object) -> None:
        self.num_seen += 1
        if len(self.data) < self.capacity:
            self.data.append(item)
            return
        j = int(self.rng.integers(0, self.num_seen))
        if j < self.capacity:
            self.data[j] = item

    def sample(self, batch_size: int) -> List[object]:
        if not self.data:
            return []
        size = min(int(batch_size), len(self.data))
        indices = self.rng.choice(len(self.data), size=size, replace=False)
        return [self.data[int(i)] for i in indices]

    def sample_stratified_by_attr(self, batch_size: int, attr_name: str) -> List[object]:
        """Sample approximately balanced minibatches across a discrete attribute.

        This is used for risk-stratified teacher training. A shared neural
        teacher can otherwise average across risk types even when risk_alpha is
        present in the input.
        """
        if not self.data:
            return []
        size = min(int(batch_size), len(self.data))
        groups: Dict[object, List[int]] = {}
        for idx, item in enumerate(self.data):
            groups.setdefault(getattr(item, attr_name, None), []).append(idx)
        if len(groups) <= 1:
            return self.sample(batch_size)
        keys = list(groups.keys())
        picked: List[int] = []
        while len(picked) < size:
            progressed = False
            self.rng.shuffle(keys)
            for key in keys:
                available = [idx for idx in groups[key] if idx not in picked]
                if not available:
                    continue
                picked.append(int(self.rng.choice(available)))
                progressed = True
                if len(picked) >= size:
                    break
            if not progressed:
                break
        if len(picked) < size:
            remaining = [idx for idx in range(len(self.data)) if idx not in picked]
            if remaining:
                extra = self.rng.choice(remaining, size=min(size - len(picked), len(remaining)), replace=False)
                picked.extend(int(i) for i in extra)
        return [self.data[int(i)] for i in picked[:size]]


class MaskedScoreModel(nn.Module):
    def __init__(self, num_nodes: int, feature_dim: int, hidden_dim: int, node_embed_dim: int) -> None:
        super().__init__()
        self.node_embedding = nn.Embedding(num_nodes, node_embed_dim)
        self.scorer = nn.Sequential(
            nn.Linear(feature_dim + node_embed_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def action_scores(self, edge_features: torch.Tensor, node_idx: int) -> torch.Tensor:
        node_id = torch.tensor(node_idx, dtype=torch.long, device=edge_features.device)
        node_emb = self.node_embedding(node_id).unsqueeze(0).expand(edge_features.shape[0], -1)
        x = torch.cat([edge_features, node_emb], dim=-1)
        return self.scorer(x).squeeze(-1)


class DeepCFRStyleTeacher:
    """Practical DeepCFR-style teacher for the rebuilt S-DNCG codebase.

    It keeps the parts that matter most here:
    - regret/value-free policy approximation over information states
    - separate strategy model for the average policy
    - reservoir buffers for regret and strategy samples
    - current-policy training / average-policy evaluation
    """

    def __init__(self, env: SDNCGEnv, cfg: DeepTeacherConfig, device: torch.device, seed: int = 0) -> None:
        self.env = env
        self.cfg = cfg
        self.device = device
        self.seed = int(seed)
        self.node_to_idx = {node: idx for idx, node in enumerate(env.graph.nodes)}
        self.rng = np.random.default_rng(seed)
        self.regret_buffer = ReservoirBuffer(cfg.buffer_capacity, seed=seed + 11)
        self.strategy_buffer = ReservoirBuffer(cfg.buffer_capacity, seed=seed + 29)
        self.regret_model = self._build_model()
        self.strategy_model = self._build_model()
        self.regret_optimizer = optim.Adam(self.regret_model.parameters(), lr=cfg.lr)
        self.strategy_optimizer = optim.Adam(self.strategy_model.parameters(), lr=cfg.lr)
        self.regret_trained = False
        self.strategy_trained = False

    def _build_model(self) -> MaskedScoreModel:
        return MaskedScoreModel(
            num_nodes=len(self.node_to_idx),
            feature_dim=self.env.feature_dim,
            hidden_dim=self.cfg.hidden_dim,
            node_embed_dim=self.cfg.node_embed_dim,
        ).to(self.device)

    def _reset_regret_model(self) -> None:
        self.regret_model = self._build_model()
        self.regret_optimizer = optim.Adam(self.regret_model.parameters(), lr=self.cfg.lr)

    def _reset_strategy_model(self) -> None:
        self.strategy_model = self._build_model()
        self.strategy_optimizer = optim.Adam(self.strategy_model.parameters(), lr=self.cfg.lr)

    def obs_to_arrays(self, obs: LocalObservation) -> Tuple[int, np.ndarray]:
        _, edge_mat = obs.to_feature_matrix()
        return self.node_to_idx[obs.current_node], edge_mat.astype(np.float32)

    @torch.no_grad()
    def current_policy_probs(self, obs: LocalObservation) -> Tuple[np.ndarray, np.ndarray]:
        if obs.done or not obs.feasible_action_ids:
            return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        if not self.regret_trained:
            size = len(obs.feasible_action_ids)
            probs = np.full(size, 1.0 / max(1, size), dtype=np.float64)
            return probs, np.zeros(size, dtype=np.float64)
        node_idx, edge_mat = self.obs_to_arrays(obs)
        edge_tensor = torch.tensor(edge_mat, dtype=torch.float32, device=self.device)
        scores = self.regret_model.action_scores(edge_tensor, node_idx)
        regrets = scores.detach().cpu().numpy().astype(np.float64, copy=False)
        positive = np.maximum(regrets, 0.0)
        total = float(np.sum(positive))
        if total <= 1e-12:
            probs = np.full(len(obs.feasible_action_ids), 1.0 / max(1, len(obs.feasible_action_ids)), dtype=np.float64)
        else:
            probs = positive / total
        return probs, regrets

    @torch.no_grad()
    def average_policy_probs(self, obs: LocalObservation) -> np.ndarray:
        if obs.done or not obs.feasible_action_ids:
            return np.zeros(0, dtype=np.float64)
        if not self.strategy_trained:
            probs, _ = self.current_policy_probs(obs)
            return probs
        node_idx, edge_mat = self.obs_to_arrays(obs)
        edge_tensor = torch.tensor(edge_mat, dtype=torch.float32, device=self.device)
        logits = self.strategy_model.action_scores(edge_tensor, node_idx)
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(probs))
        if total <= 1e-12:
            return np.full(len(obs.feasible_action_ids), 1.0 / max(1, len(obs.feasible_action_ids)), dtype=np.float64)
        return probs / total

    def select_action(self, obs: LocalObservation, policy_mode: str = "current", deterministic: bool = False) -> Tuple[int, int, np.ndarray, np.ndarray]:
        if policy_mode == "average":
            probs = self.average_policy_probs(obs)
            regrets = np.zeros_like(probs)
        else:
            probs, regrets = self.current_policy_probs(obs)
        if probs.size == 0:
            raise ValueError("Deep teacher received an observation with no feasible actions.")
        probs = np.nan_to_num(probs.astype(np.float64, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(probs))
        if total <= 1e-12:
            probs = np.full(len(obs.feasible_action_ids), 1.0 / max(1, len(obs.feasible_action_ids)), dtype=np.float64)
        else:
            probs = probs / total
        if deterministic:
            action_index = int(np.argmax(probs))
        else:
            action_index = int(self.rng.choice(len(obs.feasible_action_ids), p=probs))
        action_id = int(obs.feasible_action_ids[action_index])
        return action_id, action_index, probs.astype(np.float64, copy=False), regrets.astype(np.float64, copy=False)

    def teacher_confidence(self, obs: LocalObservation, policy_mode: "str" = "average") -> float:
        probs = self.average_policy_probs(obs) if policy_mode == "average" else self.current_policy_probs(obs)[0]
        return _teacher_confidence_from_probs(probs)

    def add_strategy_sample(self, obs: LocalObservation, probs: np.ndarray, iter_weight: float) -> None:
        if obs.done or not obs.feasible_action_ids:
            return
        node_idx, edge_mat = self.obs_to_arrays(obs)
        risk_bin = self.env.info_state_encoder.encode(obs).risk_bin if self.env.info_state_encoder is not None else 0
        self.strategy_buffer.add(
            TeacherStrategySample(
                current_node_idx=node_idx,
                feasible_action_ids=tuple(obs.feasible_action_ids),
                edge_features=edge_mat.copy(),
                policy_targets=np.asarray(probs, dtype=np.float32).copy(),
                iter_weight=float(iter_weight),
                risk_bin=int(risk_bin),
            )
        )

    def _edge_feature_vector_for_time(self, edge_id: int, snapshot: PublicStateSnapshot, current_node: str, risk_alpha: float, time_step: int) -> np.ndarray:
        edge = self.env.graph.edges[edge_id]
        node_count = snapshot.node_counts.get(current_node, 0)
        last_load_ratio = snapshot.last_load_ratio(edge_id, edge.capacity)
        recent_load = max(1, int(round(last_load_ratio * max(1, edge.capacity))))
        recent_mean, recent_var = self.env.expected_cost_statistics(edge_id, recent_load)
        return np.asarray(
            [
                edge.base_cost,
                edge.congestion_coef,
                edge.variance_base,
                edge.variance_coef,
                edge.incident_prob,
                node_count / max(1, self.env.num_agents),
                last_load_ratio,
                1.0 if snapshot.last_incidents.get(edge_id, False) else 0.0,
                time_step / max(1, self.env.config.horizon),
                risk_alpha,
                risk_alpha * edge.variance_base,
                risk_alpha * edge.variance_coef,
                risk_alpha * edge.incident_prob,
                edge.base_cost + 0.5 * risk_alpha * edge.variance_base,
                recent_mean + 0.5 * risk_alpha * recent_var,
            ],
            dtype=np.float32,
        )

    def _synthetic_observation(self, current_node: str, risk_alpha: float, snapshot: PublicStateSnapshot, time_step: int) -> LocalObservation:
        feasible = self.env.graph.outgoing.get(current_node, ())
        edge_feature_map = {
            eid: self._edge_feature_vector_for_time(eid, snapshot, current_node, risk_alpha, time_step)
            for eid in feasible
        }
        done = current_node == self.env.graph.target or time_step >= self.env.config.horizon
        return LocalObservation(
            agent_id=-1,
            time_step=time_step,
            current_node=current_node,
            risk_alpha=risk_alpha,
            feasible_action_ids=tuple(feasible),
            edge_feature_map=edge_feature_map,
            public_snapshot=snapshot,
            done=done,
        )

    def _policy_probs_for_synthetic_obs(self, obs: LocalObservation, policy_mode: str) -> np.ndarray:
        if policy_mode == "average":
            probs = self.average_policy_probs(obs)
        else:
            probs = self.current_policy_probs(obs)[0]
        return _safe_probs_np(probs)

    def _future_load_guess(self, edge_id: int, snapshot: PublicStateSnapshot) -> int:
        edge = self.env.graph.edges[edge_id]
        recent = float(snapshot.last_edge_loads.get(edge_id, 0))
        if recent <= 0.0:
            src_occ = float(snapshot.node_counts.get(edge.src, 0))
            out_deg = max(1.0, float(len(self.env.graph.outgoing.get(edge.src, ()))))
            recent = src_occ / out_deg
        recent = max(float(self.cfg.target_future_load_floor), recent)
        return int(max(1, min(edge.capacity, round(recent))))

    def _heuristic_future_step_cost(self, edge_id: int, snapshot: PublicStateSnapshot, risk_alpha: float) -> float:
        est_load = self._future_load_guess(edge_id, snapshot)
        return self.env.effective_cost(edge_id, est_load, risk_alpha)

    def _transition_distribution(self, current_node: str, edge_id: int) -> Tuple[Tuple[float, str], ...]:
        edge = self.env.graph.edges[edge_id]
        p = float(max(0.0, min(1.0, edge.incident_prob)))
        if edge.failure_mode == "stay" and p > 0.0:
            return ((1.0 - p, edge.dst), (p, current_node))
        if edge.failure_mode == "fallback" and edge.fallback_edge_id is not None and p > 0.0:
            fb_dst = self.env.graph.edges[edge.fallback_edge_id].dst
            return ((1.0 - p, edge.dst), (p, fb_dst))
        return ((1.0, edge.dst),)

    def _policy_continuation_cost(
        self,
        node: str,
        time_step: int,
        remaining_steps: int,
        snapshot: PublicStateSnapshot,
        risk_alpha: float,
        policy_mode: str,
        cache: Dict[Tuple[str, int, int, str], float],
    ) -> float:
        if node == self.env.graph.target or remaining_steps <= 0 or time_step >= self.env.config.horizon:
            return 0.0
        key = (node, int(time_step), int(remaining_steps), str(policy_mode))
        if key in cache:
            return cache[key]
        obs = self._synthetic_observation(node, risk_alpha, snapshot, time_step)
        if obs.done or not obs.feasible_action_ids:
            cache[key] = 0.0
            return 0.0
        probs = self._policy_probs_for_synthetic_obs(obs, policy_mode=policy_mode)
        total = 0.0
        for act_idx, edge_id in enumerate(obs.feasible_action_ids):
            act_prob = float(probs[act_idx]) if act_idx < len(probs) else 0.0
            if act_prob <= 1e-12:
                continue
            step_cost = self._heuristic_future_step_cost(edge_id, snapshot, risk_alpha)
            downstream = 0.0
            for trans_prob, next_node in self._transition_distribution(node, edge_id):
                downstream += float(trans_prob) * self._policy_continuation_cost(
                    next_node,
                    time_step=time_step + 1,
                    remaining_steps=remaining_steps - 1,
                    snapshot=snapshot,
                    risk_alpha=risk_alpha,
                    policy_mode=policy_mode,
                    cache=cache,
                )
            total += act_prob * (step_cost + downstream)
        cache[key] = float(total)
        return float(total)

    def counterfactual_total_effective_cost(self, record: DecisionRecord, candidate_edge_id: int, policy_mode: str | None = None) -> float:
        immediate = self.env.counterfactual_effective_cost(
            agent_id=record.agent_id,
            candidate_edge_id=candidate_edge_id,
            positions_before=record.positions_before,
            joint_actions=record.joint_actions,
        )
        remaining_budget = max(0, self.env.config.horizon - (record.time_step + 1))
        lookahead = min(max(0, self.cfg.target_lookahead_steps), remaining_budget)
        if lookahead <= 0:
            return float(immediate)
        policy_mode = str(policy_mode or self.cfg.target_policy_mode)
        cache: Dict[Tuple[str, int, int, str], float] = {}
        continuation = 0.0
        for trans_prob, next_node in self._transition_distribution(record.current_node, candidate_edge_id):
            continuation += float(trans_prob) * self._policy_continuation_cost(
                next_node,
                time_step=record.time_step + 1,
                remaining_steps=lookahead,
                snapshot=record.public_snapshot,
                risk_alpha=record.risk_alpha,
                policy_mode=policy_mode,
                cache=cache,
            )
        return float(immediate + continuation)

    def add_regret_records(self, records: Sequence[DecisionRecord], iter_weight: float) -> None:
        for record in records:
            if record.action_id is None or not record.feasible_action_ids:
                continue
            current_node = record.current_node
            snapshot = record.public_snapshot
            risk_alpha = record.risk_alpha
            edge_map = {eid: self.env._edge_feature_vector(eid, snapshot, current_node, risk_alpha) for eid in record.feasible_action_ids}
            obs = LocalObservation(
                agent_id=record.agent_id,
                time_step=record.time_step,
                current_node=current_node,
                risk_alpha=risk_alpha,
                feasible_action_ids=tuple(record.feasible_action_ids),
                edge_feature_map=edge_map,
                public_snapshot=snapshot,
                done=False,
            )
            node_idx, edge_mat = self.obs_to_arrays(obs)
            chosen_cost = self.counterfactual_total_effective_cost(record, record.action_id, policy_mode=self.cfg.target_policy_mode)
            targets = []
            for alt_action in record.feasible_action_ids:
                alt_cost = self.counterfactual_total_effective_cost(record, alt_action, policy_mode=self.cfg.target_policy_mode)
                targets.append(float(chosen_cost - alt_cost))
            risk_bin = int(record.info_state_key.risk_bin) if record.info_state_key is not None else 0
            self.regret_buffer.add(
                TeacherRegretSample(
                    current_node_idx=node_idx,
                    feasible_action_ids=tuple(record.feasible_action_ids),
                    edge_features=edge_mat.copy(),
                    regret_targets=np.asarray(targets, dtype=np.float32).copy(),
                    iter_weight=float(iter_weight),
                    risk_bin=risk_bin,
                )
            )

    def _train_regret_model(self) -> Dict[str, float]:
        if len(self.regret_buffer) < self.cfg.min_buffer_before_train:
            return {"deep_regret_loss": 0.0, "deep_regret_buffer_size": float(len(self.regret_buffer))}
        if self.cfg.retrain_from_scratch:
            self._reset_regret_model()
        losses: List[float] = []
        self.regret_model.train()
        for _ in range(self.cfg.train_steps):
            batch = self.regret_buffer.sample_stratified_by_attr(self.cfg.batch_size, "risk_bin")
            if not batch:
                break
            batch_losses = []
            weight_sum = 0.0
            for sample in batch:
                assert isinstance(sample, TeacherRegretSample)
                edge_tensor = torch.tensor(sample.edge_features, dtype=torch.float32, device=self.device)
                pred = self.regret_model.action_scores(edge_tensor, sample.current_node_idx)
                target = torch.tensor(sample.regret_targets, dtype=torch.float32, device=self.device)
                w = float(sample.iter_weight)
                batch_losses.append(((pred - target) ** 2).mean() * w)
                weight_sum += w
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).sum() / max(1e-8, weight_sum)
            self.regret_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.regret_model.parameters(), 1.0)
            self.regret_optimizer.step()
            losses.append(float(loss.item()))
        self.regret_trained = True if losses else self.regret_trained
        return {
            "deep_regret_loss": float(np.mean(losses)) if losses else 0.0,
            "deep_regret_buffer_size": float(len(self.regret_buffer)),
        }

    def _train_strategy_model(self) -> Dict[str, float]:
        if len(self.strategy_buffer) < self.cfg.min_buffer_before_train:
            return {"deep_strategy_loss": 0.0, "deep_strategy_buffer_size": float(len(self.strategy_buffer))}
        if self.cfg.retrain_from_scratch:
            self._reset_strategy_model()
        losses: List[float] = []
        self.strategy_model.train()
        for _ in range(self.cfg.strategy_train_steps):
            batch = self.strategy_buffer.sample_stratified_by_attr(self.cfg.batch_size, "risk_bin")
            if not batch:
                break
            batch_losses = []
            weight_sum = 0.0
            for sample in batch:
                assert isinstance(sample, TeacherStrategySample)
                edge_tensor = torch.tensor(sample.edge_features, dtype=torch.float32, device=self.device)
                logits = self.strategy_model.action_scores(edge_tensor, sample.current_node_idx)
                log_probs = torch.log_softmax(logits, dim=0)
                target = torch.tensor(sample.policy_targets, dtype=torch.float32, device=self.device)
                w = float(sample.iter_weight)
                batch_losses.append((-(target * log_probs).sum()) * w)
                weight_sum += w
            if not batch_losses:
                continue
            loss = torch.stack(batch_losses).sum() / max(1e-8, weight_sum)
            self.strategy_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.strategy_model.parameters(), 1.0)
            self.strategy_optimizer.step()
            losses.append(float(loss.item()))
        self.strategy_trained = True if losses else self.strategy_trained
        return {
            "deep_strategy_loss": float(np.mean(losses)) if losses else 0.0,
            "deep_strategy_buffer_size": float(len(self.strategy_buffer)),
        }

    def train_from_buffers(self) -> Dict[str, float]:
        stats = {}
        stats.update(self._train_regret_model())
        stats.update(self._train_strategy_model())
        stats["deep_teacher_regret_trained"] = float(self.regret_trained)
        stats["deep_teacher_strategy_trained"] = float(self.strategy_trained)
        return stats

    def summary(self) -> Dict[str, float]:
        return {
            "deep_regret_buffer_size": float(len(self.regret_buffer)),
            "deep_strategy_buffer_size": float(len(self.strategy_buffer)),
            "deep_teacher_regret_trained": float(self.regret_trained),
            "deep_teacher_strategy_trained": float(self.strategy_trained),
        }


def train_deep_teacher_batch(env: SDNCGEnv, teacher: DeepCFRStyleTeacher, cfg: DeepTeacherConfig, seed_offset: int, update_idx: int) -> Dict[str, float]:
    stats = {
        "episodes": 0.0,
        "train_realized_cost": 0.0,
        "train_effective_cost": 0.0,
        "train_decisions": 0.0,
        "teacher_regret_updates": 0.0,
        "teacher_positive_regret_mass": 0.0,
        "teacher_max_positive_increment": 0.0,
    }
    for ep in range(cfg.episodes_per_update):
        env.reset(seed_offset + ep)
        total_realized = 0.0
        total_effective = 0.0
        total_decisions = 0
        while env.time_step < env.config.horizon and not all(env.terminated.values()):
            observations = env.get_observations()
            action_map: Dict[int, int] = {}
            for aid, obs in observations.items():
                if obs.done or not obs.feasible_action_ids:
                    continue
                action_id, _, probs, _ = teacher.select_action(obs, policy_mode="current", deterministic=False)
                teacher.add_strategy_sample(obs, probs, iter_weight=float(update_idx))
                action_map[aid] = action_id
            if not action_map:
                break
            step_result = env.step(action_map)
            records = env.build_decision_records(step_result)
            teacher.add_regret_records(records, iter_weight=float(update_idx))
            for record in records:
                if record.action_id is None:
                    continue
                chosen_cost = teacher.counterfactual_total_effective_cost(record, record.action_id, policy_mode=teacher.cfg.target_policy_mode)
                for alt_action in record.feasible_action_ids:
                    alt_cost = teacher.counterfactual_total_effective_cost(record, alt_action, policy_mode=teacher.cfg.target_policy_mode)
                    inc = float(chosen_cost - alt_cost)
                    stats["teacher_regret_updates"] += 1.0
                    if inc > 0.0:
                        stats["teacher_positive_regret_mass"] += inc
                        stats["teacher_max_positive_increment"] = max(stats["teacher_max_positive_increment"], inc)
            total_realized += sum(out.realized_cost for out in step_result.outcomes.values())
            total_effective += sum(out.effective_cost for out in step_result.outcomes.values())
            total_decisions += len(step_result.joint_actions)
        stats["episodes"] += 1.0
        stats["train_realized_cost"] += total_realized
        stats["train_effective_cost"] += total_effective
        stats["train_decisions"] += total_decisions
    if stats["episodes"] > 0:
        stats["train_realized_cost"] /= stats["episodes"]
        stats["train_effective_cost"] /= stats["episodes"]
        stats["train_decisions"] /= stats["episodes"]
        stats["teacher_regret_updates"] /= stats["episodes"]
        stats["teacher_positive_regret_mass"] /= stats["episodes"]
    stats.update(teacher.train_from_buffers())
    stats.update(teacher.summary())
    return stats


def evaluate_deep_teacher(env: SDNCGEnv, teacher: DeepCFRStyleTeacher, cfg: EvalConfig, eval_episodes: int, seed_offset: int, policy_mode: str = "average", deterministic: bool = False) -> Dict[str, float]:
    episode_realized: List[float] = []
    episode_effective: List[float] = []
    decision_records: List[DecisionRecord] = []
    confs: List[float] = []
    risk_values = _known_risk_values(env)
    episode_realized_by_risk: Dict[str, List[float]] = {_risk_label(alpha): [] for alpha in risk_values}
    episode_effective_by_risk: Dict[str, List[float]] = {_risk_label(alpha): [] for alpha in risk_values}
    for ep in range(eval_episodes):
        env.reset(seed_offset + ep)
        total_realized = 0.0
        total_effective = 0.0
        ep_realized_by_risk, ep_effective_by_risk = _empty_risk_episode_maps(env)
        while env.time_step < env.config.horizon and not all(env.terminated.values()):
            observations = env.get_observations()
            action_map: Dict[int, int] = {}
            for aid, obs in observations.items():
                if obs.done or not obs.feasible_action_ids:
                    continue
                confs.append(teacher.teacher_confidence(obs, policy_mode=policy_mode))
                action_id, _, _, _ = teacher.select_action(obs, policy_mode=policy_mode, deterministic=deterministic)
                action_map[aid] = action_id
            if not action_map:
                break
            step_result = env.step(action_map)
            decision_records.extend(env.build_decision_records(step_result))
            total_realized += sum(out.realized_cost for out in step_result.outcomes.values())
            total_effective += sum(out.effective_cost for out in step_result.outcomes.values())
            for aid, out in step_result.outcomes.items():
                if out.chosen_edge_id is None:
                    continue
                label = _risk_label(env.agent_specs[aid].risk_alpha)
                ep_realized_by_risk[label] = ep_realized_by_risk.get(label, 0.0) + float(out.realized_cost)
                ep_effective_by_risk[label] = ep_effective_by_risk.get(label, 0.0) + float(out.effective_cost)
        episode_realized.append(total_realized)
        episode_effective.append(total_effective)
        for label in episode_realized_by_risk:
            episode_realized_by_risk[label].append(float(ep_realized_by_risk.get(label, 0.0)))
            episode_effective_by_risk[label].append(float(ep_effective_by_risk.get(label, 0.0)))
    ce_diag = CEEvaluator(env, cfg.epsilon, cfg.stability_threshold, cfg.trailing_window).evaluate(decision_records)
    result = {
        "eval_realized_cost_mean": float(np.mean(episode_realized)) if episode_realized else 0.0,
        "eval_realized_cost_std": float(np.std(episode_realized, ddof=1)) if len(episode_realized) > 1 else 0.0,
        "eval_effective_cost_mean": float(np.mean(episode_effective)) if episode_effective else 0.0,
        "eval_effective_cost_std": float(np.std(episode_effective, ddof=1)) if len(episode_effective) > 1 else 0.0,
        "ce_mean_positive_regret": ce_diag.mean_positive_regret,
        "ce_max_agent_regret": ce_diag.max_agent_regret,
        "ce_ucb95": ce_diag.ucb95,
        "ce_stability_band": ce_diag.stability_band,
        "epsilon_ce_ex_post": float(ce_diag.epsilon_ce_ex_post),
        "num_eval_decisions": float(ce_diag.num_decisions),
        "teacher_confidence_mean": float(np.mean(confs)) if confs else 0.0,
    }
    result.update(_risk_diagnostic_fields(env, cfg, decision_records, episode_realized_by_risk, episode_effective_by_risk))
    return result


def run_single_seed_deep_teacher(seed: int, agent_count: int, widths: Sequence[int], topology: str, base_capacity: int, risk_alphas: Sequence[float], env_cfg: EnvConfig, encoder_cfg: EncoderConfig, deep_teacher_cfg: DeepTeacherConfig, eval_cfg: EvalConfig, device: torch.device, logger: RunLogger) -> List[Dict[str, float]]:
    set_global_seed(seed)
    env_cfg_local = EnvConfig(
        num_agents=agent_count,
        horizon=env_cfg.horizon,
        cost_clip=env_cfg.cost_clip,
        reward_mode=env_cfg.reward_mode,
        shared_edge_incident=env_cfg.shared_edge_incident,
        normal_std_floor=env_cfg.normal_std_floor,
        random_seed=seed,
    )
    env = build_env(seed, widths, topology, base_capacity, env_cfg_local, encoder_cfg, risk_alphas, num_agents=agent_count)
    teacher = DeepCFRStyleTeacher(env, deep_teacher_cfg, device=device, seed=seed)
    rows: List[Dict[str, float]] = []
    risk_cycle_label = "|".join(f"{x:.2f}" for x in risk_alphas)
    progress_log_every = max(1, deep_teacher_cfg.total_updates // 4)
    for update_idx in range(1, deep_teacher_cfg.total_updates + 1):
        train_stats = train_deep_teacher_batch(
            env,
            teacher,
            deep_teacher_cfg,
            seed_offset=seed * 100_000 + update_idx * 1_000,
            update_idx=update_idx,
        )
        if update_idx == 1 or update_idx % deep_teacher_cfg.eval_every == 0 or update_idx == deep_teacher_cfg.total_updates:
            eval_stats = evaluate_deep_teacher(
                env,
                teacher,
                eval_cfg,
                eval_episodes=deep_teacher_cfg.eval_episodes,
                seed_offset=seed * 1_000_000 + update_idx * 10_000,
                policy_mode=deep_teacher_cfg.eval_policy,
                deterministic=False,
            )
            row = {
                "seed": float(seed),
                "seed_label": f"seed_{seed}",
                "agent_count": float(agent_count),
                "agent_count_label": f"N_{agent_count}",
                "update": float(update_idx),
                "risk_cycle_label": risk_cycle_label,
                **train_stats,
                **eval_stats,
                "loss": 0.0,
                "policy_loss": 0.0,
                "value_loss": 0.0,
                "entropy": 0.0,
                "distill_loss": 0.0,
                "distill_active_fraction": 0.0,
            }
            rows.append(row)
            if update_idx == 1 or update_idx == deep_teacher_cfg.total_updates or update_idx % progress_log_every == 0:
                logger.summary(
                    "[deepcfr] N={n:>2d} seed={s} update={u:4d} ce={ce:.4f} ucb95={ucb:.4f} cost={cost:.2f} conf={conf:.3f} rbuf={rbuf:.0f}".format(
                        n=agent_count,
                        s=seed,
                        u=update_idx,
                        ce=row["ce_mean_positive_regret"],
                        ucb=row["ce_ucb95"],
                        cost=row["eval_realized_cost_mean"],
                        conf=row.get("teacher_confidence_mean", 0.0),
                        rbuf=row.get("deep_regret_buffer_size", 0.0),
                    )
                )
    return rows


@dataclass(frozen=True)
class HybridConfig:
    teacher_warmup_updates: int = 10
    teacher_policy_mode: str = "average"
    behavior_beta_start: float = 0.65
    behavior_beta_end: float = 0.00
    # Minimum effective teacher-mixing weight during guided stages.
    # This prevents nominal guided runs from degenerating into PPO-only when
    # confidence/JS gates are overly conservative. The floor is capped by
    # the scheduled base beta, so it still decays to zero for final PPO fine-tuning.
    behavior_beta_floor: float = 0.03
    mix_conf_threshold: float = 0.03
    # Evolved guidance: behavior mixing uses a scaled confidence gate rather
    # than raw confidence, so moderately confident teachers can actually shape
    # exploration.  A small JS gate suppresses teacher forcing when student and
    # teacher are already almost identical.
    mix_conf_scale: float = 0.20
    mix_js_threshold: float = 0.001
    mix_js_scale: float = 0.02
    mix_min_js_gate: float = 0.25
    # Conservative safe KL: direct imitation is only used when the teacher is
    # confident, sufficiently different from the student, and has lower
    # counterfactual expected cost on the sampled snapshot.
    distill_lambda_max: float = 0.02
    distill_temperature: float = 2.5
    distill_conf_threshold: float = 0.05
    distill_conf_scale: float = 0.20
    distill_gap_threshold: float = 0.002
    distill_gap_scale: float = 0.02
    distill_cf_advantage_threshold: float = 0.0
    distill_cf_advantage_scale: float = 0.005
    distill_ramp_updates: int = 600
    distill_min_branching_actions: int = 2

    # Adaptive best-teacher snapshot selection. Instead of freezing at a fixed
    # hard-coded update, the hybrid tracks validation CE-UCB and uses the best
    # teacher snapshot once it is sufficiently ready/stable or the maximum
    # teacher-pretraining budget is reached.
    adaptive_snapshot: bool = True
    teacher_freeze_min_updates: int = 1000
    teacher_freeze_max_updates: int = 1800
    teacher_ready_ce_ucb_threshold: float = 0.0068
    teacher_ready_stability_window: int = 3
    teacher_ready_stability_delta: float = 0.0010
    teacher_ready_min_confidence: float = 0.03
    distill_decay_start_fraction: float = 0.30


@dataclass
class GuidedPPOSample:
    current_node_idx: int
    feasible_action_ids: Tuple[int, ...]
    edge_features: np.ndarray
    action_index: int
    old_logprob: float
    value: float
    reward: float
    done: bool
    agent_id: int
    time_step: int
    info_state_key: InfoStateKey | None
    teacher_probs: np.ndarray
    student_probs: np.ndarray
    behavior_beta: float
    teacher_confidence: float
    teacher_student_js: float
    teacher_cf_advantage: float
    is_branching: bool
    distill_gate: float
    advantage: float = 0.0
    return_: float = 0.0


class GuidedPPOAgent(PPOAgent):
    @torch.no_grad()
    def policy_probs_and_value(self, obs: LocalObservation) -> Tuple[int, np.ndarray, np.ndarray, float]:
        node_idx, edge_mat = self.obs_to_arrays(obs)
        edge_tensor = torch.tensor(edge_mat, dtype=torch.float32, device=self.device)
        logits = self.model.action_logits(edge_tensor, node_idx)
        probs = torch.softmax(logits, dim=0).detach().cpu().numpy().astype(np.float64, copy=False)
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        total = float(np.sum(probs))
        if total <= 1e-12:
            probs = np.full(len(obs.feasible_action_ids), 1.0 / max(1, len(obs.feasible_action_ids)), dtype=np.float64)
        else:
            probs = probs / total
        value = float(self.model.state_value(edge_tensor, node_idx).item())
        return node_idx, edge_mat.astype(np.float32), probs, value

    def evaluate_guided_sample(self, sample: GuidedPPOSample) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        edge_tensor = torch.tensor(sample.edge_features, dtype=torch.float32, device=self.device)
        logits = self.model.action_logits(edge_tensor, sample.current_node_idx)
        student_probs = torch.softmax(logits, dim=0)
        student_probs = torch.clamp(student_probs, min=1e-8)
        student_probs = student_probs / student_probs.sum()
        if sample.teacher_probs.size == len(sample.feasible_action_ids) and sample.behavior_beta > 0.0:
            teacher_probs = torch.tensor(sample.teacher_probs, dtype=torch.float32, device=self.device)
            teacher_probs = torch.clamp(teacher_probs, min=1e-8)
            teacher_probs = teacher_probs / teacher_probs.sum()
            behavior_probs = (1.0 - float(sample.behavior_beta)) * student_probs + float(sample.behavior_beta) * teacher_probs
            behavior_probs = torch.clamp(behavior_probs, min=1e-8)
            behavior_probs = behavior_probs / behavior_probs.sum()
        else:
            behavior_probs = student_probs
        action_idx = torch.tensor(sample.action_index, dtype=torch.long, device=self.device)
        # PPO ratio is computed on the same behavior mixture that generated
        # the action. The teacher distribution in each sample is fixed, so
        # gradients still flow only through the student component. When beta
        # decays to zero, this reduces exactly to vanilla PPO on the student.
        new_logprob = torch.log(torch.clamp(behavior_probs[action_idx], min=1e-8))
        entropy = torch.distributions.Categorical(probs=student_probs).entropy()
        value = self.model.state_value(edge_tensor, sample.current_node_idx)
        return logits, student_probs, behavior_probs, new_logprob, entropy, value

    def update_guided(self, samples: Sequence[GuidedPPOSample], distill_coef: float, distill_temperature: float, use_distill: bool) -> Dict[str, float]:
        if not samples:
            return {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "distill_loss": 0.0, "distill_active_fraction": 0.0}
        advantages = np.asarray([s.advantage for s in samples], dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        returns = np.asarray([s.return_ for s in samples], dtype=np.float32)
        old_logprobs = np.asarray([s.old_logprob for s in samples], dtype=np.float32)
        indices = np.arange(len(samples))
        meters = {"loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "distill_loss": 0.0, "distill_active_fraction": 0.0}
        num_steps = 0
        temp = max(1e-6, float(distill_temperature))
        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), self.cfg.minibatch_size):
                mb_idx = indices[start:start + self.cfg.minibatch_size]
                if len(mb_idx) == 0:
                    continue
                policy_terms = []
                value_terms = []
                entropy_terms = []
                distill_terms = []
                active_gates = []
                for idx in mb_idx:
                    sample = samples[idx]
                    logits, student_probs, _, new_logprob, entropy, value = self.evaluate_guided_sample(sample)
                    ratio = torch.exp(new_logprob - torch.tensor(old_logprobs[idx], dtype=torch.float32, device=self.device))
                    adv = torch.tensor(advantages[idx], dtype=torch.float32, device=self.device)
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - self.cfg.clip_ratio, 1.0 + self.cfg.clip_ratio) * adv
                    policy_terms.append(-torch.min(unclipped, clipped))
                    target_return = torch.tensor(returns[idx], dtype=torch.float32, device=self.device)
                    value_terms.append((value - target_return) ** 2)
                    entropy_terms.append(entropy)
                    gate = float(sample.distill_gate) if use_distill and distill_coef > 0.0 else 0.0
                    active_gates.append(1.0 if gate > 1e-8 else 0.0)
                    if gate > 1e-8 and sample.teacher_probs.size == len(sample.feasible_action_ids):
                        teacher_probs = torch.tensor(sample.teacher_probs, dtype=torch.float32, device=self.device)
                        teacher_probs = torch.clamp(teacher_probs, min=1e-8)
                        teacher_probs = teacher_probs / teacher_probs.sum()
                        teacher_soft = torch.pow(teacher_probs, 1.0 / temp)
                        teacher_soft = torch.clamp(teacher_soft, min=1e-8)
                        teacher_soft = teacher_soft / teacher_soft.sum()
                        student_log_soft = torch.log_softmax(logits / temp, dim=0)
                        kl = torch.sum(teacher_soft * (torch.log(torch.clamp(teacher_soft, min=1e-8)) - student_log_soft))
                        distill_terms.append(torch.tensor(gate, dtype=torch.float32, device=self.device) * (temp ** 2) * kl)
                    else:
                        distill_terms.append(torch.tensor(0.0, dtype=torch.float32, device=self.device))
                policy_loss = torch.stack(policy_terms).mean()
                value_loss = torch.stack(value_terms).mean()
                entropy = torch.stack(entropy_terms).mean()
                distill_loss = torch.stack(distill_terms).mean() if distill_terms else torch.tensor(0.0, dtype=torch.float32, device=self.device)
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy + float(distill_coef) * distill_loss
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()
                meters["loss"] += float(loss.item())
                meters["policy_loss"] += float(policy_loss.item())
                meters["value_loss"] += float(value_loss.item())
                meters["entropy"] += float(entropy.item())
                meters["distill_loss"] += float(distill_loss.item())
                meters["distill_active_fraction"] += float(np.mean(active_gates)) if active_gates else 0.0
                num_steps += 1
        if num_steps > 0:
            for key in meters:
                meters[key] /= num_steps
        return meters


def compute_gae_guided(trajectory: List[GuidedPPOSample], gamma: float, gae_lambda: float) -> None:
    gae = 0.0
    next_value = 0.0
    for sample in reversed(trajectory):
        mask = 0.0 if sample.done else 1.0
        delta = sample.reward + gamma * next_value * mask - sample.value
        gae = delta + gamma * gae_lambda * mask * gae
        sample.advantage = float(gae)
        sample.return_ = float(sample.value + sample.advantage)
        next_value = sample.value


def _safe_probs_np(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs, dtype=np.float64)
    probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(probs))
    if probs.size == 0:
        return probs
    if total <= 1e-12:
        return np.full(len(probs), 1.0 / max(1, len(probs)), dtype=np.float64)
    return probs / total


def _teacher_confidence_from_probs(probs: np.ndarray) -> float:
    probs = _safe_probs_np(probs)
    if probs.size <= 1:
        return 0.0
    ent = float(-(probs * np.log(np.clip(probs, 1e-8, 1.0))).sum())
    max_ent = math.log(float(len(probs)))
    ent_conf = max(0.0, 1.0 - ent / max(1e-8, max_ent))
    sorted_probs = np.sort(probs)
    top1 = float(sorted_probs[-1])
    top2 = float(sorted_probs[-2]) if probs.size >= 2 else 0.0
    margin_conf = max(0.0, min(1.0, top1 - top2))
    return float(0.5 * ent_conf + 0.5 * margin_conf)


def _js_divergence_np(p: np.ndarray, q: np.ndarray) -> float:
    p = _safe_probs_np(p)
    q = _safe_probs_np(q)
    if p.size == 0 or q.size == 0 or p.size != q.size:
        return 0.0
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * (np.log(np.clip(p, 1e-8, 1.0)) - np.log(np.clip(m, 1e-8, 1.0)))))
    kl_qm = float(np.sum(q * (np.log(np.clip(q, 1e-8, 1.0)) - np.log(np.clip(m, 1e-8, 1.0)))))
    return float(max(0.0, 0.5 * (kl_pm + kl_qm)))


def _soft_gate(value: float, threshold: float) -> float:
    if value <= threshold:
        return 0.0
    numer = value - threshold
    denom = max(1e-8, 1.0 - threshold)
    return float(min(1.0, max(0.0, numer / denom)))


def _scaled_gate(value: float, threshold: float, scale: float) -> float:
    """Gate small diagnostic quantities such as JS and normalized CF advantage."""
    if value <= threshold:
        return 0.0
    return float(min(1.0, max(0.0, (value - threshold) / max(1e-8, scale))))


def _beta_schedule(cfg: HybridConfig, update_idx: int, total_updates: int) -> float:
    if total_updates <= 1:
        return float(cfg.behavior_beta_end)
    frac = min(1.0, max(0.0, (update_idx - 1) / max(1, total_updates - 1)))
    return float(cfg.behavior_beta_start + frac * (cfg.behavior_beta_end - cfg.behavior_beta_start))


def _distill_schedule(cfg: HybridConfig, update_idx: int, use_distill: bool) -> float:
    if not use_distill or update_idx <= cfg.teacher_warmup_updates:
        return 0.0
    ramp = min(1.0, max(0.0, (update_idx - cfg.teacher_warmup_updates) / max(1, cfg.distill_ramp_updates)))
    return float(cfg.distill_lambda_max * ramp)


def collect_guided_training_batch(env: SDNCGEnv, agent: GuidedPPOAgent, teacher: DeepCFRStyleTeacher, ppo_cfg: PPOConfig, hybrid_cfg: HybridConfig, seed_offset: int, update_idx: int, use_distill: bool) -> Tuple[List[GuidedPPOSample], Dict[str, float]]:
    rng = np.random.default_rng(seed_offset + 917)
    samples: List[GuidedPPOSample] = []
    base_beta = _beta_schedule(hybrid_cfg, update_idx, ppo_cfg.total_updates)
    stats = {
        "episodes": 0.0,
        "train_realized_cost": 0.0,
        "train_effective_cost": 0.0,
        "train_decisions": 0.0,
        "teacher_behavior_beta_base": float(base_beta),
        "teacher_behavior_beta_floor": float(hybrid_cfg.behavior_beta_floor),
        "teacher_behavior_beta_mean": 0.0,
        "teacher_beta_floor_applied_fraction": 0.0,
        "teacher_mix_active_fraction": 0.0,
        "teacher_confidence_mean": 0.0,
        "teacher_confidence_std": 0.0,
        "teacher_student_js_mean": 0.0,
        "branching_fraction": 0.0,
        "distill_gate_mean": 0.0,
        "distill_gate_active_fraction": 0.0,
        "distill_branching_active_fraction": 0.0,
        "teacher_cf_advantage_mean": 0.0,
        "teacher_cf_advantage_positive_fraction": 0.0,
    }
    mix_betas: List[float] = []
    beta_floor_flags: List[float] = []
    confidences: List[float] = []
    js_gaps: List[float] = []
    cf_advantages: List[float] = []
    gates: List[float] = []
    branch_flags: List[float] = []
    branch_gate_flags: List[float] = []
    for ep in range(ppo_cfg.episodes_per_update):
        episode_seed = seed_offset + ep
        env.reset(episode_seed)
        per_agent_traj: Dict[int, List[GuidedPPOSample]] = {aid: [] for aid in env.positions.keys()}
        total_realized = 0.0
        total_effective = 0.0
        total_decisions = 0
        while env.time_step < env.config.horizon and not all(env.terminated.values()):
            observations = env.get_observations()
            action_map: Dict[int, int] = {}
            meta: Dict[int, Tuple[int, float, float, np.ndarray, int, InfoStateKey | None, np.ndarray, np.ndarray, float, float, float, bool, float]] = {}
            for aid, obs in observations.items():
                if obs.done or not obs.feasible_action_ids:
                    continue
                node_idx, edge_mat, student_probs, value = agent.policy_probs_and_value(obs)
                teacher_probs = teacher.average_policy_probs(obs) if hybrid_cfg.teacher_policy_mode == "average" else teacher.current_policy_probs(obs)[0]
                teacher_probs = _safe_probs_np(teacher_probs)
                student_probs = _safe_probs_np(student_probs)
                branching = len(obs.feasible_action_ids) >= hybrid_cfg.distill_min_branching_actions
                confidence = float(teacher.teacher_confidence(obs, policy_mode=hybrid_cfg.teacher_policy_mode)) if branching else 0.0
                js_gap = _js_divergence_np(student_probs, teacher_probs) if branching else 0.0
                if update_idx <= hybrid_cfg.teacher_warmup_updates or not branching:
                    mix_gate = 0.0
                else:
                    conf_mix_gate = _scaled_gate(confidence, hybrid_cfg.mix_conf_threshold, hybrid_cfg.mix_conf_scale)
                    js_mix_gate = _scaled_gate(js_gap, hybrid_cfg.mix_js_threshold, hybrid_cfg.mix_js_scale)
                    mix_gate = float(conf_mix_gate * max(float(hybrid_cfg.mix_min_js_gate), js_mix_gate))
                beta_eff = float(base_beta * mix_gate)
                floor_applied = 0.0
                if update_idx > hybrid_cfg.teacher_warmup_updates and branching and base_beta > 1e-8 and hybrid_cfg.behavior_beta_floor > 0.0:
                    beta_floor = min(float(base_beta), float(hybrid_cfg.behavior_beta_floor))
                    if beta_eff < beta_floor:
                        beta_eff = beta_floor
                        floor_applied = 1.0
                beta_eff = float(np.clip(beta_eff, 0.0, 1.0))
                mix_betas.append(beta_eff)
                beta_floor_flags.append(floor_applied)
                confidences.append(confidence)
                js_gaps.append(js_gap)
                branch_flags.append(1.0 if branching else 0.0)
                behavior_probs = _safe_probs_np((1.0 - beta_eff) * student_probs + beta_eff * teacher_probs)
                action_index = int(rng.choice(len(obs.feasible_action_ids), p=behavior_probs))
                action_id = int(obs.feasible_action_ids[action_index])
                old_logprob = float(np.log(max(1e-8, behavior_probs[action_index])))
                info_state = env.info_state_encoder.encode(obs) if env.info_state_encoder is not None else None
                if use_distill and update_idx > hybrid_cfg.teacher_warmup_updates and branching:
                    conf_gate = _scaled_gate(confidence, hybrid_cfg.distill_conf_threshold, hybrid_cfg.distill_conf_scale)
                    gap_gate = _scaled_gate(js_gap, hybrid_cfg.distill_gap_threshold, hybrid_cfg.distill_gap_scale)
                    # Provisional gate. After the joint action is known, an
                    # additional counterfactual teacher-advantage gate is applied.
                    distill_gate = float(conf_gate * gap_gate)
                else:
                    distill_gate = 0.0
                action_map[aid] = action_id
                meta[aid] = (
                    action_index,
                    old_logprob,
                    float(value),
                    edge_mat.copy(),
                    node_idx,
                    info_state,
                    teacher_probs.copy(),
                    student_probs.copy(),
                    beta_eff,
                    confidence,
                    js_gap,
                    bool(branching),
                    float(distill_gate),
                )
            if not action_map:
                break
            step_result = env.step(action_map)
            total_realized += sum(out.realized_cost for out in step_result.outcomes.values())
            total_effective += sum(out.effective_cost for out in step_result.outcomes.values())
            total_decisions += len(step_result.joint_actions)
            for aid in step_result.joint_actions.keys():
                action_index, old_logprob, value, edge_mat, node_idx, info_state, teacher_probs, student_probs, beta_eff, confidence, js_gap, branching, distill_gate = meta[aid]
                feasible_ids = tuple(observations[aid].feasible_action_ids)
                teacher_cf_advantage = 0.0
                if use_distill and branching and distill_gate > 1e-12 and len(feasible_ids) == len(teacher_probs) == len(student_probs):
                    cf_costs = np.asarray([
                        env.counterfactual_effective_cost(
                            agent_id=aid,
                            candidate_edge_id=eid,
                            positions_before=step_result.positions_before,
                            joint_actions=step_result.joint_actions,
                        )
                        for eid in feasible_ids
                    ], dtype=np.float64)
                    teacher_exp_cf = float(np.dot(_safe_probs_np(teacher_probs), cf_costs))
                    student_exp_cf = float(np.dot(_safe_probs_np(student_probs), cf_costs))
                    teacher_cf_advantage = max(0.0, student_exp_cf - teacher_exp_cf) / max(1e-8, env.estimate_cost_normalizer())
                    cf_gate = _scaled_gate(teacher_cf_advantage, hybrid_cfg.distill_cf_advantage_threshold, hybrid_cfg.distill_cf_advantage_scale)
                    distill_gate = float(distill_gate * cf_gate)
                else:
                    distill_gate = 0.0 if use_distill else float(distill_gate)
                cf_advantages.append(float(teacher_cf_advantage))
                gates.append(float(distill_gate))
                branch_gate_flags.append(1.0 if (branching and distill_gate > 1e-8) else 0.0)
                per_agent_traj[aid].append(
                    GuidedPPOSample(
                        current_node_idx=node_idx,
                        feasible_action_ids=feasible_ids,
                        edge_features=edge_mat,
                        action_index=action_index,
                        old_logprob=float(old_logprob),
                        value=float(value),
                        reward=float(step_result.rewards[aid]),
                        done=bool(step_result.dones[aid]),
                        agent_id=aid,
                        time_step=observations[aid].time_step,
                        info_state_key=info_state,
                        teacher_probs=teacher_probs,
                        student_probs=student_probs,
                        behavior_beta=float(beta_eff),
                        teacher_confidence=float(confidence),
                        teacher_student_js=float(js_gap),
                        teacher_cf_advantage=float(teacher_cf_advantage),
                        is_branching=bool(branching),
                        distill_gate=float(distill_gate),
                    )
                )
        for traj in per_agent_traj.values():
            if traj:
                compute_gae_guided(traj, gamma=ppo_cfg.gamma, gae_lambda=ppo_cfg.gae_lambda)
                samples.extend(traj)
        stats["episodes"] += 1.0
        stats["train_realized_cost"] += total_realized
        stats["train_effective_cost"] += total_effective
        stats["train_decisions"] += total_decisions
    if stats["episodes"] > 0:
        stats["train_realized_cost"] /= stats["episodes"]
        stats["train_effective_cost"] /= stats["episodes"]
        stats["train_decisions"] /= stats["episodes"]
    stats["teacher_behavior_beta_mean"] = float(np.mean(mix_betas)) if mix_betas else 0.0
    stats["teacher_beta_floor_applied_fraction"] = float(np.mean(beta_floor_flags)) if beta_floor_flags else 0.0
    stats["teacher_mix_active_fraction"] = float(np.mean([1.0 if b > 1e-8 else 0.0 for b in mix_betas])) if mix_betas else 0.0
    stats["teacher_confidence_mean"] = float(np.mean(confidences)) if confidences else 0.0
    stats["teacher_confidence_std"] = float(np.std(confidences, ddof=1)) if len(confidences) > 1 else 0.0
    stats["teacher_student_js_mean"] = float(np.mean(js_gaps)) if js_gaps else 0.0
    stats["teacher_cf_advantage_mean"] = float(np.mean(cf_advantages)) if cf_advantages else 0.0
    stats["teacher_cf_advantage_positive_fraction"] = float(np.mean([1.0 if a > 1e-12 else 0.0 for a in cf_advantages])) if cf_advantages else 0.0
    stats["branching_fraction"] = float(np.mean(branch_flags)) if branch_flags else 0.0
    stats["distill_gate_mean"] = float(np.mean(gates)) if gates else 0.0
    stats["distill_gate_active_fraction"] = float(np.mean([1.0 if g > 1e-8 else 0.0 for g in gates])) if gates else 0.0
    stats["distill_branching_active_fraction"] = float(np.mean(branch_gate_flags)) if branch_gate_flags else 0.0
    return samples, stats




def clone_deep_teacher_snapshot(teacher: DeepCFRStyleTeacher) -> Dict[str, object]:
    """Clone the inference state needed by the teacher reference policy."""
    return {
        "regret_model": {k: v.detach().cpu().clone() for k, v in teacher.regret_model.state_dict().items()},
        "strategy_model": {k: v.detach().cpu().clone() for k, v in teacher.strategy_model.state_dict().items()},
        "regret_trained": bool(teacher.regret_trained),
        "strategy_trained": bool(teacher.strategy_trained),
    }


def load_deep_teacher_snapshot(teacher: DeepCFRStyleTeacher, snapshot: Mapping[str, object]) -> None:
    regret_state = {k: v.to(teacher.device) for k, v in snapshot["regret_model"].items()}  # type: ignore[index, union-attr]
    strategy_state = {k: v.to(teacher.device) for k, v in snapshot["strategy_model"].items()}  # type: ignore[index, union-attr]
    teacher.regret_model.load_state_dict(regret_state)
    teacher.strategy_model.load_state_dict(strategy_state)
    teacher.regret_trained = bool(snapshot.get("regret_trained", False))
    teacher.strategy_trained = bool(snapshot.get("strategy_trained", False))
    teacher.regret_model.eval()
    teacher.strategy_model.eval()


def _risk_aware_teacher_eval_metric(eval_stats: Mapping[str, float]) -> float:
    """Metric used for teacher snapshot selection.

    We use worst-risk CE-UCB when per-risk fields are available. This avoids
    selecting a teacher that is good on average but bad for one risk type.
    """
    risk_ucbs = [float(v) for k, v in eval_stats.items() if k.startswith("risk_alpha_") and k.endswith("_ce_ucb95") and isinstance(v, (int, float, np.floating))]
    if risk_ucbs:
        return float(max(risk_ucbs))
    return float(eval_stats.get("ce_ucb95", 1e9))


def teacher_ready_from_history(history: Sequence[Mapping[str, float]], cfg: HybridConfig, update_idx: int) -> Tuple[bool, str]:
    if update_idx < cfg.teacher_freeze_min_updates:
        return False, "min_updates_not_reached"
    if not history:
        return False, "no_teacher_eval"
    last = history[-1]
    last_ucb = _risk_aware_teacher_eval_metric(last)
    last_conf = float(last.get("teacher_confidence_mean", 0.0))
    if last_ucb > cfg.teacher_ready_ce_ucb_threshold:
        return False, "risk_aware_ce_ucb_above_threshold"
    if last_conf < cfg.teacher_ready_min_confidence:
        return False, "confidence_below_threshold"
    window = max(1, int(cfg.teacher_ready_stability_window))
    if len(history) < window:
        return False, "stability_window_not_filled"
    recent = [_risk_aware_teacher_eval_metric(h) for h in history[-window:]]
    if max(recent) - min(recent) > cfg.teacher_ready_stability_delta:
        return False, "teacher_not_stable"
    return True, "ready"


def distill_coef_two_stage(cfg: HybridConfig, guided_step: int, guided_total: int, use_distill: bool) -> float:
    if not use_distill or guided_step <= 0 or guided_total <= 0:
        return 0.0
    progress = min(1.0, max(0.0, guided_step / max(1, guided_total)))
    ramp_frac = 0.15
    if progress < ramp_frac:
        scale = progress / ramp_frac
    elif progress < cfg.distill_decay_start_fraction:
        scale = 1.0
    else:
        denom = max(1e-8, 1.0 - cfg.distill_decay_start_fraction)
        scale = max(0.0, 1.0 - (progress - cfg.distill_decay_start_fraction) / denom)
    return float(cfg.distill_lambda_max * scale)


def prefix_stats(prefix: str, stats: Mapping[str, float]) -> Dict[str, float]:
    return {f"{prefix}{k}": float(v) for k, v in stats.items() if isinstance(v, (int, float, np.floating))}

def run_single_seed_hybrid(seed: int, mode_name: str, use_distill: bool, agent_count: int, widths: Sequence[int], topology: str, base_capacity: int, risk_alphas: Sequence[float], env_cfg: EnvConfig, encoder_cfg: EncoderConfig, ppo_cfg: PPOConfig, deep_teacher_cfg: DeepTeacherConfig, hybrid_cfg: HybridConfig, eval_cfg: EvalConfig, device: torch.device, logger: RunLogger) -> List[Dict[str, float]]:
    """Hybrid runner with adaptive best-teacher snapshot selection.

    Stage 1: teacher is trained and evaluated; the student still learns PPO.
    Stage 2: the best teacher snapshot is frozen and used for guidance.
    Stage 3: beta and KL decay, leaving a PPO fine-tuning tail.
    """
    set_global_seed(seed)
    env_cfg_local = EnvConfig(
        num_agents=agent_count,
        horizon=env_cfg.horizon,
        cost_clip=env_cfg.cost_clip,
        reward_mode=env_cfg.reward_mode,
        shared_edge_incident=env_cfg.shared_edge_incident,
        normal_std_floor=env_cfg.normal_std_floor,
        random_seed=seed,
    )
    env = build_env(seed, widths, topology, base_capacity, env_cfg_local, encoder_cfg, risk_alphas, num_agents=agent_count)
    teacher = DeepCFRStyleTeacher(env, deep_teacher_cfg, device=device, seed=seed)
    agent = GuidedPPOAgent(env, ppo_cfg, device)
    rows: List[Dict[str, float]] = []
    risk_cycle_label = "|".join(f"{x:.2f}" for x in risk_alphas)
    progress_log_every = max(1, ppo_cfg.total_updates // 4)

    teacher_history: List[Dict[str, float]] = []
    best_snapshot: Dict[str, object] | None = None
    best_snapshot_metric = float("inf")
    best_snapshot_update = 0
    teacher_frozen = False
    teacher_freeze_update = 0
    teacher_freeze_reason = "not_frozen"

    for update_idx in range(1, ppo_cfg.total_updates + 1):
        teacher_training_active = not teacher_frozen
        teacher_eval_current: Dict[str, float] | None = None

        if teacher_training_active:
            teacher_stats_raw = train_deep_teacher_batch(
                env, teacher, deep_teacher_cfg,
                seed_offset=seed * 100_000 + update_idx * 10_000,
                update_idx=update_idx,
            )
        else:
            teacher_stats_raw = {"teacher_training_episodes": 0.0, **teacher.summary()}

        should_eval_teacher_for_snapshot = (
            teacher_training_active
            and (update_idx == 1 or update_idx % ppo_cfg.eval_every == 0 or update_idx == ppo_cfg.total_updates or update_idx >= hybrid_cfg.teacher_freeze_max_updates)
        )
        if should_eval_teacher_for_snapshot:
            teacher_eval_current = evaluate_deep_teacher(
                env, teacher, eval_cfg,
                eval_episodes=deep_teacher_cfg.eval_episodes,
                seed_offset=seed * 20_000_000 + update_idx * 40_000,
                policy_mode=hybrid_cfg.teacher_policy_mode,
                deterministic=False,
            )
            metric = _risk_aware_teacher_eval_metric(teacher_eval_current)
            allow_snapshot_selection = update_idx >= hybrid_cfg.teacher_freeze_min_updates
            if allow_snapshot_selection:
                teacher_history.append(dict(teacher_eval_current))
            if allow_snapshot_selection and metric < best_snapshot_metric and bool(teacher.regret_trained or teacher.strategy_trained):
                best_snapshot_metric = metric
                best_snapshot_update = update_idx
                best_snapshot = clone_deep_teacher_snapshot(teacher)
            ready, ready_reason = teacher_ready_from_history(teacher_history, hybrid_cfg, update_idx)
            maxed = update_idx >= hybrid_cfg.teacher_freeze_max_updates
            if hybrid_cfg.adaptive_snapshot and (ready or maxed):
                if best_snapshot is None:
                    best_snapshot = clone_deep_teacher_snapshot(teacher)
                    best_snapshot_metric = metric
                    best_snapshot_update = update_idx
                load_deep_teacher_snapshot(teacher, best_snapshot)
                teacher_frozen = True
                teacher_freeze_update = update_idx
                teacher_freeze_reason = "ready" if ready else f"max_pretrain:{ready_reason}"

        if teacher_frozen:
            guided_step = max(1, update_idx - teacher_freeze_update + 1)
            guided_total = max(1, ppo_cfg.total_updates - teacher_freeze_update + 1)
            ppo_cfg_guided = replace(ppo_cfg, total_updates=guided_total)
            hybrid_cfg_guided = replace(hybrid_cfg, teacher_warmup_updates=0)
            current_lambda = distill_coef_two_stage(hybrid_cfg_guided, guided_step, guided_total, use_distill=use_distill)
            guided_samples, train_stats = collect_guided_training_batch(
                env, agent, teacher, ppo_cfg_guided, hybrid_cfg_guided,
                seed_offset=seed * 1_000_000 + update_idx * 20_000,
                update_idx=guided_step,
                use_distill=use_distill,
            )
            opt_stats = agent.update_guided(
                guided_samples,
                distill_coef=current_lambda,
                distill_temperature=hybrid_cfg_guided.distill_temperature,
                use_distill=use_distill,
            )
            beta_mean = float(train_stats.get("teacher_behavior_beta_mean", 0.0))
            if beta_mean <= 1e-5 and current_lambda <= 1e-8:
                hybrid_stage = "ppo_finetune"
            elif use_distill and current_lambda > 1e-8:
                hybrid_stage = "guided_distill"
            else:
                hybrid_stage = "guided_no_distill"
        else:
            guided_step = 0
            guided_total = max(1, ppo_cfg.total_updates)
            current_lambda = 0.0
            ppo_samples, train_stats = collect_training_batch(
                env, agent, ppo_cfg,
                seed_offset=seed * 1_000_000 + update_idx * 20_000,
            )
            opt_stats = agent.update(ppo_samples)
            train_stats = {
                **train_stats,
                "teacher_behavior_beta_base": 0.0,
                "teacher_behavior_beta_mean": 0.0,
                "teacher_mix_active_fraction": 0.0,
                "teacher_confidence_mean": 0.0,
                "teacher_confidence_std": 0.0,
                "teacher_student_js_mean": 0.0,
                "branching_fraction": 0.0,
                "distill_gate_mean": 0.0,
                "distill_gate_active_fraction": 0.0,
                "distill_branching_active_fraction": 0.0,
            }
            opt_stats = {**opt_stats, "distill_loss": 0.0, "distill_active_fraction": 0.0}
            hybrid_stage = "teacher_select_student_ppo"

        if update_idx == 1 or update_idx % ppo_cfg.eval_every == 0 or update_idx == ppo_cfg.total_updates:
            student_eval = evaluate_policy(
                env, agent, eval_cfg,
                eval_episodes=ppo_cfg.eval_episodes,
                seed_offset=seed * 10_000_000 + update_idx * 30_000,
                deterministic=False,
            )
            if teacher_eval_current is not None and not teacher_frozen:
                teacher_eval = teacher_eval_current
            else:
                teacher_eval = evaluate_deep_teacher(
                    env, teacher, eval_cfg,
                    eval_episodes=deep_teacher_cfg.eval_episodes,
                    seed_offset=seed * 20_000_000 + update_idx * 40_000,
                    policy_mode=hybrid_cfg.teacher_policy_mode,
                    deterministic=False,
                )
            row = {
                "seed": float(seed),
                "seed_label": f"seed_{seed}",
                "agent_count": float(agent_count),
                "agent_count_label": f"N_{agent_count}",
                "update": float(update_idx),
                "risk_cycle_label": risk_cycle_label,
                **train_stats,
                **opt_stats,
                **prefix_stats("teacher_train_", teacher_stats_raw),
                **student_eval,
                "teacher_eval_realized_cost_mean": float(teacher_eval.get("eval_realized_cost_mean", 0.0)),
                "teacher_eval_effective_cost_mean": float(teacher_eval.get("eval_effective_cost_mean", 0.0)),
                "teacher_eval_ce_mean_positive_regret": float(teacher_eval.get("ce_mean_positive_regret", 0.0)),
                "teacher_eval_ce_ucb95": float(teacher_eval.get("ce_ucb95", 0.0)),
                "teacher_eval_confidence_mean": float(teacher_eval.get("teacher_confidence_mean", 0.0)),
                **{f"teacher_eval_{k}": float(v) for k, v in teacher_eval.items() if k.startswith("risk_alpha_") and isinstance(v, (int, float, np.floating))},
                "teacher_best_ce_ucb95": float(best_snapshot_metric if math.isfinite(best_snapshot_metric) else 0.0),
                "teacher_best_snapshot_update": float(best_snapshot_update),
                "teacher_snapshot_frozen": float(1.0 if teacher_frozen else 0.0),
                "teacher_freeze_update": float(teacher_freeze_update),
                "teacher_training_active": float(1.0 if teacher_training_active else 0.0),
                "guided_step": float(guided_step),
                "guided_total": float(guided_total),
                "distill_lambda_current": float(current_lambda),
                "hybrid_use_distill": float(1.0 if use_distill else 0.0),
                "hybrid_stage": hybrid_stage,
                "teacher_freeze_reason": teacher_freeze_reason,
            }
            rows.append(row)
            if update_idx == 1 or update_idx == ppo_cfg.total_updates or update_idx % progress_log_every == 0:
                logger.summary(
                    "[{mode}] N={n:>2d} seed={s} update={u:4d} stage={stage} ce={ce:.4f} cost={cost:.2f} teacher_ce={tce:.4f} best_ucb={best:.4f}@{bu:.0f} frozen={frozen:.0f} beta={beta:.3f} lam={lam:.3f} conf={conf:.3f} js={js:.4f} cfadv={cfadv:.4f} d_active={dact:.3f}".format(
                        mode=mode_name,
                        n=agent_count,
                        s=seed,
                        u=update_idx,
                        stage=hybrid_stage,
                        ce=row["ce_mean_positive_regret"],
                        cost=row["eval_realized_cost_mean"],
                        tce=row["teacher_eval_ce_mean_positive_regret"],
                        best=row["teacher_best_ce_ucb95"],
                        bu=row["teacher_best_snapshot_update"],
                        frozen=row["teacher_snapshot_frozen"],
                        beta=row.get("teacher_behavior_beta_mean", 0.0),
                        lam=row["distill_lambda_current"],
                        conf=row.get("teacher_confidence_mean", 0.0),
                        js=row.get("teacher_student_js_mean", 0.0),
                        cfadv=row.get("teacher_cf_advantage_mean", 0.0),
                        dact=row.get("distill_branching_active_fraction", 0.0),
                    )
                )
    return rows


def extract_final_rows_by_mode_topology_count_seed(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, int, int], Dict[str, float]] = {}
    for row in rows:
        key = (
            str(row.get("topology", "unknown")),
            str(row.get("mode", "unknown")),
            int(row.get("agent_count", 0)),
            int(row.get("seed", 0)),
        )
        prev = grouped.get(key)
        if prev is None or float(row.get("update", 0.0)) >= float(prev.get("update", -1.0)):
            grouped[key] = dict(row)
    return [grouped[k] for k in sorted(grouped.keys())]


def build_final_summary_rows_with_topology(final_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, float]]] = {}
    for row in final_rows:
        key = (
            str(row.get("topology", "unknown")),
            str(row.get("mode", "unknown")),
            int(row.get("agent_count", 0)),
        )
        grouped.setdefault(key, []).append(dict(row))
    summary_rows: List[Dict[str, float]] = []
    for (topology, mode, agent_count), rows_ in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        ce_vals = [float(r.get("ce_mean_positive_regret", 0.0)) for r in rows_]
        ce_ucb_vals = [float(r.get("ce_ucb95", 0.0)) for r in rows_]
        cost_vals = [float(r.get("eval_realized_cost_mean", 0.0)) for r in rows_]
        eps_vals = [float(r.get("epsilon_ce_ex_post", 0.0)) for r in rows_]
        ce_mean, ce_std, ce_stderr, ce_ci95 = _mean_std_ci(ce_vals)
        cost_mean, cost_std, cost_stderr, cost_ci95 = _mean_std_ci(cost_vals)
        summary_rows.append({
            "topology": topology,
            "mode": mode,
            "agent_count": agent_count,
            "num_seeds": len(rows_),
            "final_update": int(max(float(r.get("update", 0.0)) for r in rows_)) if rows_ else 0,
            "ce_mean_positive_regret_mean": ce_mean,
            "ce_mean_positive_regret_std": ce_std,
            "ce_mean_positive_regret_stderr": ce_stderr,
            "ce_mean_positive_regret_ci95": ce_ci95,
            "ce_ucb95_mean": float(np.mean(ce_ucb_vals)) if ce_ucb_vals else 0.0,
            "eval_realized_cost_mean": cost_mean,
            "eval_realized_cost_std": cost_std,
            "eval_realized_cost_stderr": cost_stderr,
            "eval_realized_cost_ci95": cost_ci95,
            "fraction_ex_post_epsilon_ce": float(np.mean(eps_vals)) if eps_vals else 0.0,
        })
    return summary_rows



def extract_final_rows_by_mode_graph_size_count_seed(rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, int, int], Dict[str, float]] = {}
    for row in rows:
        key = (
            str(row.get("graph_size", "unknown")),
            str(row.get("mode", "unknown")),
            int(row.get("agent_count", 0)),
            int(row.get("seed", 0)),
        )
        prev = grouped.get(key)
        if prev is None or float(row.get("update", 0.0)) >= float(prev.get("update", -1.0)):
            grouped[key] = dict(row)
    return [grouped[k] for k in sorted(grouped.keys())]


def build_final_summary_rows_with_graph_size(final_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    grouped: Dict[Tuple[str, str, int], List[Dict[str, float]]] = {}
    for row in final_rows:
        key = (
            str(row.get("graph_size", "unknown")),
            str(row.get("mode", "unknown")),
            int(row.get("agent_count", 0)),
        )
        grouped.setdefault(key, []).append(dict(row))
    summary_rows: List[Dict[str, float]] = []
    for (graph_size, mode, agent_count), rows_ in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][2], x[0][1])):
        ce_vals = [float(r.get("ce_mean_positive_regret", 0.0)) for r in rows_]
        ce_ucb_vals = [float(r.get("ce_ucb95", 0.0)) for r in rows_]
        cost_vals = [float(r.get("eval_realized_cost_mean", 0.0)) for r in rows_]
        eps_vals = [float(r.get("epsilon_ce_ex_post", 0.0)) for r in rows_]
        ce_mean, ce_std, ce_stderr, ce_ci95 = _mean_std_ci(ce_vals)
        cost_mean, cost_std, cost_stderr, cost_ci95 = _mean_std_ci(cost_vals)
        summary_rows.append({
            "graph_size": graph_size,
            "mode": mode,
            "agent_count": agent_count,
            "num_seeds": len(rows_),
            "final_update": int(max(float(r.get("update", 0.0)) for r in rows_)) if rows_ else 0,
            "ce_mean_positive_regret_mean": ce_mean,
            "ce_mean_positive_regret_std": ce_std,
            "ce_mean_positive_regret_stderr": ce_stderr,
            "ce_mean_positive_regret_ci95": ce_ci95,
            "ce_mean_positive_regret_ci95_low": ce_mean - ce_ci95,
            "ce_mean_positive_regret_ci95_high": ce_mean + ce_ci95,
            "ce_ucb95_mean": float(np.mean(ce_ucb_vals)) if ce_ucb_vals else 0.0,
            "eval_realized_cost_mean": cost_mean,
            "eval_realized_cost_std": cost_std,
            "eval_realized_cost_stderr": cost_stderr,
            "eval_realized_cost_ci95": cost_ci95,
            "eval_realized_cost_ci95_low": cost_mean - cost_ci95,
            "eval_realized_cost_ci95_high": cost_mean + cost_ci95,
            "fraction_ex_post_epsilon_ce": float(np.mean(eps_vals)) if eps_vals else 0.0,
            "widths": str(rows_[0].get("widths", "")) if rows_ else "",
            "num_nodes": int(rows_[0].get("num_nodes", 0)) if rows_ else 0,
            "num_edges": int(rows_[0].get("num_edges", 0)) if rows_ else 0,
            "topology": str(rows_[0].get("topology", "")) if rows_ else "",
        })
    return summary_rows


def _risk_labels_in_rows(rows: Sequence[Dict[str, float]]) -> List[str]:
    labels = set()
    for row in rows:
        for key in row.keys():
            if key.startswith("risk_alpha_") and key.endswith("_ce_mean_positive_regret"):
                labels.add(key[len("risk_alpha_"):-len("_ce_mean_positive_regret")])
    def _label_value(label: str) -> float:
        try:
            return float(label.replace("m", "-").replace("p", "."))
        except Exception:
            return 0.0
    return sorted(labels, key=_label_value)


def build_final_per_risk_summary_rows(final_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    """Aggregate final metrics by graph size, mode, N, and risk-alpha bin."""
    risk_labels = _risk_labels_in_rows(final_rows)
    out: List[Dict[str, float]] = []
    grouped: Dict[Tuple[str, str, int, str], List[Dict[str, float]]] = {}
    for row in final_rows:
        for label in risk_labels:
            grouped.setdefault((str(row.get("graph_size", "unknown")), str(row.get("mode", "unknown")), int(row.get("agent_count", 0)), label), []).append(dict(row))
    for (graph_size, mode, agent_count, label), rows_ in sorted(grouped.items(), key=lambda x: (x[0][0], x[0][2], x[0][1], x[0][3])):
        ce_key = f"risk_alpha_{label}_ce_mean_positive_regret"
        ucb_key = f"risk_alpha_{label}_ce_ucb95"
        cost_key = f"risk_alpha_{label}_eval_realized_cost_mean"
        eff_key = f"risk_alpha_{label}_eval_effective_cost_mean"
        n_key = f"risk_alpha_{label}_num_eval_decisions"
        ce_vals = [float(r.get(ce_key, 0.0)) for r in rows_]
        ucb_vals = [float(r.get(ucb_key, 0.0)) for r in rows_]
        cost_vals = [float(r.get(cost_key, 0.0)) for r in rows_]
        eff_vals = [float(r.get(eff_key, 0.0)) for r in rows_]
        dec_vals = [float(r.get(n_key, 0.0)) for r in rows_]
        ce_mean, ce_std, ce_stderr, ce_ci95 = _mean_std_ci(ce_vals)
        cost_mean, cost_std, cost_stderr, cost_ci95 = _mean_std_ci(cost_vals)
        out.append({
            "graph_size": graph_size,
            "mode": mode,
            "agent_count": agent_count,
            "risk_alpha_label": label,
            "risk_alpha": float(label.replace("m", "-").replace("p", ".")),
            "num_seeds": len(rows_),
            "final_update": int(max(float(r.get("update", 0.0)) for r in rows_)) if rows_ else 0,
            "risk_ce_mean_positive_regret_mean": ce_mean,
            "risk_ce_mean_positive_regret_std": ce_std,
            "risk_ce_mean_positive_regret_stderr": ce_stderr,
            "risk_ce_mean_positive_regret_ci95": ce_ci95,
            "risk_ce_ucb95_mean": float(np.mean(ucb_vals)) if ucb_vals else 0.0,
            "risk_eval_realized_cost_mean": cost_mean,
            "risk_eval_realized_cost_std": cost_std,
            "risk_eval_realized_cost_stderr": cost_stderr,
            "risk_eval_realized_cost_ci95": cost_ci95,
            "risk_eval_effective_cost_mean": float(np.mean(eff_vals)) if eff_vals else 0.0,
            "risk_num_eval_decisions_mean": float(np.mean(dec_vals)) if dec_vals else 0.0,
            "widths": str(rows_[0].get("widths", "")) if rows_ else "",
            "num_nodes": int(rows_[0].get("num_nodes", 0)) if rows_ else 0,
            "num_edges": int(rows_[0].get("num_edges", 0)) if rows_ else 0,
            "topology": str(rows_[0].get("topology", "")) if rows_ else "",
        })
    return out


def plot_final_per_risk_metric_bars(per_risk_rows: Sequence[Dict[str, float]], graph_size: str, family: str, metric_key: str, output_path: Path, title: str, ylabel: str) -> None:
    rows = [dict(r) for r in per_risk_rows if str(r.get("graph_size", "")) == graph_size]
    if not rows:
        return
    risk_labels = sorted({str(r.get("risk_alpha_label", "")) for r in rows}, key=lambda x: float(x.replace("m", "-").replace("p", ".")))
    modes_present = sorted({str(r.get("mode", "")) for r in rows})
    preferred_order = ["ppo", "deepcfr", "deepcfr_rl_no_distill", "deepcfr_rl_distill"]
    modes = [m for m in preferred_order if m in modes_present] + [m for m in modes_present if m not in preferred_order]
    x = np.arange(len(risk_labels), dtype=float)
    total_width = 0.82
    width = total_width / max(1, len(modes))
    offsets = (np.arange(len(modes)) - (len(modes) - 1) / 2.0) * width
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)
    for mode, off in zip(modes, offsets):
        vals, cis = [], []
        for label in risk_labels:
            matched = [r for r in rows if str(r.get("mode")) == mode and str(r.get("risk_alpha_label")) == label]
            if matched:
                vals.append(float(matched[0].get(metric_key, 0.0)))
                # CI field naming follows metric family.
                if metric_key == "risk_ce_mean_positive_regret_mean":
                    cis.append(float(matched[0].get("risk_ce_mean_positive_regret_ci95", 0.0)))
                elif metric_key == "risk_eval_realized_cost_mean":
                    cis.append(float(matched[0].get("risk_eval_realized_cost_ci95", 0.0)))
                else:
                    cis.append(0.0)
            else:
                vals.append(float("nan")); cis.append(0.0)
        ax.bar(x + off, vals, width=width * 0.92, yerr=cis, capsize=3, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace("p", ".").replace("m", "-") for label in risk_labels])
    ax.set_xlabel("Risk aversion alpha")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _resolve_graph_size_map(args: argparse.Namespace) -> Dict[str, Tuple[int, ...]]:
    width_map = {
        "small": parse_int_tuple(args.small_widths),
        "medium": parse_int_tuple(args.medium_widths),
        "large": parse_int_tuple(args.large_widths),
    }
    graph_sizes = parse_str_tuple(args.graph_sizes)
    if not graph_sizes:
        graph_sizes = ("small", "medium", "large")
    bad = [g for g in graph_sizes if g not in width_map]
    if bad:
        raise ValueError(f"Unsupported graph sizes: {bad}; supported={sorted(width_map)}")
    for key, widths in width_map.items():
        if not widths:
            raise ValueError(f"Width preset for graph_size={key} is empty")
    return {g: width_map[g] for g in graph_sizes}


def _graph_metadata(widths: Sequence[int], topology: str, base_capacity: int) -> Dict[str, object]:
    graph = LayeredGraphBuilder.build(widths=widths, topology=topology, base_capacity=base_capacity)
    return {
        "widths": tuple(int(x) for x in widths),
        "widths_str": ",".join(str(int(x)) for x in widths),
        "num_nodes": int(len(graph.nodes)),
        "num_edges": int(len(graph.edges)),
        "num_layers": int(len(widths)),
        "topology": str(topology),
    }



# Override the final-bar plotting function so all modes present in a run are shown.
def plot_final_metric_bars_by_mode(rows: List[Dict[str, float]], graph_size: str, family: str, metric_key: str, output_path: Path, title: str, ylabel: str) -> None:
    if not rows:
        return
    grouped: Dict[int, Dict[str, List[float]]] = {}
    for row in rows:
        if str(row.get("graph_size", "")) != graph_size:
            continue
        n = int(row["agent_count"])
        mode = str(row["mode"])
        grouped.setdefault(n, {}).setdefault(mode, []).append(float(row[metric_key]))
    if not grouped:
        return
    preferred_order = ["ppo", "deepcfr", "deepcfr_rl_no_distill", "deepcfr_rl_distill"]
    present = sorted({m for by_mode in grouped.values() for m in by_mode.keys()})
    all_modes = [m for m in preferred_order if m in present] + [m for m in present if m not in preferred_order]
    agent_counts = sorted(grouped.keys())
    x = np.arange(len(agent_counts), dtype=float)
    total_width = 0.82
    width = total_width / max(1, len(all_modes))
    offsets = (np.arange(len(all_modes)) - (len(all_modes) - 1) / 2.0) * width
    fig = plt.figure(figsize=(10, 5.5))
    ax = fig.add_subplot(111)
    for mode, off in zip(all_modes, offsets):
        means = []
        cis = []
        for n in agent_counts:
            vals = grouped.get(n, {}).get(mode, [])
            mean, _std, _stderr, ci95 = _mean_std_ci(vals)
            means.append(mean if vals else float('nan'))
            cis.append(ci95)
        ax.bar(x + off, means, width=width * 0.92, yerr=cis, capsize=4, label=mode)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in agent_counts])
    ax.set_xlabel("Agent count")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


# -----------------------------------------------------------------------------
# Unified final CLI parser
# -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the single final parser.

    All defaults come from FINAL_DEFAULTS. This avoids multiple hidden
    parameter-setting sections in the file.
    """
    d = FINAL_DEFAULTS
    parser = argparse.ArgumentParser(
        description=(
            "Risk-stratified shared-teacher S-DNCG run: "
            "PPO / DeepCFR / guided PPO / guided PPO + KL."
        )
    )

    # Sweep / environment
    parser.add_argument("--output-dir", type=str, default=d["output_dir"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(d["seeds"]))
    parser.add_argument("--modes", type=str, default=d["modes"])
    parser.add_argument("--graph-sizes", type=str, default=d["graph_sizes"])
    parser.add_argument("--small-widths", type=str, default=d["small_widths"])
    parser.add_argument("--medium-widths", type=str, default=d["medium_widths"])
    parser.add_argument("--large-widths", type=str, default=d["large_widths"])
    parser.add_argument("--topology", choices=["fully_connected", "adjacent_only", "main_path"], default=d["topology"])
    parser.add_argument("--base-capacity", type=int, default=d["base_capacity"])
    parser.add_argument("--num-agents", type=int, default=d["num_agents"])
    parser.add_argument("--agent-counts", type=str, default=d["agent_counts"])
    parser.add_argument("--horizon", type=int, default=d["horizon"])
    parser.add_argument("--risk-alphas", type=str, default=d["risk_alphas"])
    parser.add_argument("--reward-mode", choices=["effective", "realized"], default=d["reward_mode"])

    # PPO
    parser.add_argument("--actor-lr", type=float, default=d["actor_lr"])
    parser.add_argument("--critic-lr", type=float, default=d["critic_lr"])
    parser.add_argument("--gamma", type=float, default=d["gamma"])
    parser.add_argument("--gae-lambda", type=float, default=d["gae_lambda"])
    parser.add_argument("--clip-ratio", type=float, default=d["clip_ratio"])
    parser.add_argument("--entropy-coef", type=float, default=d["entropy_coef"])
    parser.add_argument("--value-coef", type=float, default=d["value_coef"])
    parser.add_argument("--max-grad-norm", type=float, default=d["max_grad_norm"])
    parser.add_argument("--hidden-dim", type=int, default=d["hidden_dim"])
    parser.add_argument("--node-embed-dim", type=int, default=d["node_embed_dim"])
    parser.add_argument("--ppo-epochs", type=int, default=d["ppo_epochs"])
    parser.add_argument("--minibatch-size", type=int, default=d["minibatch_size"])
    parser.add_argument("--episodes-per-update", type=int, default=d["episodes_per_update"])
    parser.add_argument("--total-updates", type=int, default=d["total_updates"])
    parser.add_argument("--eval-every", type=int, default=d["eval_every"])
    parser.add_argument("--eval-episodes", type=int, default=d["eval_episodes"])

    # Budgeted shared DeepCFR-style teacher
    parser.add_argument("--teacher-eval-policy", choices=["average", "current"], default=d["teacher_eval_policy"])
    parser.add_argument("--deep-teacher-hidden-dim", type=int, default=d["deep_teacher_hidden_dim"])
    parser.add_argument("--deep-teacher-node-embed-dim", type=int, default=d["deep_teacher_node_embed_dim"])
    parser.add_argument("--deep-teacher-lr", type=float, default=d["deep_teacher_lr"])
    parser.add_argument("--deep-teacher-buffer-capacity", type=int, default=d["deep_teacher_buffer_capacity"])
    parser.add_argument("--deep-teacher-train-steps", type=int, default=d["deep_teacher_train_steps"])
    parser.add_argument("--deep-teacher-strategy-train-steps", type=int, default=d["deep_teacher_strategy_train_steps"])
    parser.add_argument("--deep-teacher-batch-size", type=int, default=d["deep_teacher_batch_size"])
    parser.add_argument("--deep-teacher-min-buffer", type=int, default=d["deep_teacher_min_buffer"])
    parser.add_argument("--deep-teacher-retrain-from-scratch", action="store_true", default=d["deep_teacher_retrain_from_scratch"])
    parser.add_argument("--deep-teacher-target-lookahead-steps", type=int, default=d["deep_teacher_target_lookahead_steps"])
    parser.add_argument("--deep-teacher-target-policy-mode", choices=["current", "average"], default=d["deep_teacher_target_policy_mode"])
    parser.add_argument("--deep-teacher-target-future-load-floor", type=float, default=d["deep_teacher_target_future_load_floor"])

    # Teacher-student guidance / KL ablation
    parser.add_argument("--hybrid-teacher-warmup-updates", type=int, default=d["hybrid_teacher_warmup_updates"])
    parser.add_argument("--hybrid-teacher-policy-mode", choices=["average", "current"], default=d["hybrid_teacher_policy_mode"])
    parser.add_argument("--hybrid-beta-start", type=float, default=d["hybrid_beta_start"])
    parser.add_argument("--hybrid-beta-end", type=float, default=d["hybrid_beta_end"])
    parser.add_argument("--hybrid-beta-floor", type=float, default=d["hybrid_beta_floor"])
    parser.add_argument("--hybrid-mix-conf-threshold", type=float, default=d["hybrid_mix_conf_threshold"])
    parser.add_argument("--hybrid-mix-conf-scale", type=float, default=d["hybrid_mix_conf_scale"])
    parser.add_argument("--hybrid-mix-js-threshold", type=float, default=d["hybrid_mix_js_threshold"])
    parser.add_argument("--hybrid-mix-js-scale", type=float, default=d["hybrid_mix_js_scale"])
    parser.add_argument("--hybrid-mix-min-js-gate", type=float, default=d["hybrid_mix_min_js_gate"])
    parser.add_argument("--distill-lambda-max", type=float, default=d["distill_lambda_max"])
    parser.add_argument("--distill-temperature", type=float, default=d["distill_temperature"])
    parser.add_argument("--distill-conf-threshold", type=float, default=d["distill_conf_threshold"])
    parser.add_argument("--distill-conf-scale", type=float, default=d["distill_conf_scale"])
    parser.add_argument("--distill-gap-threshold", type=float, default=d["distill_gap_threshold"])
    parser.add_argument("--distill-gap-scale", type=float, default=d["distill_gap_scale"])
    parser.add_argument("--distill-cf-advantage-threshold", type=float, default=d["distill_cf_advantage_threshold"])
    parser.add_argument("--distill-cf-advantage-scale", type=float, default=d["distill_cf_advantage_scale"])
    parser.add_argument("--distill-ramp-updates", type=int, default=d["distill_ramp_updates"])
    parser.add_argument("--distill-min-branching-actions", type=int, default=d["distill_min_branching_actions"])
    parser.add_argument("--adaptive-teacher-snapshot", action="store_true", default=d["adaptive_teacher_snapshot"])
    parser.add_argument("--no-adaptive-teacher-snapshot", dest="adaptive_teacher_snapshot", action="store_false")
    parser.add_argument("--teacher-freeze-min-updates", type=int, default=d["teacher_freeze_min_updates"])
    parser.add_argument("--teacher-freeze-max-updates", type=int, default=d["teacher_freeze_max_updates"])
    parser.add_argument("--teacher-ready-ce-ucb-threshold", type=float, default=d["teacher_ready_ce_ucb_threshold"])
    parser.add_argument("--teacher-ready-stability-window", type=int, default=d["teacher_ready_stability_window"])
    parser.add_argument("--teacher-ready-stability-delta", type=float, default=d["teacher_ready_stability_delta"])
    parser.add_argument("--teacher-ready-min-confidence", type=float, default=d["teacher_ready_min_confidence"])
    parser.add_argument("--distill-decay-start-fraction", type=float, default=d["distill_decay_start_fraction"])

    # Evaluation
    parser.add_argument("--epsilon-ce", type=float, default=d["epsilon_ce"])
    parser.add_argument("--stability-threshold", type=float, default=d["stability_threshold"])
    parser.add_argument("--trailing-window", type=int, default=d["trailing_window"])
    parser.add_argument("--device", type=str, default=d["device"])
    parser.add_argument("--verbose", action="store_true")
    return parser

def _build_env_cfg_local(seed: int, agent_count: int, env_cfg: EnvConfig) -> EnvConfig:
    return EnvConfig(
        num_agents=agent_count,
        horizon=env_cfg.horizon,
        cost_clip=env_cfg.cost_clip,
        reward_mode=env_cfg.reward_mode,
        shared_edge_incident=env_cfg.shared_edge_incident,
        normal_std_floor=env_cfg.normal_std_floor,
        random_seed=seed,
    )


def _make_deep_teacher_for_snapshot(
    seed: int,
    agent_count: int,
    widths: Sequence[int],
    topology: str,
    base_capacity: int,
    risk_alphas: Sequence[float],
    env_cfg: EnvConfig,
    encoder_cfg: EncoderConfig,
    deep_teacher_cfg: DeepTeacherConfig,
    device: torch.device,
    snapshot: Mapping[str, object] | None = None,
) -> Tuple[SDNCGEnv, DeepCFRStyleTeacher]:
    env = build_env(
        seed,
        widths,
        topology,
        base_capacity,
        _build_env_cfg_local(seed, agent_count, env_cfg),
        encoder_cfg,
        risk_alphas,
        num_agents=agent_count,
    )
    teacher = DeepCFRStyleTeacher(env, deep_teacher_cfg, device=device, seed=seed)
    if snapshot is not None:
        load_deep_teacher_snapshot(teacher, snapshot)
    return env, teacher


def train_shared_teacher_snapshot(
    seed: int,
    agent_count: int,
    widths: Sequence[int],
    topology: str,
    base_capacity: int,
    risk_alphas: Sequence[float],
    env_cfg: EnvConfig,
    encoder_cfg: EncoderConfig,
    ppo_cfg: PPOConfig,
    deep_teacher_cfg: DeepTeacherConfig,
    hybrid_cfg: HybridConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
    logger: RunLogger,
) -> Tuple[Dict[str, object], List[Dict[str, float]], Dict[str, float]]:
    """Train one risk-aware DeepCFR-style teacher and return its best snapshot.

    The teacher is trained until the readiness/convergence criterion is met, or
    until the user-specified `total_updates` budget is exhausted. We no longer
    force-stop the shared teacher at a separate `teacher_freeze_max_updates` value.
    If the teacher does not become ready, the best CE-UCB snapshot observed over
    the full teacher budget is used.
    """
    set_global_seed(seed)
    env, teacher = _make_deep_teacher_for_snapshot(
        seed, agent_count, widths, topology, base_capacity, risk_alphas,
        env_cfg, encoder_cfg, deep_teacher_cfg, device, snapshot=None,
    )
    risk_cycle_label = "|".join(f"{x:.2f}" for x in risk_alphas)
    rows: List[Dict[str, float]] = []
    history: List[Dict[str, float]] = []
    best_snapshot: Dict[str, object] | None = None
    best_metric = float("inf")
    best_update = 0
    freeze_update = 0
    freeze_reason = "not_frozen"

    # Train the shared teacher until it satisfies the readiness criterion or
    # until the same update budget requested for the experiment is exhausted.
    # The legacy teacher_freeze_max_updates argument is intentionally ignored
    # here to avoid premature teacher freezing.
    max_pretrain = max(1, int(ppo_cfg.total_updates))
    eval_points = set([1, int(ppo_cfg.total_updates)])
    eval_points.update(range(max(1, int(ppo_cfg.eval_every)), int(ppo_cfg.total_updates) + 1, max(1, int(ppo_cfg.eval_every))))
    eval_points.add(max_pretrain)
    progress_log_every = max(1, max_pretrain // 3)

    def _make_teacher_row(update_idx: int, train_stats: Mapping[str, float], eval_stats: Mapping[str, float], training_active: bool) -> Dict[str, float]:
        return {
            "seed": float(seed),
            "seed_label": f"seed_{seed}",
            "agent_count": float(agent_count),
            "agent_count_label": f"N_{agent_count}",
            "update": float(update_idx),
            "risk_cycle_label": risk_cycle_label,
            **{k: float(v) for k, v in train_stats.items() if isinstance(v, (int, float, np.floating))},
            **{k: float(v) for k, v in eval_stats.items() if isinstance(v, (int, float, np.floating))},
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "distill_loss": 0.0,
            "distill_active_fraction": 0.0,
            "shared_teacher_snapshot": 1.0,
            "teacher_training_active": float(1.0 if training_active else 0.0),
            "teacher_best_ce_ucb95": float(best_metric if math.isfinite(best_metric) else 0.0),
            "teacher_best_snapshot_update": float(best_update),
            "teacher_freeze_update": float(freeze_update),
            "teacher_freeze_reason_code": 1.0 if freeze_reason == "ready" else 0.0,
        }

    for update_idx in range(1, max_pretrain + 1):
        train_stats = train_deep_teacher_batch(
            env, teacher, deep_teacher_cfg,
            seed_offset=seed * 100_000 + update_idx * 10_000,
            update_idx=update_idx,
        )
        should_eval = update_idx in eval_points or update_idx >= max_pretrain
        if should_eval:
            eval_stats = evaluate_deep_teacher(
                env, teacher, eval_cfg,
                eval_episodes=deep_teacher_cfg.eval_episodes,
                seed_offset=seed * 20_000_000 + update_idx * 40_000,
                policy_mode=hybrid_cfg.teacher_policy_mode,
                deterministic=False,
            )
            metric = _risk_aware_teacher_eval_metric(eval_stats)
            allow_snapshot_selection = update_idx >= hybrid_cfg.teacher_freeze_min_updates
            if allow_snapshot_selection:
                history.append(dict(eval_stats))
            if allow_snapshot_selection and metric < best_metric and bool(teacher.regret_trained or teacher.strategy_trained):
                best_metric = metric
                best_update = update_idx
                best_snapshot = clone_deep_teacher_snapshot(teacher)
            ready, ready_reason = teacher_ready_from_history(history, hybrid_cfg, update_idx)
            completed_budget = update_idx >= max_pretrain
            if ready or completed_budget:
                if best_snapshot is None:
                    best_snapshot = clone_deep_teacher_snapshot(teacher)
                    best_metric = metric
                    best_update = update_idx
                freeze_update = update_idx
                freeze_reason = "ready" if ready else f"completed_full_budget:{ready_reason}"
                load_deep_teacher_snapshot(teacher, best_snapshot)
                # Evaluate the selected snapshot at the freeze point for the
                # recorded DeepCFR row.
                eval_stats = evaluate_deep_teacher(
                    env, teacher, eval_cfg,
                    eval_episodes=deep_teacher_cfg.eval_episodes,
                    seed_offset=seed * 20_000_000 + update_idx * 40_000 + 17,
                    policy_mode=hybrid_cfg.teacher_policy_mode,
                    deterministic=False,
                )
                rows.append(_make_teacher_row(update_idx, train_stats, eval_stats, training_active=True))
                logger.summary(
                    "[shared_teacher] N={n:>2d} seed={s} freeze={u:4d} reason={reason} best_ucb={best:.4f}@{bu} ce={ce:.4f} cost={cost:.2f}".format(
                        n=agent_count, s=seed, u=update_idx, reason=freeze_reason,
                        best=best_metric if math.isfinite(best_metric) else 0.0,
                        bu=best_update, ce=eval_stats.get("ce_mean_positive_regret", 0.0),
                        cost=eval_stats.get("eval_realized_cost_mean", 0.0),
                    )
                )
                break
            else:
                rows.append(_make_teacher_row(update_idx, train_stats, eval_stats, training_active=True))
                if update_idx == 1 or update_idx % progress_log_every == 0:
                    logger.summary(
                        "[shared_teacher] N={n:>2d} seed={s} update={u:4d} ce={ce:.4f} ucb={ucb:.4f} best_ucb={best:.4f}@{bu}".format(
                            n=agent_count, s=seed, u=update_idx,
                            ce=eval_stats.get("ce_mean_positive_regret", 0.0),
                            ucb=_risk_aware_teacher_eval_metric(eval_stats),
                            best=best_metric if math.isfinite(best_metric) else 0.0,
                            bu=best_update,
                        )
                    )

    if best_snapshot is None:
        best_snapshot = clone_deep_teacher_snapshot(teacher)
        best_update = max_pretrain
        best_metric = 0.0
        freeze_update = max_pretrain
        freeze_reason = "fallback_no_snapshot"
    load_deep_teacher_snapshot(teacher, best_snapshot)

    # Fill post-freeze DeepCFR eval points so plots and final tables align with
    # the student update horizon. These are evaluations of the same frozen best
    # teacher snapshot, not additional teacher training.
    for update_idx in sorted(u for u in eval_points if u > max_pretrain):
        eval_stats = evaluate_deep_teacher(
            env, teacher, eval_cfg,
            eval_episodes=deep_teacher_cfg.eval_episodes,
            seed_offset=seed * 20_000_000 + update_idx * 40_000 + 31,
            policy_mode=hybrid_cfg.teacher_policy_mode,
            deterministic=False,
        )
        rows.append(_make_teacher_row(update_idx, {"deep_regret_buffer_size": float(len(teacher.regret_buffer)), "deep_strategy_buffer_size": float(len(teacher.strategy_buffer))}, eval_stats, training_active=False))

    info = {
        "teacher_best_ce_ucb95": float(best_metric if math.isfinite(best_metric) else 0.0),
        "teacher_best_snapshot_update": float(best_update),
        "teacher_freeze_update": float(freeze_update),
        "teacher_freeze_reason_ready": float(1.0 if freeze_reason == "ready" else 0.0),
        "teacher_completed_full_budget": float(1.0 if str(freeze_reason).startswith("completed_full_budget") else 0.0),
    }
    return best_snapshot, rows, info


def run_single_seed_guided_with_shared_teacher(
    seed: int,
    mode_name: str,
    use_distill: bool,
    teacher_snapshot: Mapping[str, object],
    teacher_info: Mapping[str, float],
    agent_count: int,
    widths: Sequence[int],
    topology: str,
    base_capacity: int,
    risk_alphas: Sequence[float],
    env_cfg: EnvConfig,
    encoder_cfg: EncoderConfig,
    ppo_cfg: PPOConfig,
    deep_teacher_cfg: DeepTeacherConfig,
    hybrid_cfg: HybridConfig,
    eval_cfg: EvalConfig,
    device: torch.device,
    logger: RunLogger,
) -> List[Dict[str, float]]:
    """Run guided PPO using a pre-trained frozen shared teacher snapshot."""
    set_global_seed(seed)
    env, teacher = _make_deep_teacher_for_snapshot(
        seed, agent_count, widths, topology, base_capacity, risk_alphas,
        env_cfg, encoder_cfg, deep_teacher_cfg, device, snapshot=teacher_snapshot,
    )
    agent = GuidedPPOAgent(env, ppo_cfg, device)
    risk_cycle_label = "|".join(f"{x:.2f}" for x in risk_alphas)
    rows: List[Dict[str, float]] = []
    progress_log_every = max(1, ppo_cfg.total_updates // 4)
    guided_total = max(1, int(ppo_cfg.total_updates))
    hybrid_cfg_guided = replace(hybrid_cfg, teacher_warmup_updates=0)

    for update_idx in range(1, ppo_cfg.total_updates + 1):
        current_lambda = distill_coef_two_stage(hybrid_cfg_guided, update_idx, guided_total, use_distill=use_distill)
        guided_samples, train_stats = collect_guided_training_batch(
            env, agent, teacher, ppo_cfg, hybrid_cfg_guided,
            seed_offset=seed * 1_000_000 + update_idx * 20_000,
            update_idx=update_idx,
            use_distill=use_distill,
        )
        opt_stats = agent.update_guided(
            guided_samples,
            distill_coef=current_lambda,
            distill_temperature=hybrid_cfg_guided.distill_temperature,
            use_distill=use_distill,
        )
        beta_mean = float(train_stats.get("teacher_behavior_beta_mean", 0.0))
        if beta_mean <= 1e-5 and current_lambda <= 1e-8:
            hybrid_stage = "ppo_finetune"
        elif use_distill and current_lambda > 1e-8:
            hybrid_stage = "guided_distill"
        else:
            hybrid_stage = "guided_no_distill"

        if update_idx == 1 or update_idx % ppo_cfg.eval_every == 0 or update_idx == ppo_cfg.total_updates:
            student_eval = evaluate_policy(
                env, agent, eval_cfg,
                eval_episodes=ppo_cfg.eval_episodes,
                seed_offset=seed * 10_000_000 + update_idx * 30_000,
                deterministic=False,
            )
            teacher_eval = evaluate_deep_teacher(
                env, teacher, eval_cfg,
                eval_episodes=deep_teacher_cfg.eval_episodes,
                seed_offset=seed * 20_000_000 + update_idx * 40_000,
                policy_mode=hybrid_cfg.teacher_policy_mode,
                deterministic=False,
            )
            row = {
                "seed": float(seed),
                "seed_label": f"seed_{seed}",
                "agent_count": float(agent_count),
                "agent_count_label": f"N_{agent_count}",
                "update": float(update_idx),
                "risk_cycle_label": risk_cycle_label,
                **train_stats,
                **opt_stats,
                **student_eval,
                "teacher_eval_realized_cost_mean": float(teacher_eval.get("eval_realized_cost_mean", 0.0)),
                "teacher_eval_effective_cost_mean": float(teacher_eval.get("eval_effective_cost_mean", 0.0)),
                "teacher_eval_ce_mean_positive_regret": float(teacher_eval.get("ce_mean_positive_regret", 0.0)),
                "teacher_eval_ce_ucb95": float(teacher_eval.get("ce_ucb95", 0.0)),
                "teacher_eval_confidence_mean": float(teacher_eval.get("teacher_confidence_mean", 0.0)),
                **{f"teacher_eval_{k}": float(v) for k, v in teacher_eval.items() if k.startswith("risk_alpha_") and isinstance(v, (int, float, np.floating))},
                "teacher_best_ce_ucb95": float(teacher_info.get("teacher_best_ce_ucb95", 0.0)),
                "teacher_best_snapshot_update": float(teacher_info.get("teacher_best_snapshot_update", 0.0)),
                "teacher_freeze_update": float(teacher_info.get("teacher_freeze_update", 0.0)),
                "teacher_snapshot_shared": 1.0,
                "teacher_training_active": 0.0,
                "guided_step": float(update_idx),
                "guided_total": float(guided_total),
                "distill_lambda_current": float(current_lambda),
                "hybrid_use_distill": float(1.0 if use_distill else 0.0),
                "hybrid_stage": hybrid_stage,
            }
            rows.append(row)
            if update_idx == 1 or update_idx == ppo_cfg.total_updates or update_idx % progress_log_every == 0:
                logger.summary(
                    "[{mode}] N={n:>2d} seed={s} update={u:4d} stage={stage} ce={ce:.4f} cost={cost:.2f} teacher_ce={tce:.4f} beta={beta:.3f} lam={lam:.3f} conf={conf:.3f} js={js:.4f} cfadv={cfadv:.4f} d_active={dact:.3f}".format(
                        mode=mode_name, n=agent_count, s=seed, u=update_idx, stage=hybrid_stage,
                        ce=row["ce_mean_positive_regret"], cost=row["eval_realized_cost_mean"],
                        tce=row["teacher_eval_ce_mean_positive_regret"],
                        beta=row.get("teacher_behavior_beta_mean", 0.0), lam=row["distill_lambda_current"],
                        conf=row.get("teacher_confidence_mean", 0.0), js=row.get("teacher_student_js_mean", 0.0),
                        cfadv=row.get("teacher_cf_advantage_mean", 0.0),
                        dact=row.get("distill_branching_active_fraction", 0.0),
                    )
                )
    return rows


def _annotate_rows(rows: List[Dict[str, float]], mode: str, graph_size: str, topology: str, meta: Mapping[str, object]) -> List[Dict[str, float]]:
    for row in rows:
        row["mode"] = mode
        row["graph_size"] = graph_size
        row["topology"] = topology
        row["widths"] = meta["widths_str"]
        row["num_nodes"] = meta["num_nodes"]
        row["num_edges"] = meta["num_edges"]
    return rows




# =============================================================================
# Paper-ready visualization suite
# =============================================================================

PAPER_MODE_ORDER = ["ppo", "deepcfr", "deepcfr_rl_no_distill", "deepcfr_rl_distill"]
PAPER_MODE_LABELS = {
    "ppo": "PPO",
    "deepcfr": "DeepCFR",
    "deepcfr_rl_no_distill": "Guided PPO",
    "deepcfr_rl_distill": "Guided PPO + KL",
}
PAPER_GRAPH_ORDER = ["small", "medium", "large"]
PAPER_MARKERS = ["o", "s", "^", "D", "v", "P"]
PAPER_LINESTYLES = ["-", "--", "-.", ":", "-"]


def _paper_style() -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })


def _paper_mode_label(mode: str) -> str:
    return PAPER_MODE_LABELS.get(str(mode), str(mode))


def _paper_graph_order(values: Sequence[str]) -> List[str]:
    present = [g for g in PAPER_GRAPH_ORDER if g in set(values)]
    return present + [g for g in sorted(set(values)) if g not in present]


def _paper_mode_order(values: Sequence[str]) -> List[str]:
    present = [m for m in PAPER_MODE_ORDER if m in set(values)]
    return present + [m for m in sorted(set(values)) if m not in present]


def _paper_save(fig: plt.Figure, out_png: Path) -> Dict[str, str]:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    out_pdf = out_png.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(out_png), "pdf": str(out_pdf)}


def _summary_lookup(summary_rows: Sequence[Dict[str, float]]) -> Dict[Tuple[str, int, str], Dict[str, float]]:
    lookup: Dict[Tuple[str, int, str], Dict[str, float]] = {}
    for row in summary_rows:
        lookup[(str(row.get("graph_size")), int(row.get("agent_count", 0)), str(row.get("mode")))] = dict(row)
    return lookup


def paper_plot_final_metric_grid(
    summary_rows: Sequence[Dict[str, float]],
    metric_mean_key: str,
    metric_ci_key: str,
    ylabel: str,
    output_path: Path,
    title: str,
) -> Dict[str, str] | None:
    rows = [dict(r) for r in summary_rows]
    if not rows:
        return None
    graph_sizes = _paper_graph_order([str(r.get("graph_size")) for r in rows])
    modes = _paper_mode_order([str(r.get("mode")) for r in rows])
    agent_counts = sorted({int(r.get("agent_count", 0)) for r in rows})
    lookup = _summary_lookup(rows)
    fig, axes = plt.subplots(1, len(graph_sizes), figsize=(4.1 * len(graph_sizes), 3.5), sharey=False)
    if len(graph_sizes) == 1:
        axes = [axes]
    for ax, graph_size in zip(axes, graph_sizes):
        for mi, mode in enumerate(modes):
            xs, means, cis = [], [], []
            for n in agent_counts:
                row = lookup.get((graph_size, n, mode))
                if row is None:
                    continue
                xs.append(n)
                means.append(float(row.get(metric_mean_key, np.nan)))
                cis.append(float(row.get(metric_ci_key, 0.0)))
            if xs:
                ax.errorbar(
                    xs, means, yerr=cis, marker=PAPER_MARKERS[mi % len(PAPER_MARKERS)],
                    linestyle=PAPER_LINESTYLES[mi % len(PAPER_LINESTYLES)], capsize=3,
                    linewidth=1.8, markersize=5, label=_paper_mode_label(mode),
                )
        ax.set_title(f"{graph_size.capitalize()} graph")
        ax.set_xlabel("Number of agents")
        ax.set_xticks(agent_counts)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(title, y=1.14)
    return _paper_save(fig, output_path)


def paper_plot_learning_curves(
    all_rows: Sequence[Dict[str, float]],
    graph_size: str,
    agent_count: int,
    metric_key: str,
    ylabel: str,
    output_path: Path,
    title: str,
) -> Dict[str, str] | None:
    rows = [dict(r) for r in all_rows if str(r.get("graph_size")) == graph_size and int(r.get("agent_count", -1)) == int(agent_count)]
    if not rows:
        return None
    modes = _paper_mode_order([str(r.get("mode")) for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for mi, mode in enumerate(modes):
        mode_rows = [r for r in rows if str(r.get("mode")) == mode]
        by_update: Dict[int, List[float]] = {}
        for r in mode_rows:
            by_update.setdefault(int(r.get("update", 0)), []).append(float(r.get(metric_key, np.nan)))
        updates = sorted(by_update.keys())
        if not updates:
            continue
        means, cis = [], []
        for u in updates:
            mean, _std, _stderr, ci95 = _mean_std_ci(by_update[u])
            means.append(mean)
            cis.append(ci95)
        means_arr = np.asarray(means, dtype=float)
        cis_arr = np.asarray(cis, dtype=float)
        ax.plot(updates, means_arr, marker=PAPER_MARKERS[mi % len(PAPER_MARKERS)], linestyle=PAPER_LINESTYLES[mi % len(PAPER_LINESTYLES)], linewidth=1.8, markersize=4, label=_paper_mode_label(mode))
        ax.fill_between(updates, means_arr - cis_arr, means_arr + cis_arr, alpha=0.12)
    ax.set_xlabel("Training update")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, ncol=2)
    return _paper_save(fig, output_path)


def paper_build_gap_closure_table(summary_rows: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
    lookup = _summary_lookup(summary_rows)
    graph_sizes = _paper_graph_order([str(r.get("graph_size")) for r in summary_rows])
    agent_counts = sorted({int(r.get("agent_count", 0)) for r in summary_rows})
    out: List[Dict[str, float]] = []
    for graph_size in graph_sizes:
        for n in agent_counts:
            ppo = lookup.get((graph_size, n, "ppo"))
            deep = lookup.get((graph_size, n, "deepcfr"))
            guided = lookup.get((graph_size, n, "deepcfr_rl_no_distill"))
            distill = lookup.get((graph_size, n, "deepcfr_rl_distill"))
            if ppo is None or deep is None or guided is None:
                continue
            ppo_ce = float(ppo.get("ce_mean_positive_regret_mean", np.nan))
            deep_ce = float(deep.get("ce_mean_positive_regret_mean", np.nan))
            guided_ce = float(guided.get("ce_mean_positive_regret_mean", np.nan))
            denom = ppo_ce - deep_ce
            gap_closure = (ppo_ce - guided_ce) / denom if abs(denom) > 1e-12 else np.nan
            rel_improve = (ppo_ce - guided_ce) / ppo_ce if abs(ppo_ce) > 1e-12 else np.nan
            row = {
                "graph_size": graph_size,
                "agent_count": n,
                "ppo_ce": ppo_ce,
                "deepcfr_ce": deep_ce,
                "guided_ce": guided_ce,
                "ce_gap_closure_vs_deepcfr": float(gap_closure),
                "guided_relative_ce_improvement_vs_ppo": float(rel_improve),
                "ppo_cost": float(ppo.get("eval_realized_cost_mean", np.nan)),
                "guided_cost": float(guided.get("eval_realized_cost_mean", np.nan)),
            }
            if distill is not None:
                row["distill_ce"] = float(distill.get("ce_mean_positive_regret_mean", np.nan))
                row["kl_delta_ce_vs_no_kl"] = row["distill_ce"] - guided_ce
                row["distill_cost"] = float(distill.get("eval_realized_cost_mean", np.nan))
                row["kl_delta_cost_vs_no_kl"] = row["distill_cost"] - row["guided_cost"]
            out.append(row)
    return out


def paper_plot_gap_closure(gap_rows: Sequence[Dict[str, float]], output_path: Path) -> Dict[str, str] | None:
    rows = [dict(r) for r in gap_rows if not math.isnan(float(r.get("ce_gap_closure_vs_deepcfr", np.nan)))]
    if not rows:
        return None
    graph_sizes = _paper_graph_order([str(r.get("graph_size")) for r in rows])
    agent_counts = sorted({int(r.get("agent_count", 0)) for r in rows})
    x = np.arange(len(agent_counts), dtype=float)
    total_width = 0.82
    width = total_width / max(1, len(graph_sizes))
    offsets = (np.arange(len(graph_sizes)) - (len(graph_sizes) - 1) / 2.0) * width
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    for gi, graph_size in enumerate(graph_sizes):
        vals = []
        for n in agent_counts:
            matched = [r for r in rows if str(r.get("graph_size")) == graph_size and int(r.get("agent_count", 0)) == n]
            vals.append(float(matched[0].get("ce_gap_closure_vs_deepcfr", np.nan)) if matched else np.nan)
        ax.bar(x + offsets[gi], vals, width=width * 0.92, label=f"{graph_size.capitalize()}")
    ax.axhline(0.0, linewidth=1.0)
    ax.axhline(1.0, linewidth=1.0, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"N={n}" for n in agent_counts])
    ax.set_ylabel("CE gap closure vs. DeepCFR")
    ax.set_xlabel("Number of agents")
    ax.set_title("How much of DeepCFR's CE advantage is transferred to Guided PPO")
    ax.legend(frameon=False)
    return _paper_save(fig, output_path)


def paper_plot_kl_ablation(gap_rows: Sequence[Dict[str, float]], output_path: Path) -> Dict[str, str] | None:
    rows = [dict(r) for r in gap_rows if "kl_delta_ce_vs_no_kl" in r]
    if not rows:
        return None
    labels = [f"{str(r['graph_size'])[0].upper()}-N{int(r['agent_count'])}" for r in rows]
    x = np.arange(len(rows), dtype=float)
    ce_delta = np.asarray([float(r.get("kl_delta_ce_vs_no_kl", np.nan)) for r in rows])
    cost_delta = np.asarray([float(r.get("kl_delta_cost_vs_no_kl", np.nan)) for r in rows])
    fig, axes = plt.subplots(2, 1, figsize=(max(7.4, len(rows) * 0.55), 6.0), sharex=True)
    axes[0].bar(x, ce_delta)
    axes[0].axhline(0.0, linewidth=1.0)
    axes[0].set_ylabel("Δ CE regret\n(KL - no KL)")
    axes[0].set_title("KL distillation ablation: positive values mean KL is worse")
    axes[1].bar(x, cost_delta)
    axes[1].axhline(0.0, linewidth=1.0)
    axes[1].set_ylabel("Δ realized cost\n(KL - no KL)")
    axes[1].set_xlabel("Graph size and population")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=45, ha="right")
    return _paper_save(fig, output_path)


def paper_plot_per_risk_metric(
    per_risk_rows: Sequence[Dict[str, float]],
    agent_count: int,
    metric_mean_key: str,
    metric_ci_key: str,
    ylabel: str,
    output_path: Path,
    title: str,
) -> Dict[str, str] | None:
    rows = [dict(r) for r in per_risk_rows if int(r.get("agent_count", -1)) == int(agent_count)]
    if not rows:
        return None
    graph_sizes = _paper_graph_order([str(r.get("graph_size")) for r in rows])
    modes = _paper_mode_order([str(r.get("mode")) for r in rows])
    risk_vals = sorted({float(r.get("risk_alpha", 0.0)) for r in rows})
    fig, axes = plt.subplots(1, len(graph_sizes), figsize=(4.2 * len(graph_sizes), 3.6), sharey=False)
    if len(graph_sizes) == 1:
        axes = [axes]
    for ax, graph_size in zip(axes, graph_sizes):
        for mi, mode in enumerate(modes):
            xs, means, cis = [], [], []
            for alpha in risk_vals:
                matched = [r for r in rows if str(r.get("graph_size")) == graph_size and str(r.get("mode")) == mode and abs(float(r.get("risk_alpha", 0.0)) - alpha) < 1e-9]
                if not matched:
                    continue
                xs.append(alpha)
                means.append(float(matched[0].get(metric_mean_key, np.nan)))
                cis.append(float(matched[0].get(metric_ci_key, 0.0)))
            if xs:
                ax.errorbar(xs, means, yerr=cis, marker=PAPER_MARKERS[mi % len(PAPER_MARKERS)], linestyle=PAPER_LINESTYLES[mi % len(PAPER_LINESTYLES)], capsize=3, linewidth=1.8, markersize=5, label=_paper_mode_label(mode))
        ax.set_title(f"{graph_size.capitalize()} graph")
        ax.set_xlabel("Risk aversion α")
        ax.set_xticks(risk_vals)
    axes[0].set_ylabel(ylabel)
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.06), ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(title, y=1.14)
    return _paper_save(fig, output_path)


def paper_plot_ce_cost_tradeoff(summary_rows: Sequence[Dict[str, float]], output_path: Path) -> Dict[str, str] | None:
    rows = [dict(r) for r in summary_rows]
    if not rows:
        return None
    modes = _paper_mode_order([str(r.get("mode")) for r in rows])
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for mi, mode in enumerate(modes):
        xs = [float(r.get("ce_mean_positive_regret_mean", np.nan)) for r in rows if str(r.get("mode")) == mode]
        ys = [float(r.get("eval_realized_cost_mean", np.nan)) for r in rows if str(r.get("mode")) == mode]
        sizes = [45 + 2.5 * int(r.get("agent_count", 0)) for r in rows if str(r.get("mode")) == mode]
        if xs:
            ax.scatter(xs, ys, s=sizes, marker=PAPER_MARKERS[mi % len(PAPER_MARKERS)], alpha=0.75, label=_paper_mode_label(mode))
    ax.set_xlabel("Final empirical CE regret")
    ax.set_ylabel("Final realized social cost")
    ax.set_title("CE–welfare trade-off across all graph sizes and populations")
    ax.legend(frameon=False, ncol=2)
    return _paper_save(fig, output_path)


def generate_paper_ready_figures(
    output_dir: Path,
    all_rows: Sequence[Dict[str, float]],
    final_summary_rows: Sequence[Dict[str, float]],
    per_risk_summary_rows: Sequence[Dict[str, float]],
    seed_tag: str,
) -> None:
    """Generate compact figures intended for the manuscript, not debugging."""
    _paper_style()
    fig_dir = output_dir / "paper_figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    manifest: List[Dict[str, str]] = []

    def add(item: Dict[str, str] | None, kind: str, caption: str) -> None:
        if item is None:
            return
        manifest.append({"plot_kind": kind, "png": str(Path(item["png"]).relative_to(output_dir)), "pdf": str(Path(item["pdf"]).relative_to(output_dir)), "caption_suggestion": caption})

    add(
        paper_plot_final_metric_grid(
            final_summary_rows,
            metric_mean_key="ce_mean_positive_regret_mean",
            metric_ci_key="ce_mean_positive_regret_ci95",
            ylabel="Final empirical CE regret",
            output_path=fig_dir / f"paper_main_final_ce__seeds={seed_tag}.png",
            title="Final empirical CE diagnostic across graph sizes and populations",
        ),
        "paper_main_final_ce",
        "Final empirical CE regret for PPO, DeepCFR, Guided PPO, and Guided PPO + KL. Points show seed means; error bars show 95% confidence intervals.",
    )
    add(
        paper_plot_final_metric_grid(
            final_summary_rows,
            metric_mean_key="eval_realized_cost_mean",
            metric_ci_key="eval_realized_cost_ci95",
            ylabel="Final realized social cost",
            output_path=fig_dir / f"paper_main_final_cost__seeds={seed_tag}.png",
            title="Final realized social cost across graph sizes and populations",
        ),
        "paper_main_final_cost",
        "Final realized social cost under the same settings as the CE plot. Lower values indicate better welfare performance.",
    )

    # Key convergence curves: medium N=20 if available; otherwise the first available graph/N.
    graph_values = _paper_graph_order([str(r.get("graph_size")) for r in all_rows])
    n_values = sorted({int(r.get("agent_count", 0)) for r in all_rows})
    key_graph = "medium" if "medium" in graph_values else (graph_values[0] if graph_values else "")
    key_n = 20 if 20 in n_values else (n_values[0] if n_values else 0)
    if key_graph and key_n:
        add(
            paper_plot_learning_curves(
                all_rows, graph_size=key_graph, agent_count=key_n,
                metric_key="ce_mean_positive_regret",
                ylabel="Empirical CE regret",
                output_path=fig_dir / f"paper_curve_ce__graph={key_graph}__N={key_n}__seeds={seed_tag}.png",
                title=f"CE convergence on {key_graph} graph, N={key_n}",
            ),
            "paper_curve_ce_key_setting",
            f"Training curve of empirical CE regret on the representative {key_graph} graph with N={key_n}. Shaded regions show 95% confidence intervals across seeds.",
        )
        add(
            paper_plot_learning_curves(
                all_rows, graph_size=key_graph, agent_count=key_n,
                metric_key="eval_realized_cost_mean",
                ylabel="Realized social cost",
                output_path=fig_dir / f"paper_curve_cost__graph={key_graph}__N={key_n}__seeds={seed_tag}.png",
                title=f"Cost convergence on {key_graph} graph, N={key_n}",
            ),
            "paper_curve_cost_key_setting",
            f"Training curve of realized social cost on the representative {key_graph} graph with N={key_n}.",
        )

    gap_rows = paper_build_gap_closure_table(final_summary_rows)
    save_rows_csv(gap_rows, fig_dir / f"paper_ce_gap_closure_table__seeds={seed_tag}.csv")
    add(
        paper_plot_gap_closure(gap_rows, fig_dir / f"paper_ce_gap_closure__seeds={seed_tag}.png"),
        "paper_ce_gap_closure",
        "Fraction of the CE gap between PPO and DeepCFR closed by Guided PPO. Values above zero indicate that guidance moves PPO toward the DeepCFR CE reference.",
    )
    add(
        paper_plot_kl_ablation(gap_rows, fig_dir / f"paper_kl_ablation_delta__seeds={seed_tag}.png"),
        "paper_kl_ablation_delta",
        "KL distillation ablation. Positive values mean the KL variant is worse than behavior-only guidance.",
    )
    add(
        paper_plot_ce_cost_tradeoff(final_summary_rows, fig_dir / f"paper_ce_cost_tradeoff__seeds={seed_tag}.png"),
        "paper_ce_cost_tradeoff",
        "CE–welfare trade-off across all graph sizes and populations. Marker size is proportional to the number of agents.",
    )

    if per_risk_summary_rows:
        risk_n = 20 if 20 in {int(r.get("agent_count", 0)) for r in per_risk_summary_rows} else int(per_risk_summary_rows[0].get("agent_count", 0))
        add(
            paper_plot_per_risk_metric(
                per_risk_summary_rows, agent_count=risk_n,
                metric_mean_key="risk_ce_mean_positive_regret_mean",
                metric_ci_key="risk_ce_mean_positive_regret_ci95",
                ylabel="Final empirical CE regret",
                output_path=fig_dir / f"paper_per_risk_ce__N={risk_n}__seeds={seed_tag}.png",
                title=f"Risk-conditioned CE diagnostic, N={risk_n}",
            ),
            "paper_per_risk_ce",
            f"Final CE regret by risk-aversion coefficient α for N={risk_n}. This figure checks whether guidance is risk-conditioned rather than only improving the aggregate average.",
        )
        add(
            paper_plot_per_risk_metric(
                per_risk_summary_rows, agent_count=risk_n,
                metric_mean_key="risk_eval_realized_cost_mean",
                metric_ci_key="risk_eval_realized_cost_ci95",
                ylabel="Final realized cost",
                output_path=fig_dir / f"paper_per_risk_cost__N={risk_n}__seeds={seed_tag}.png",
                title=f"Risk-conditioned realized cost, N={risk_n}",
            ),
            "paper_per_risk_cost",
            f"Final realized cost by risk-aversion coefficient α for N={risk_n}.",
        )

    (fig_dir / "paper_figure_manifest.json").write_text(json.dumps({"figures": manifest}, indent=2, ensure_ascii=False), encoding="utf-8")

def main_shared_teacher() -> None:
    args = build_parser().parse_args()
    output_dir = Path(args.output_dir)
    logger = RunLogger(output_dir=output_dir, verbose=args.verbose)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    risk_alphas = parse_float_tuple(args.risk_alphas)
    agent_counts = parse_int_tuple(args.agent_counts) or (args.num_agents,)
    raw_modes = parse_str_tuple(args.modes)
    alias_map = {
        "ppo": "ppo", "rl": "ppo",
        "deepcfr": "deepcfr",
        "deepcfr_rl": "deepcfr_rl_no_distill",
        "deepcfr_rl_distill": "deepcfr_rl_distill",
        "deepcfr_rl_no_distill": "deepcfr_rl_no_distill",
        "guided": "deepcfr_rl_no_distill",
        "guided_kl": "deepcfr_rl_distill",
    }
    unsupported = [m for m in raw_modes if m not in alias_map]
    if unsupported:
        raise ValueError(f"Unsupported modes: {unsupported}; supported={sorted(alias_map)}")
    modes = tuple(dict.fromkeys(alias_map[m] for m in raw_modes))
    graph_size_map = _resolve_graph_size_map(args)
    graph_size_meta = {name: _graph_metadata(widths, topology=args.topology, base_capacity=args.base_capacity) for name, widths in graph_size_map.items()}

    env_cfg = EnvConfig(num_agents=args.num_agents, horizon=args.horizon, reward_mode=args.reward_mode)
    encoder_cfg = EncoderConfig()
    ppo_cfg = PPOConfig(
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, gamma=args.gamma, gae_lambda=args.gae_lambda,
        clip_ratio=args.clip_ratio, entropy_coef=args.entropy_coef, value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm, hidden_dim=args.hidden_dim, node_embed_dim=args.node_embed_dim,
        ppo_epochs=args.ppo_epochs, minibatch_size=args.minibatch_size,
        episodes_per_update=args.episodes_per_update, total_updates=args.total_updates,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
    )
    deep_teacher_cfg = DeepTeacherConfig(
        episodes_per_update=args.episodes_per_update, total_updates=args.total_updates,
        eval_every=args.eval_every, eval_episodes=args.eval_episodes,
        eval_policy=args.teacher_eval_policy, hidden_dim=args.deep_teacher_hidden_dim,
        node_embed_dim=args.deep_teacher_node_embed_dim, lr=args.deep_teacher_lr,
        buffer_capacity=args.deep_teacher_buffer_capacity, train_steps=args.deep_teacher_train_steps,
        strategy_train_steps=args.deep_teacher_strategy_train_steps, batch_size=args.deep_teacher_batch_size,
        min_buffer_before_train=args.deep_teacher_min_buffer,
        retrain_from_scratch=args.deep_teacher_retrain_from_scratch,
        target_lookahead_steps=args.deep_teacher_target_lookahead_steps,
        target_policy_mode=args.deep_teacher_target_policy_mode,
        target_future_load_floor=args.deep_teacher_target_future_load_floor,
    )
    hybrid_cfg = HybridConfig(
        teacher_warmup_updates=args.hybrid_teacher_warmup_updates,
        teacher_policy_mode=args.hybrid_teacher_policy_mode,
        behavior_beta_start=args.hybrid_beta_start,
        behavior_beta_end=args.hybrid_beta_end,
        behavior_beta_floor=args.hybrid_beta_floor,
        mix_conf_threshold=args.hybrid_mix_conf_threshold,
        mix_conf_scale=args.hybrid_mix_conf_scale,
        mix_js_threshold=args.hybrid_mix_js_threshold,
        mix_js_scale=args.hybrid_mix_js_scale,
        mix_min_js_gate=args.hybrid_mix_min_js_gate,
        distill_lambda_max=args.distill_lambda_max,
        distill_temperature=args.distill_temperature,
        distill_conf_threshold=args.distill_conf_threshold,
        distill_conf_scale=args.distill_conf_scale,
        distill_gap_threshold=args.distill_gap_threshold,
        distill_gap_scale=args.distill_gap_scale,
        distill_cf_advantage_threshold=args.distill_cf_advantage_threshold,
        distill_cf_advantage_scale=args.distill_cf_advantage_scale,
        distill_ramp_updates=args.distill_ramp_updates,
        distill_min_branching_actions=args.distill_min_branching_actions,
        adaptive_snapshot=args.adaptive_teacher_snapshot,
        teacher_freeze_min_updates=args.teacher_freeze_min_updates,
        teacher_freeze_max_updates=args.teacher_freeze_max_updates,
        teacher_ready_ce_ucb_threshold=args.teacher_ready_ce_ucb_threshold,
        teacher_ready_stability_window=args.teacher_ready_stability_window,
        teacher_ready_stability_delta=args.teacher_ready_stability_delta,
        teacher_ready_min_confidence=args.teacher_ready_min_confidence,
        distill_decay_start_fraction=args.distill_decay_start_fraction,
    )
    eval_cfg = EvalConfig(args.epsilon_ce, args.stability_threshold, args.trailing_window)
    seed_tag = make_seed_tag(args.seeds)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "graph_sizes": graph_size_meta,
        "base_capacity": args.base_capacity,
        "agent_counts": agent_counts,
        "risk_alphas": risk_alphas,
        "population_templates": {f"N_{n}": summarize_population(n, risk_alphas) for n in agent_counts},
        "modes": modes,
        "env_cfg": asdict(env_cfg), "encoder_cfg": asdict(encoder_cfg),
        "ppo_cfg": asdict(ppo_cfg), "deep_teacher_cfg": asdict(deep_teacher_cfg),
        "hybrid_cfg": asdict(hybrid_cfg), "eval_cfg": asdict(eval_cfg),
        "seeds": args.seeds, "device": str(device),
        "note": "Shared-teacher risk-stratified run: one DeepCFR teacher snapshot is trained per graph/N/seed and reused by deepcfr, no-distill guidance, and KL ablation.",
    }
    logger.write_json("run_manifest.json", manifest)

    all_rows: List[Dict[str, float]] = []
    summary_by_graph_size_mode_and_count: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

    for graph_size, widths in graph_size_map.items():
        meta = graph_size_meta[graph_size]
        logger.summary(f"\n######## graph_size={graph_size} widths={meta['widths_str']} nodes={meta['num_nodes']} edges={meta['num_edges']} family={meta['topology']} ########")
        summary_by_graph_size_mode_and_count[graph_size] = {m: {} for m in modes}
        size_dir = output_dir / f"graph_size={graph_size}"
        size_dir.mkdir(parents=True, exist_ok=True)
        size_rows: List[Dict[str, float]] = []
        for agent_count in agent_counts:
            logger.summary(f"\n-- graph_size={graph_size} shared-teacher block population N={agent_count}")
            for seed in args.seeds:
                teacher_snapshot: Dict[str, object] | None = None
                teacher_rows: List[Dict[str, float]] = []
                teacher_info: Dict[str, float] = {}
                need_teacher = any(m in modes for m in ("deepcfr", "deepcfr_rl_no_distill", "deepcfr_rl_distill"))
                if need_teacher:
                    logger.summary(f"\n=== Training shared teacher graph_size={graph_size} N={agent_count} seed={seed} on device={device} ===")
                    teacher_snapshot, teacher_rows, teacher_info = train_shared_teacher_snapshot(
                        seed=seed, agent_count=agent_count, widths=widths, topology=args.topology,
                        base_capacity=args.base_capacity, risk_alphas=risk_alphas,
                        env_cfg=env_cfg, encoder_cfg=encoder_cfg, ppo_cfg=ppo_cfg,
                        deep_teacher_cfg=deep_teacher_cfg, hybrid_cfg=hybrid_cfg,
                        eval_cfg=eval_cfg, device=device, logger=logger,
                    )
                for mode in modes:
                    logger.summary(f"\n=== Running graph_size={graph_size} mode={mode} N={agent_count} seed={seed} ===")
                    if mode == "ppo":
                        seed_rows = run_single_seed_ppo(
                            seed=seed, agent_count=agent_count, widths=widths,
                            topology=args.topology, base_capacity=args.base_capacity,
                            risk_alphas=risk_alphas, env_cfg=env_cfg, encoder_cfg=encoder_cfg,
                            ppo_cfg=ppo_cfg, eval_cfg=eval_cfg, device=device, logger=logger,
                        )
                    elif mode == "deepcfr":
                        seed_rows = list(teacher_rows)
                    else:
                        assert teacher_snapshot is not None
                        seed_rows = run_single_seed_guided_with_shared_teacher(
                            seed=seed, mode_name=mode, use_distill=(mode == "deepcfr_rl_distill"),
                            teacher_snapshot=teacher_snapshot, teacher_info=teacher_info,
                            agent_count=agent_count, widths=widths, topology=args.topology,
                            base_capacity=args.base_capacity, risk_alphas=risk_alphas,
                            env_cfg=env_cfg, encoder_cfg=encoder_cfg, ppo_cfg=ppo_cfg,
                            deep_teacher_cfg=deep_teacher_cfg, hybrid_cfg=hybrid_cfg,
                            eval_cfg=eval_cfg, device=device, logger=logger,
                        )
                    _annotate_rows(seed_rows, mode, graph_size, args.topology, meta)
                    mode_dir = size_dir / mode
                    count_dir = mode_dir / f"N_{agent_count}"
                    count_dir.mkdir(parents=True, exist_ok=True)
                    save_rows_csv(seed_rows, count_dir / f"metrics_seed_{seed}.csv")
                    plot_seed_curves(
                        seed_rows,
                        count_dir / f"plot_kind=seed_dual_metric__graph_size={graph_size}__family={args.topology}__mode={mode}__N={agent_count}__seed={seed}.png",
                        title=f"plot_kind=seed_dual_metric | graph_size={graph_size} | family={args.topology} | mode={mode} | N={agent_count} | seed={seed} | metric=CE+cost",
                    )
                    all_rows.extend(seed_rows)
                    size_rows.extend(seed_rows)

        # Save mode/count aggregations for this graph size.
        for mode in modes:
            mode_rows = [dict(r) for r in size_rows if str(r.get("mode")) == mode]
            mode_dir = size_dir / mode
            mode_dir.mkdir(parents=True, exist_ok=True)
            if mode_rows:
                save_rows_csv(mode_rows, mode_dir / f"all_results__graph_size={graph_size}__family={args.topology}__mode={mode}__seeds={seed_tag}.csv")
                for agent_count in agent_counts:
                    count_rows = [dict(r) for r in mode_rows if int(r.get("agent_count", -1)) == agent_count]
                    if count_rows:
                        summary = summarize_final_rows(count_rows)
                        summary_by_graph_size_mode_and_count[graph_size][mode][f"N_{agent_count}"] = summary
                        logger.write_json(str(Path(f"graph_size={graph_size}") / mode / f"N_{agent_count}" / "summary_final.json"), summary)
                        plot_aggregate_curves(
                            count_rows,
                            mode_dir / f"N_{agent_count}" / f"plot_kind=aggregate_dual_metric__graph_size={graph_size}__family={args.topology}__mode={mode}__N={agent_count}__seeds={seed_tag}.png",
                            title=f"plot_kind=aggregate_dual_metric | graph_size={graph_size} | family={args.topology} | mode={mode} | N={agent_count} | seeds={seed_tag} | metric=CE+cost",
                        )
        for agent_count in agent_counts:
            filtered = [dict(r) for r in size_rows if int(r.get("agent_count", -1)) == agent_count]
            if filtered:
                plot_metric_by_mode_for_agent_count(
                    filtered, agent_count=agent_count, metric_key="ce_mean_positive_regret",
                    output_path=size_dir / f"plot_kind=ce_compare_modes__graph_size={graph_size}__family={args.topology}__N={agent_count}__seeds={seed_tag}.png",
                    title=f"plot_kind=ce_compare_modes | graph_size={graph_size} | family={args.topology} | N={agent_count} | seeds={seed_tag} | metric=ce_mean_positive_regret",
                    ylabel="Mean positive regret",
                )
                plot_metric_by_mode_for_agent_count(
                    filtered, agent_count=agent_count, metric_key="eval_realized_cost_mean",
                    output_path=size_dir / f"plot_kind=cost_compare_modes__graph_size={graph_size}__family={args.topology}__N={agent_count}__seeds={seed_tag}.png",
                    title=f"plot_kind=cost_compare_modes | graph_size={graph_size} | family={args.topology} | N={agent_count} | seeds={seed_tag} | metric=eval_realized_cost_mean",
                    ylabel="Eval realized cost",
                )

    if not all_rows:
        raise RuntimeError("No results produced.")
    save_rows_csv(all_rows, output_dir / f"all_results__seeds={seed_tag}.csv")
    logger.write_json(f"summary_by_graph_size_mode_and_agent_count__seeds={seed_tag}.json", summary_by_graph_size_mode_and_count)
    final_rows = extract_final_rows_by_mode_graph_size_count_seed(all_rows)
    save_rows_csv(final_rows, output_dir / f"final_rows__seeds={seed_tag}.csv")
    final_summary_rows = build_final_summary_rows_with_graph_size(final_rows)
    save_rows_csv(final_summary_rows, output_dir / f"final_summary_table__seeds={seed_tag}.csv")
    per_risk_summary_rows = build_final_per_risk_summary_rows(final_rows)
    save_rows_csv(per_risk_summary_rows, output_dir / f"final_per_risk_summary_table__seeds={seed_tag}.csv")

    for graph_size in graph_size_map.keys():
        graph_dir = output_dir / f"graph_size={graph_size}"
        plot_final_metric_bars_by_mode(
            final_rows, graph_size=graph_size, family=args.topology,
            metric_key="ce_mean_positive_regret",
            output_path=graph_dir / f"plot_kind=final_ce_bar_by_mode__graph_size={graph_size}__family={args.topology}__seeds={seed_tag}.png",
            title=f"plot_kind=final_ce_bar_by_mode | graph_size={graph_size} | family={args.topology} | seeds={seed_tag} | metric=ce_mean_positive_regret",
            ylabel="Final mean positive regret",
        )
        plot_final_metric_bars_by_mode(
            final_rows, graph_size=graph_size, family=args.topology,
            metric_key="eval_realized_cost_mean",
            output_path=graph_dir / f"plot_kind=final_cost_bar_by_mode__graph_size={graph_size}__family={args.topology}__seeds={seed_tag}.png",
            title=f"plot_kind=final_cost_bar_by_mode | graph_size={graph_size} | family={args.topology} | seeds={seed_tag} | metric=eval_realized_cost_mean",
            ylabel="Final eval realized cost",
        )
        plot_final_per_risk_metric_bars(
            per_risk_summary_rows, graph_size=graph_size, family=args.topology,
            metric_key="risk_ce_mean_positive_regret_mean",
            output_path=graph_dir / f"plot_kind=final_ce_by_risk_and_mode__graph_size={graph_size}__family={args.topology}__seeds={seed_tag}.png",
            title=f"plot_kind=final_ce_by_risk_and_mode | graph_size={graph_size} | family={args.topology} | seeds={seed_tag}",
            ylabel="Final mean positive regret",
        )
        plot_final_per_risk_metric_bars(
            per_risk_summary_rows, graph_size=graph_size, family=args.topology,
            metric_key="risk_eval_realized_cost_mean",
            output_path=graph_dir / f"plot_kind=final_cost_by_risk_and_mode__graph_size={graph_size}__family={args.topology}__seeds={seed_tag}.png",
            title=f"plot_kind=final_cost_by_risk_and_mode | graph_size={graph_size} | family={args.topology} | seeds={seed_tag}",
            ylabel="Final realized cost by risk type",
        )
    generate_paper_ready_figures(
        output_dir=output_dir,
        all_rows=all_rows,
        final_summary_rows=final_summary_rows,
        per_risk_summary_rows=per_risk_summary_rows,
        seed_tag=seed_tag,
    )
    plot_manifest = sorted([str(path.relative_to(output_dir)) for path in output_dir.rglob("*.png")])
    logger.write_json("plot_manifest.json", {"files": plot_manifest})
    logger.summary("\nFinished shared-teacher four-mode guided runs.")
    logger.summary(f"Results saved under: {output_dir.resolve()}")


if __name__ == "__main__":
    main_shared_teacher()
