# S-DNCG Experiments

This repository contains the reproducibility code for the S-DNCG experiments reported in the manuscript.

## Overview

The code implements a stochastic dynamic network congestion game with heterogeneous risk aversion and compares four methods:

1. PPO-only
2. DeepCFR-style teacher-only
3. DeepCFR-guided PPO without KL distillation
4. DeepCFR-guided PPO with KL distillation

## Main Script

The main experiment file is:

```bash
main_experiment.py
[200~```

## Environment

The experiments require Python 3.9 or later. Install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Running the Experiments

The main experiment can be executed by running:

```bash
python main_experiment.py
```

The script uses the default configuration specified in the code, including the graph scales, population sizes, risk-aversion settings, method arms, and random seeds used in the manuscript.

## Reproducibility Note

The experiments are computationally expensive because each graph-population-seed block trains and selects a DeepCFR-style teacher before evaluating the teacher and training the PPO-based variants. The reported seed-level intervals in the manuscript are descriptive uncertainty summaries rather than formal large-sample hypothesis-test evidence.
EOF~
```

## Environment

The experiments require Python 3.9 or later. Install the dependencies by running:

```bash
pip install -r requirements.txt
```

## Running the Experiments

The main experiment can be executed by running:

```bash
python main_experiment.py
```

The script uses the default configuration specified in the code, including the graph scales, population sizes, risk-aversion settings, method arms, and random seeds used in the manuscript.

## Reproducibility Note

The experiments are computationally expensive because each graph-population-seed block trains and selects a DeepCFR-style teacher before evaluating the teacher and training the PPO-based variants. The reported seed-level intervals in the manuscript are descriptive uncertainty summaries rather than formal large-sample hypothesis-test evidence.
