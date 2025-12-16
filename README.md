# Strategic Punishment: A Computational Model of Reputational Signaling

This repository contains the code and results for a computational modeling project exploring how strategic considerations about audience beliefs influence punishment behavior. This is my 9.660 Fall 2025 final project.

## Overview

This project implements a computational model comparing **naive** and **strategic** punishers. The key distinction is that strategic punishers consider how their actions will be interpreted by different audiences (in-group vs. out-group observers), while naive punishers only consider the intrinsic utility of punishment actions.

### Core Theoretical Framework

The model assumes that punishers have hidden mental states:
- **Wrongness** (w): How wrong the act is (0.0 = not wrong, 0.5 = somewhat, 1.0 = fully wrong)
- **Bias** (b): Punisher's bias toward the target (-1.0 = against, 0.0 = neutral, 1.0 = for)
- **Justice** (j): Whether the punisher values justice (0 = unjust, 1 = just)
- **Selfishness** (s): Self-interest level (0 = altruistic, 1 = selfish)

Observers (both in-group and out-group) use **inverse planning** to infer these hidden states from observed punishment actions. Strategic punishers then optimize their actions to shape these inferences, creating a signaling game.

## Main Ideas

### The Strategic Punishment Hypothesis

When punishers care about their reputation with different audiences, they face a dilemma:
- **In-group observers** may value justice and want harsh punishment for wrong acts
- **Out-group observers** may interpret harsh punishment as evidence of bias or hostility

Strategic punishers navigate this tension by choosing actions that balance:
1. Direct utility (matching punishment to wrongness)
2. Reputational utility (how different audiences will interpret the action)

This leads to several counterintuitive predictions:
- **Moderation under polarization**: As out-group distrust increases, strategic punishers become less harsh to avoid appearing biased
- **Divergent signaling**: The same action can signal different things to different audiences (dog whistle effects)
- **Over/under-punishment**: Strategic punishers may punish more or less harshly than the act warrants, depending on audience concerns

## Experiments

The project includes four computational experiments that test different aspects of strategic punishment:

### Experiment 1: The Politician's Dilemma (Polarization)

**Question**: How does increasing polarization (out-group distrust) affect punishment severity?

**Setup**: Varies the out-group's prior belief that the punisher is biased against them, from 0.0 (trust) to 1.0 (complete distrust).


**Output**: `exp1_results.json` → `exp1_bifurcation.png`

### Experiment 2: Divergent Signaling (Dog Whistle)

**Question**: How do strategic punishers behave when in-group and out-group have different beliefs about the act's wrongness?

**Setup**: In-group knows the act is fully wrong (w=1.0), while out-group believes it's not wrong (w=0.0). Both groups care about the punisher's justice and bias.

**Output**: `exp2_results.json` → `exp2_action_bar.png`, `exp2_posteriors.png`

### Experiment 3: Martyrdom / Signaling Commitment

**Question**: How does concern for out-group reputation affect punishment of minor infractions?

**Setup**: Tests a minor infraction (wrongness=0.5) while varying the weight placed on out-group reputation.

**Output**: `exp3_results.json` → `exp3_martyrdom.png`, `exp3_severity_curve.png`

### Experiment 4: Recursive Dynamics

**Question**: How do beliefs evolve over multiple rounds of interaction?

**Setup**: Simulates 4 rounds where punishers act, observers update beliefs, and these updated beliefs become priors for the next round.

**Output**: `exp4_results.json` → `exp4_dynamics.png`

## Usage

### Prerequisites

The project requires:
- Python 3.8+
- JAX (for numerical computation)
- NumPy
- Matplotlib (for plotting)
- The `memo` library (for probabilistic programming with inverse planning)

Install dependencies:
```bash
pip install jax numpy matplotlib memo
```

Note: JAX is configured to use CPU only in this project. If you have a GPU, you can modify the configuration in `experiments.py`.

### Running Experiments

To run all experiments and generate results:

```bash
python experiments.py
```

This will:
1. Create an `experiment_results/` directory if it doesn't exist
2. Run all four experiments sequentially
3. Save results as JSON files in `experiment_results/`

### Generating Plots

After running experiments, generate visualizations:

```bash
python create_plots.py
```

This will:
1. Load results from `experiment_results/`
2. Generate publication-quality plots (PNG and PDF formats)
3. Save plots to the `plots/` directory
4. Create a summary figure combining all key results

### File Structure

```
code/
├── experiments.py           # Main experiment runner and model implementation
├── create_plots.py          # Visualization script
├── README.md               # This file
├── experiment_results/     # JSON results from experiments
│   ├── exp1_results.json
│   ├── exp2_results.json
│   ├── exp3_results.json
│   └── exp4_results.json
└── plots/                  # Generated visualizations
    ├── exp1_bifurcation.png/pdf
    ├── exp2_action_bar.png/pdf
    ├── exp2_posteriors.png/pdf
    ├── exp3_martyrdom.png/pdf
    ├── exp3_severity_curve.png/pdf
    ├── exp4_dynamics.png/pdf
    └── summary_figure.png/pdf
```

## Model Details

### Naive Punisher

The naive punisher chooses actions based solely on intrinsic utility:
- Computes utility for each action (None, Mild, Harsh) given their true mental state
- Uses softmax with temperature α=5.0 to select actions probabilistically
- Does not consider how observers will interpret their actions

### Strategic Punisher

The strategic punisher:
1. **Forward models observer inference**: Uses inverse planning to predict how in-group and out-group observers will update their beliefs about the punisher's hidden states
2. **Computes reputational utility**: Values being seen as just (by in-group) and avoiding being seen as biased (by out-group)
3. **Optimizes action selection**: Chooses actions that maximize combined direct + reputational utility

### Observer Inference

Observers use Bayesian inverse planning:
- Start with priors over punisher's hidden states (wrongness, bias, justice, selfishness)
- Model the punisher as a rational agent choosing actions based on utility
- Update beliefs after observing an action using Bayes' rule
- The implementation uses the `memo` library for efficient probabilistic inference

## Key Parameters

- **ALPHA_RATIONALITY** (5.0): Softmax temperature controlling action selection determinism
- **WRONGNESS_VALUES**: [0.0, 0.5, 1.0] - Discrete wrongness levels
- **BIAS_VALUES**: [-1.0, 0.0, 1.0] - Bias toward target
- **ACTION_SEVERITY**: [0.0, 0.5, 1.0] - Severity of None, Mild, Harsh actions
- **Reputation weights**: Control how much punishers care about in-group vs. out-group beliefs

## Notes

- The model uses discrete state spaces for computational tractability
- All inference is done via enumeration over the discrete state space
- Results are deterministic given fixed priors and parameters
- The plots use publication-quality styling suitable for academic presentations
