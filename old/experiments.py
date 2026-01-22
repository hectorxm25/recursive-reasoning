import os
import json
import jax.numpy as jnp
import numpy as np
from memo import memo
import jax

# Force JAX to use CPU only
jax.config.update('jax_platform_name', 'cpu')

# -----------------------------------------------------------------------------
# CONFIGURATION & GLOBAL CONSTANTS
# -----------------------------------------------------------------------------

# Discrete States for Enumerative Inference (as Python lists for memo)
WRONGNESS_LEVELS = [0, 1, 2]  # Indices: 0=Not Wrong, 1=Somewhat, 2=Wrong
BIAS_LEVELS = [0, 1, 2]       # Indices: 0=Against, 1=Neutral, 2=For  
JUSTICE_LEVELS = [0, 1]       # Indices: 0=Unjust, 1=Just
SELFISH_LEVELS = [0, 1]       # Indices: 0=Altruistic, 1=Selfish
ACTIONS = [0, 1, 2]           # None, Mild, Harsh

# Map indices to actual values
WRONGNESS_VALUES = jnp.array([0.0, 0.5, 1.0])
BIAS_VALUES = jnp.array([-1.0, 0.0, 1.0])
JUSTICE_VALUES = jnp.array([0.0, 1.0])
SELFISH_VALUES = jnp.array([0.0, 1.0])
ACTION_SEVERITY = jnp.array([0.0, 0.5, 1.0])

# Utility Constants
COST_TARGET = jnp.array([0.0, -1.0, -2.0])
COST_SELF = jnp.array([0.0, -0.1, -0.2])

# Softmax Temperature
ALPHA_RATIONALITY = 5.0

# -----------------------------------------------------------------------------
# UTILITY FUNCTIONS (Pure JAX)
# -----------------------------------------------------------------------------

@jax.jit
def compute_action_utility(a_idx, w_idx, b_idx, j_idx, s_idx):
    """Compute utility for action given state indices."""
    w = WRONGNESS_VALUES[w_idx]
    b = BIAS_VALUES[b_idx]
    j = JUSTICE_VALUES[j_idx]
    s = SELFISH_VALUES[s_idx]
    
    severity = ACTION_SEVERITY[a_idx]
    cost_target = COST_TARGET[a_idx]
    cost_self = COST_SELF[a_idx]
    
    u_bias = b * cost_target
    u_self = (1.0 + s) * cost_self
    u_justice = j * (-(severity - w)**2) * 4.0
    
    return u_bias + u_self + u_justice

# -----------------------------------------------------------------------------
# NAIVE PUNISHER POLICY (Pure JAX - Forward Inference)
# -----------------------------------------------------------------------------

def naive_punisher_policy_probs(w_idx, b_idx, j_idx, s_idx):
    """
    Compute action probability distribution for naive punisher.
    Uses softmax over utilities.
    Returns: jnp.array of shape (3,) - probabilities for each action
    """
    utilities = jnp.array([
        compute_action_utility(a, w_idx, b_idx, j_idx, s_idx)
        for a in ACTIONS
    ])
    return jax.nn.softmax(ALPHA_RATIONALITY * utilities)

# -----------------------------------------------------------------------------
# OBSERVER INFERENCE (Memo - Inverse Inference)
# -----------------------------------------------------------------------------

@memo
def observer_inference_joint[w: WRONGNESS_LEVELS, b: BIAS_LEVELS, j: JUSTICE_LEVELS, s: SELFISH_LEVELS, a: ACTIONS](
    observed_action, 
    prior_w: ..., prior_b: ..., prior_j: ..., prior_s: ...
):
    """
    Inverse planning: compute joint P(w, b, j, s, a | observed_action, priors).
    The result array contains probability weights (not normalized).
    We condition on the observed action using observes_that.
    """
    # Sample from priors
    observer: chooses(w in WRONGNESS_LEVELS, wpp=array_index(prior_w, w))
    observer: chooses(b in BIAS_LEVELS, wpp=array_index(prior_b, b))
    observer: chooses(j in JUSTICE_LEVELS, wpp=array_index(prior_j, j))
    observer: chooses(s in SELFISH_LEVELS, wpp=array_index(prior_s, s))
    
    # Action choice based on utility
    observer: chooses(a in ACTIONS, wpp=exp(5.0 * compute_action_utility(a, w, b, j, s)))
    
    # Condition on observed action
    observer: observes_that[a == observed_action]
    
    return w
    return b
    return j
    return s
    return a

def compute_observer_posteriors(observed_action, prior_w, prior_b, prior_j, prior_s):
    """
    Compute posterior marginals P(w|a), P(b|a), P(j|a), P(s|a) given observed action.
    Returns dict with marginal probability arrays for each hidden variable.
    """
    # Get result from memo - with 5 returns, shape is (5, w, b, j, s, a)
    result = observer_inference_joint(observed_action, prior_w, prior_b, prior_j, prior_s)
    
  
    num_w = len(WRONGNESS_LEVELS)
    num_b = len(BIAS_LEVELS)
    num_j = len(JUSTICE_LEVELS)
    num_s = len(SELFISH_LEVELS)
    
    # Create meshgrid of all state combinations
    w_grid, b_grid, j_grid, s_grid = jnp.meshgrid(
        jnp.arange(num_w), jnp.arange(num_b), 
        jnp.arange(num_j), jnp.arange(num_s),
        indexing='ij'
    )
    
    # Prior probabilities
    pw_grid = prior_w[w_grid]
    pb_grid = prior_b[b_grid]
    pj_grid = prior_j[j_grid]
    ps_grid = prior_s[s_grid]
    prior_joint = pw_grid * pb_grid * pj_grid * ps_grid
    
    # Likelihood: P(observed_action | w, b, j, s)
    # Compute action probabilities for each state combination
    def compute_likelihood(w_idx, b_idx, j_idx, s_idx):
        probs = naive_punisher_policy_probs(w_idx, b_idx, j_idx, s_idx)
        return probs[observed_action]
    
    # Vectorize over all states
    likelihood = jnp.zeros((num_w, num_b, num_j, num_s))
    for wi in range(num_w):
        for bi in range(num_b):
            for ji in range(num_j):
                for si in range(num_s):
                    likelihood = likelihood.at[wi, bi, ji, si].set(
                        naive_punisher_policy_probs(wi, bi, ji, si)[observed_action]
                    )
    
    # Posterior (unnormalized)
    posterior = prior_joint * likelihood
    
    # Normalize
    total = jnp.sum(posterior) + 1e-10
    posterior = posterior / total
    
    # Compute marginals
    marginals = {
        'w': jnp.sum(posterior, axis=(1, 2, 3)),
        'b': jnp.sum(posterior, axis=(0, 2, 3)),
        'j': jnp.sum(posterior, axis=(0, 1, 3)),
        's': jnp.sum(posterior, axis=(0, 1, 2)),
    }
    
    return marginals

# -----------------------------------------------------------------------------
# STRATEGIC PUNISHER (Pure JAX - Forward Planning)
# -----------------------------------------------------------------------------

def compute_strategic_utility(action, in_group_priors, out_group_priors, weights):
    """
    Compute reputational utility for a strategic punisher choosing an action.
    """
    w_justice_in = weights.get('justice_in', 0.0)
    w_justice_out = weights.get('justice_out', 0.0)
    w_wrong_in = weights.get('wrongness_in', 0.0)
    
    # Get posterior marginals for in-group and out-group
    post_in = compute_observer_posteriors(
        action,
        in_group_priors['w'], in_group_priors['b'],
        in_group_priors['j'], in_group_priors['s']
    )
    
    post_out = compute_observer_posteriors(
        action,
        out_group_priors['w'], out_group_priors['b'],
        out_group_priors['j'], out_group_priors['s']
    )
    
    # P(Just | action) for in-group: index 1 means Just
    prob_just_in = post_in['j'][1]
    prob_just_out = post_out['j'][1]
    
    # P(Wrongness=High | action) for in-group: index 2 means fully wrong
    prob_wrong_in = post_in['w'][2]
    
    # Total utility
    u_strat = (w_justice_in * prob_just_in + 
               w_justice_out * prob_just_out + 
               w_wrong_in * prob_wrong_in)
    
    return u_strat

def strategic_punisher_policy(in_group_priors, out_group_priors, weights):
    """
    Strategic punisher chooses action to maximize reputational utility.
    Returns probability distribution over actions.
    """
    utilities = jnp.array([
        compute_strategic_utility(a, in_group_priors, out_group_priors, weights)
        for a in ACTIONS
    ])
    
    return jax.nn.softmax(ALPHA_RATIONALITY * utilities)

# -----------------------------------------------------------------------------
# EXPERIMENT RUNNERS
# -----------------------------------------------------------------------------

def run_experiment_1_polarization(output_dir):
    """
    Exp 1: The Politician's Dilemma
    Vary Out-Group bias prior (Polarization Distance).
    """
    print("Running Experiment 1: Polarization Distance...")
    results = []
    
    weights = {'justice_in': 1.0, 'justice_out': 1.0, 'wrongness_in': 0.0}
    distrust_levels = np.linspace(0.0, 1.0, 11)
    
    for distrust in distrust_levels:
        # In-Group: Neutral bias prior
        priors_in = {
            'w': jnp.array([0.33, 0.33, 0.34]),
            'b': jnp.array([0.0, 1.0, 0.0]),  # Certain Neutral (idx 1)
            'j': jnp.array([0.5, 0.5]),
            's': jnp.array([0.5, 0.5])
        }
        
        # Out-Group: Varying belief in Bias=Against (idx 0)
        p_bias_neg = distrust
        p_bias_neu = 1.0 - distrust
        priors_out = {
            'w': jnp.array([0.33, 0.33, 0.34]),
            'b': jnp.array([p_bias_neg, p_bias_neu, 0.0]),
            'j': jnp.array([0.5, 0.5]),
            's': jnp.array([0.5, 0.5])
        }
        
        # Run Strategic Model
        probs = strategic_punisher_policy(priors_in, priors_out, weights)
        
        results.append({
            "polarization_level": float(distrust),
            "prob_none": float(probs[0].item()),
            "prob_mild": float(probs[1].item()),
            "prob_harsh": float(probs[2].item())
        })
        
    with open(os.path.join(output_dir, "exp1_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Experiment 1 complete.")

def run_experiment_2_dog_whistle(output_dir):
    """
    Exp 2: Divergent Signaling (The Dog Whistle)
    """
    print("Running Experiment 2: Dog Whistle...")
    results = []
    
    # In-Group: Knows Wrongness=1.0 (idx 2)
    priors_in = {
        'w': jnp.array([0.0, 0.0, 1.0]),
        'b': jnp.array([0.0, 1.0, 0.0]),
        'j': jnp.array([0.5, 0.5]),
        's': jnp.array([0.5, 0.5])
    }
    # Out-Group: Thinks Wrongness=0.0 (idx 0)
    priors_out = {
        'w': jnp.array([1.0, 0.0, 0.0]),
        'b': jnp.array([0.1, 0.8, 0.1]),
        'j': jnp.array([0.5, 0.5]),
        's': jnp.array([0.5, 0.5])
    }
    
    # Naive punisher (w=2, b=1, j=1, s=0 - Just, unbiased, altruistic, knows wrong)
    naive_probs = naive_punisher_policy_probs(2, 1, 1, 0)
    
    results.append({
        "model_type": "Naive",
        "action_probs": [float(p.item()) for p in naive_probs]
    })
    
    # Strategic punisher
    weights = {'justice_in': 1.0, 'justice_out': 2.0, 'wrongness_in': 0.0}
    strategic_probs = strategic_punisher_policy(priors_in, priors_out, weights)
    
    results.append({
        "model_type": "Strategic",
        "action_probs": [float(p.item()) for p in strategic_probs]
    })
    
    with open(os.path.join(output_dir, "exp2_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Experiment 2 complete.")

def run_experiment_3_martyrdom(output_dir):
    """
    Exp 3: Martyrdom / Signaling Commitment.
    """
    print("Running Experiment 3: Martyrdom...")
    results = []
    
    w_out_range = np.linspace(0.0, 2.0, 5)
    
    # Minor infraction (Wrongness=0.5, idx 1)
    priors_in = {
        'w': jnp.array([0.0, 1.0, 0.0]),
        'b': jnp.array([0.0, 1.0, 0.0]),
        'j': jnp.array([0.5, 0.5]),
        's': jnp.array([0.5, 0.5])
    }
    priors_out = priors_in.copy()
    
    for w_out in w_out_range:
        weights = {'justice_in': 2.0, 'justice_out': w_out, 'wrongness_in': 0.0}
        probs = strategic_punisher_policy(priors_in, priors_out, weights)
        
        results.append({
            "weight_out_group": float(w_out),
            "prob_harsh": float(probs[2].item())
        })
        
    with open(os.path.join(output_dir, "exp3_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Experiment 3 complete.")

def run_experiment_4_dynamics(output_dir):
    """
    Exp 4: Recursive Dynamics.
    """
    print("Running Experiment 4: Recursive Dynamics...")
    results = []
    
    ROUNDS = 4
    MODELS = ['Naive', 'Strategic']
    
    init_in = {
        'w': jnp.array([0.2, 0.6, 0.2]),
        'b': jnp.array([0.0, 1.0, 0.0]),
        'j': jnp.array([0.5, 0.5]),
        's': jnp.array([1.0, 0.0])
    }
    init_out = {
        'w': jnp.array([0.2, 0.6, 0.2]),
        'b': jnp.array([0.8, 0.2, 0.0]),
        'j': jnp.array([0.5, 0.5]),
        's': jnp.array([1.0, 0.0])
    }
    
    # True punisher state: w=1 (somewhat wrong), b=1 (neutral), j=1 (just), s=0 (altruistic)
    true_w, true_b, true_j, true_s = 1, 1, 1, 0
    
    for m_name in MODELS:
        curr_in = {k: v.copy() for k, v in init_in.items()}
        curr_out = {k: v.copy() for k, v in init_out.items()}
        
        history = []
        
        for r in range(ROUNDS):
            if m_name == 'Naive':
                probs = naive_punisher_policy_probs(true_w, true_b, true_j, true_s)
                action = int(jnp.argmax(probs))
            else:
                weights = {'justice_in': 1.0, 'justice_out': 2.0, 'wrongness_in': 0.0}
                probs = strategic_punisher_policy(curr_in, curr_out, weights)
                action = int(jnp.argmax(probs))
            
            # Observers update
            post_in = compute_observer_posteriors(action, curr_in['w'], curr_in['b'], curr_in['j'], curr_in['s'])
            post_out = compute_observer_posteriors(action, curr_out['w'], curr_out['b'], curr_out['j'], curr_out['s'])
            
            # Calculate expected bias for polarization measure
            exp_bias_in = float(jnp.sum(post_in['b'] * BIAS_VALUES))
            exp_bias_out = float(jnp.sum(post_out['b'] * BIAS_VALUES))
            polarization = abs(exp_bias_in - exp_bias_out)
            
            history.append({
                "round": r,
                "action": action,
                "polarization": polarization
            })
            
            # Update priors for next round
            curr_in = {
                'w': post_in['w'],
                'b': post_in['b'],
                'j': post_in['j'],
                's': post_in['s']
            }
            curr_out = {
                'w': post_out['w'],
                'b': post_out['b'],
                'j': post_out['j'],
                's': post_out['s']
            }
            
        results.append({"model": m_name, "history": history})
    
    with open(os.path.join(output_dir, "exp4_results.json"), 'w') as f:
        json.dump(results, f, indent=2)
    print("  Experiment 4 complete.")

# -----------------------------------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    OUTPUT_DIR = "experiment_results"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Starting Strategic Punishment Experiments...")
    print(f"Saving results to: {OUTPUT_DIR}")
    
    run_experiment_1_polarization(OUTPUT_DIR)
    run_experiment_2_dog_whistle(OUTPUT_DIR)
    run_experiment_3_martyrdom(OUTPUT_DIR)
    run_experiment_4_dynamics(OUTPUT_DIR)
    
    print("All experiments completed successfully.")
