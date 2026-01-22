import jax
import jax.numpy as jnp
from jax import vmap
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from scipy.stats import beta

# Force CPU
jax.config.update('jax_platform_name', 'cpu')

# -----------------------------------------------------------------------------
# 1. CONSTANTS & GRID SETUP
# -----------------------------------------------------------------------------
GRID_SIZE = 200
W_GRID = jnp.linspace(0.0, 1.0, GRID_SIZE)
B_GRID = jnp.linspace(-0.5, 0.5, GRID_SIZE) # TODO: change
J_GRID = jnp.linspace(0.0, 1.0, GRID_SIZE)

ACTIONS = jnp.array([0, 1, 2])       # None, Mild, Harsh
SEVERITY = jnp.array([0.0, 0.5, 1.0])
COST_TARGET = jnp.array([0.0, -0.5, -1.0])
COST_SELF = jnp.array([0.0, -0.1, -0.2])

BETA_NAIVE = 10.0  # Observer's assumption of punisher rationality

# -----------------------------------------------------------------------------
# 2. NAIVE UTILITY & LIKELIHOODS
# -----------------------------------------------------------------------------

@jax.jit
def compute_naive_utility(a_idx, w, b, j):
    u_bias = b * COST_TARGET[a_idx]
    dist = jnp.abs(SEVERITY[a_idx] - w)
    u_justice = j * (-dist)
    u_self = COST_SELF[a_idx]
    return u_bias + u_justice + u_self

def precompute_likelihoods():
    w_mesh, b_mesh, j_mesh = jnp.meshgrid(W_GRID, B_GRID, J_GRID, indexing='ij')
    def get_grid_utility(a_idx):
        return compute_naive_utility(a_idx, w_mesh, b_mesh, j_mesh)
    utilities = vmap(get_grid_utility)(ACTIONS)
    # Numerical stability: subtract max before exp
    utilities = utilities - jnp.max(utilities, axis=0)
    probs = jax.nn.softmax(BETA_NAIVE * utilities, axis=0)
    return probs

LIKELIHOOD_TENSOR = precompute_likelihoods()

# -----------------------------------------------------------------------------
# 3. OBSERVER INFERENCE & METRICS
# -----------------------------------------------------------------------------

@jax.jit
def observer_update(prior_tensor, action_idx):
    likelihood = LIKELIHOOD_TENSOR[action_idx]
    unnorm_post = prior_tensor * likelihood
    return unnorm_post / (jnp.sum(unnorm_post) + 1e-10)

@jax.jit
def get_metrics(posterior_tensor):
    # Marginalize
    p_w = jnp.sum(posterior_tensor, axis=(1, 2))
    p_b = jnp.sum(posterior_tensor, axis=(0, 2))
    p_j = jnp.sum(posterior_tensor, axis=(0, 1))
    
    # Expectations
    e_w = jnp.sum(p_w * W_GRID)
    e_b = jnp.sum(p_b * B_GRID)
    e_j = jnp.sum(p_j * J_GRID)
    
    # Std Devs
    var_w = jnp.sum(p_w * (W_GRID**2)) - e_w**2
    var_b = jnp.sum(p_b * (B_GRID**2)) - e_b**2
    var_j = jnp.sum(p_j * (J_GRID**2)) - e_j**2
    
    return {
        'e_w': e_w, 'std_w': jnp.sqrt(jnp.maximum(0.0, var_w)),
        'e_b': e_b, 'std_b': jnp.sqrt(jnp.maximum(0.0, var_b)),
        'e_j': e_j, 'std_j': jnp.sqrt(jnp.maximum(0.0, var_j))
    }

# -----------------------------------------------------------------------------
# 4. STRATEGIC PLANNER
# -----------------------------------------------------------------------------

@jax.jit
def get_strategic_action_probs(prior_in, prior_out, true_w, true_b, true_j, weights):
    
    def utility_for_action(a_idx):
        # A. Intrinsic Utility
        u_int = compute_naive_utility(a_idx, true_w, true_b, true_j)
        
        # B. Reputational Utility
        post_in = observer_update(prior_in, a_idx)
        metrics_in = get_metrics(post_in)
        
        post_out = observer_update(prior_out, a_idx)
        metrics_out = get_metrics(post_out)
        
        # Calculate Reputational Score
        u_rep = (
            weights['w_J_in'] * metrics_in['e_j'] + 
            weights['w_B_in'] * metrics_in['e_b'] +
            weights['w_J_out'] * metrics_out['e_j'] + 
            weights['w_B_out'] * metrics_out['e_b']
        )
        
        return weights['scale_int'] * u_int + weights['scale_rep'] * u_rep

    strat_utilities = vmap(utility_for_action)(ACTIONS)
    
    # Higher beta_strat makes the agent follow the utility more strictly
    return jax.nn.softmax(weights['beta_strat'] * strat_utilities)


# -----------------------------------------------------------------------------
# DIAGNOSTIC HELPER
# -----------------------------------------------------------------------------
def debug_utilities(prior_in, prior_out, true_w, true_b, true_j, weights):
    """
    Prints the raw utility values for Action 0 (None) vs Action 2 (Harsh).
    Call this ONCE at the start of a simulation to sanity check scales.
    """
    print(f"\n--- DEBUG UTILITIES (Weight B_out = {weights['w_B_out']}) ---")
    
    # 1. Calculate Intrinsic
    u_int_none = compute_naive_utility(0, true_w, true_b, true_j)
    u_int_harsh = compute_naive_utility(2, true_w, true_b, true_j)
    
    # 2. Calculate Reputational (Expected Bias b)
    # Simulate update for None
    post_out_none = observer_update(prior_out, 0)
    b_none = get_metrics(post_out_none)['e_b']
    
    # Simulate update for Harsh
    post_out_harsh = observer_update(prior_out, 2)
    b_harsh = get_metrics(post_out_harsh)['e_b']
    
    print(f"Intrinsic Val:  None={u_int_none:.4f} | Harsh={u_int_harsh:.4f} (Diff={u_int_harsh - u_int_none:.4f})")
    print(f"Out-Group b:    None={b_none:.4f}   | Harsh={b_harsh:.4f}   (Diff={b_none - b_harsh:.4f})")
    
    # 3. Weighted Total
    u_rep_none = weights['w_B_out'] * b_none
    u_rep_harsh = weights['w_B_out'] * b_harsh
    
    total_none = weights['scale_int'] * u_int_none + weights['scale_rep'] * u_rep_none
    total_harsh = weights['scale_int'] * u_int_harsh + weights['scale_rep'] * u_rep_harsh
    
    print(f"Total Util:     None={total_none:.4f} | Harsh={total_harsh:.4f}")
    print(f"WINNER:         {'HARSH' if total_harsh > total_none else 'NONE'}")
    print("------------------------------------------------\n")


# -----------------------------------------------------------------------------
# 5. SIMULATION & SWEEP
# -----------------------------------------------------------------------------


def create_prior_tensor(w_params, b_params, j_params):
    pdf_w = beta.pdf(W_GRID, *w_params); pdf_w /= pdf_w.sum()
    pdf_b = beta.pdf(B_GRID, *b_params); pdf_b /= pdf_b.sum()
    pdf_j = beta.pdf(J_GRID, *j_params); pdf_j /= pdf_j.sum()
    return jnp.einsum('i,j,k->ijk', pdf_w, pdf_b, pdf_j)

def run_simulation(num_rounds, true_state, start_priors_in, start_priors_out, weights):
    curr_in = create_prior_tensor(*start_priors_in)
    curr_out = create_prior_tensor(*start_priors_out)
    
    history = []

    # Create a local random state for this specific simulation run
    rng = np.random.RandomState(42)
    
    for r in range(num_rounds):
        # 1. Strategic Decision
        probs = get_strategic_action_probs(
            curr_in, curr_out, 
            true_state['w'], true_state['b'], true_state['j'], 
            weights
        )
        
        # Sample Action from the strategic policy
        key = jax.random.PRNGKey(np.random.randint(0, 100000) + r)
        action = int(jax.random.categorical(key, jnp.log(probs)))
        
        # 2. Record Metrics (Beliefs BEFORE update)
        m_in = get_metrics(curr_in)
        m_out = get_metrics(curr_out)
        
        history.append({
            'round': r,
            'action': action,
            'prob_harsh': float(probs[2]),
            'prob_mild': float(probs[1]),
            'prob_none': float(probs[0]),

            # Save Wrongness
            'in_wrongness_mean': float(m_in['e_w']), 'in_wrongness_std': float(m_in['std_w']),
            'out_wrongness_mean': float(m_out['e_w']), 'out_wrongness_std': float(m_out['std_w']),
            
            # Save In-Group
            'in_justice_mean': float(m_in['e_j']), 'in_justice_std': float(m_in['std_j']),
            'in_bias_mean':    float(m_in['e_b']), 'in_bias_std':    float(m_in['std_b']),
            
            # Save Out-Group
            'out_justice_mean': float(m_out['e_j']), 'out_justice_std': float(m_out['std_j']),
            'out_bias_mean':    float(m_out['e_b']), 'out_bias_std':    float(m_out['std_b']),
        })
        
        # 3. Update Beliefs
        curr_in = observer_update(curr_in, action)
        curr_out = observer_update(curr_out, action)
        
    return pd.DataFrame(history)

def run_conflict_sweep(debug=False):
    print("Running Information Asymmetry Sweep (Corrected)...")
    
    # true state of punisher
    true_state = {'w': 0.5, 'b': 0.0, 'j': 1.0}
    
    # in group priors
    priors_in = ((1,1), (5,5), (1,1)) 
    
    # out group priors
    priors_out = ((1, 5), (5, 5), (1, 1)) 
    
    # sweep of weights
    sweep_values = [10.0, 0.0, -10.0] 
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    
    # Pre-build tensors for debug
    if debug:
        t_in = create_prior_tensor(*priors_in)
        t_out = create_prior_tensor(*priors_out)
    
    for i, sweep_value in enumerate(sweep_values):
        weights = {
            'scale_int': 0.5,      
            'scale_rep': 2.0,       
            'w_J_in': 10.0,          
            'w_B_in': 1.0,
            'w_J_out': 0.0,
            'w_B_out': sweep_value, 
            'beta_strat': 10.0       
        }
        
        # This will tell you explicitly why the agent is choosing what it chooses
        if debug:
            debug_utilities(t_in, t_out, 1.0, 0.0, 1.0, weights)

        
        df = run_simulation(10, true_state, priors_in, priors_out, weights)
        
        ax = axes[i]
        rounds = df['round']
        
       # ==========================================
        # ROW 1: WRONGNESS BELIEFS
        # ==========================================
        ax_wrong = axes[0, i]
        
        plot_belief_ribbon(ax_wrong, rounds, 
                          df['in_wrongness_mean'], df['in_wrongness_std'],
                          color='blue', label='In-Group E[w]')
        
        plot_belief_ribbon(ax_wrong, rounds,
                          df['out_wrongness_mean'], df['out_wrongness_std'],
                          color='red', label='Out-Group E[w]')
        
        ax_wrong.set_title(f"{sweep_variable}={sweep_value}")
        ax_wrong.set_ylim(-0.1, 1.1)
        ax_wrong.set_xlabel("Round")
        if i == 0:
            ax_wrong.set_ylabel("Expected Wrongness E[w]")
        if i == 2:
            ax_wrong.legend(loc='upper right', fontsize=8)

        # ==========================================
        # ROW 2: BIAS BELIEFS
        # ==========================================
        ax_bias = axes[1, i] 
        
        plot_belief_ribbon(ax_bias, rounds, 
                          df['in_bias_mean'], df['in_bias_std'],
                          color='blue', label='In-Group E[b]')
        
        plot_belief_ribbon(ax_bias, rounds,
                          df['out_bias_mean'], df['out_bias_std'],
                          color='red', label='Out-Group E[b]')
        
        # Plot Action Probabilities on twin axis
        ax2_bias = plot_action_probabilities(ax_bias, df)
        
        ax_bias.set_ylim(-0.6, 0.6)
        ax_bias.axhline(0, color='k', linestyle='-', linewidth=0.5)
        ax_bias.set_xlabel("Round")
        
        if i == 0:
            ax_bias.set_ylabel("Expected Bias E[b]")
        if i == 2:
            ax2_bias.set_ylabel("Action Probability")
            lines, labels = ax_bias.get_legend_handles_labels()
            lines2, labels2 = ax2_bias.get_legend_handles_labels()
            ax2_bias.legend(lines + lines2, labels + labels2, loc='upper right', fontsize=8)
        
        # ==========================================
        # ROW 3: JUSTICE BELIEFS
        # ==========================================
        ax_justice = axes[2, i]
        
        plot_belief_ribbon(ax_justice, rounds,
                          df['in_justice_mean'], df['in_justice_std'],
                          color='blue', label='In-Group E[j]')
        
        plot_belief_ribbon(ax_justice, rounds,
                          df['out_justice_mean'], df['out_justice_std'],
                          color='red', label='Out-Group E[j]')
        
        ax_justice.set_ylim(0.0, 1.0)
        ax_justice.set_xlabel("Round")
        
        if i == 0:
            ax_justice.set_ylabel("Expected Justice E[j]")
        if i == 2:
            ax_justice.legend(loc='upper right', fontsize=8)

    # Row Labels
    fig.text(0.02, 0.88, 'Wrongness\nBeliefs', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.62, 'Bias\nBeliefs', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.28, 'Justice\nBeliefs', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    # Save to output file (relative to script location)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, "..", "sweep_results.png")
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved sweep results to {os.path.abspath(output_path)}")

if __name__ == "__main__":
    run_conflict_sweep(debug=True)