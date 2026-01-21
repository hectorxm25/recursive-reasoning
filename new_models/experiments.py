import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from models import ModelContext


# -----------------------------------------------------------------------------
# CONSTANTS: Valid sweep variables organized by category
# -----------------------------------------------------------------------------

WEIGHT_VARIABLES = ['w_J_in', 'w_J_out', 'w_B_in', 'w_B_out', 'scale_int', 'scale_rep', 'beta_strat']
MODEL_CONTEXT_VARIABLES = ['GRID_SIZE', 'BETA_NAIVE']
TRUE_STATE_VARIABLES = ['true_state_w', 'true_state_b', 'true_state_j']
PRIOR_IN_VARIABLES = [
    'prior_in_w_alpha', 'prior_in_w_beta',
    'prior_in_b_alpha', 'prior_in_b_beta',
    'prior_in_j_alpha', 'prior_in_j_beta'
]
PRIOR_OUT_VARIABLES = [
    'prior_out_w_alpha', 'prior_out_w_beta',
    'prior_out_b_alpha', 'prior_out_b_beta',
    'prior_out_j_alpha', 'prior_out_j_beta'
]

ALL_SWEEPABLE_VARIABLES = (
    WEIGHT_VARIABLES + 
    MODEL_CONTEXT_VARIABLES + 
    TRUE_STATE_VARIABLES + 
    PRIOR_IN_VARIABLES + 
    PRIOR_OUT_VARIABLES
)


# -----------------------------------------------------------------------------
# SIMULATION
# -----------------------------------------------------------------------------

def run_simulation(num_rounds, true_state, start_priors_in, start_priors_out, weights, model_ctx):
    """
    Run a single simulation with the given parameters.
    
    Args:
        num_rounds: Number of rounds to simulate
        true_state: Dict with 'w', 'b', 'j' keys for the true state
        start_priors_in: Tuple of (w_params, b_params, j_params) for in-group prior
        start_priors_out: Tuple of (w_params, b_params, j_params) for out-group prior
        weights: Dict with weight parameters
        model_ctx: ModelContext instance with precomputed tensors and functions
    
    Returns:
        DataFrame with simulation history
    """
    curr_in = model_ctx.create_prior_tensor(*start_priors_in)
    curr_out = model_ctx.create_prior_tensor(*start_priors_out)
    
    history = []
    
    for r in range(num_rounds):
        # 1. Strategic Decision
        probs = model_ctx.get_strategic_action_probs(
            curr_in, curr_out, 
            true_state['w'], true_state['b'], true_state['j'], 
            weights
        )
        
        # Sample Action from the strategic policy
        key = jax.random.PRNGKey(np.random.randint(0, 100000) + r)
        action = int(jax.random.categorical(key, jnp.log(probs)))
        
        # 2. Record Metrics (Beliefs BEFORE update)
        m_in = model_ctx.get_metrics(curr_in)
        m_out = model_ctx.get_metrics(curr_out)
        
        history.append({
            'round': r,
            'action': action,
            'prob_harsh': float(probs[2]),
            'prob_mild': float(probs[1]),
            'prob_none': float(probs[0]),

            # Save Wrongness Metrics
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
        curr_in = model_ctx.observer_update(curr_in, action)
        curr_out = model_ctx.observer_update(curr_out, action)
        
    return pd.DataFrame(history)


# -----------------------------------------------------------------------------
# HELPER FUNCTIONS FOR PARAMETER CONSTRUCTION
# -----------------------------------------------------------------------------

def build_weights(config, sweep_variable=None, sweep_value=None):
    """
    Build the weights dictionary from config, optionally overriding one variable.
    
    Args:
        config: The full configuration dict
        sweep_variable: The variable being swept (or None)
        sweep_value: The value to use for the sweep variable (or None)
    
    Returns:
        Dict with weight parameters
    """
    weights = {
        'scale_int': config['scale_int'],      
        'scale_rep': config['scale_rep'],       
        'w_J_in': config['w_J_in'],          
        'w_B_in': config['w_B_in'],
        'w_J_out': config['w_J_out'],
        'w_B_out': config['w_B_out'], 
        'beta_strat': config['beta_strat']       
    }
    
    # Override if sweeping a weight variable
    if sweep_variable in WEIGHT_VARIABLES and sweep_value is not None:
        weights[sweep_variable] = sweep_value
    
    return weights


def build_true_state(config, sweep_variable=None, sweep_value=None):
    """
    Build the true state dictionary from config, optionally overriding one variable.
    
    Args:
        config: The full configuration dict
        sweep_variable: The variable being swept (or None)
        sweep_value: The value to use for the sweep variable (or None)
    
    Returns:
        Dict with 'w', 'b', 'j' keys
    """
    true_state = {
        'w': config['true_state_w'], 
        'b': config['true_state_b'], 
        'j': config['true_state_j']
    }
    
    # Override if sweeping a true state variable
    if sweep_variable == 'true_state_w' and sweep_value is not None:
        true_state['w'] = sweep_value
    elif sweep_variable == 'true_state_b' and sweep_value is not None:
        true_state['b'] = sweep_value
    elif sweep_variable == 'true_state_j' and sweep_value is not None:
        true_state['j'] = sweep_value
    
    return true_state


def build_priors_in(config, sweep_variable=None, sweep_value=None):
    """
    Build the in-group priors tuple from config, optionally overriding one variable.
    
    Args:
        config: The full configuration dict
        sweep_variable: The variable being swept (or None)
        sweep_value: The value to use for the sweep variable (or None)
    
    Returns:
        Tuple of (w_params, b_params, j_params)
    """
    # Start with config values
    w_alpha = config['prior_in_w_alpha']
    w_beta = config['prior_in_w_beta']
    b_alpha = config['prior_in_b_alpha']
    b_beta = config['prior_in_b_beta']
    j_alpha = config['prior_in_j_alpha']
    j_beta = config['prior_in_j_beta']
    
    # Override if sweeping an in-group prior variable
    if sweep_value is not None:
        if sweep_variable == 'prior_in_w_alpha':
            w_alpha = sweep_value
        elif sweep_variable == 'prior_in_w_beta':
            w_beta = sweep_value
        elif sweep_variable == 'prior_in_b_alpha':
            b_alpha = sweep_value
        elif sweep_variable == 'prior_in_b_beta':
            b_beta = sweep_value
        elif sweep_variable == 'prior_in_j_alpha':
            j_alpha = sweep_value
        elif sweep_variable == 'prior_in_j_beta':
            j_beta = sweep_value
    
    return ((w_alpha, w_beta), (b_alpha, b_beta), (j_alpha, j_beta))


def build_priors_out(config, sweep_variable=None, sweep_value=None):
    """
    Build the out-group priors tuple from config, optionally overriding one variable.
    
    Args:
        config: The full configuration dict
        sweep_variable: The variable being swept (or None)
        sweep_value: The value to use for the sweep variable (or None)
    
    Returns:
        Tuple of (w_params, b_params, j_params)
    """
    # Start with config values
    w_alpha = config['prior_out_w_alpha']
    w_beta = config['prior_out_w_beta']
    b_alpha = config['prior_out_b_alpha']
    b_beta = config['prior_out_b_beta']
    j_alpha = config['prior_out_j_alpha']
    j_beta = config['prior_out_j_beta']
    
    # Override if sweeping an out-group prior variable
    if sweep_value is not None:
        if sweep_variable == 'prior_out_w_alpha':
            w_alpha = sweep_value
        elif sweep_variable == 'prior_out_w_beta':
            w_beta = sweep_value
        elif sweep_variable == 'prior_out_b_alpha':
            b_alpha = sweep_value
        elif sweep_variable == 'prior_out_b_beta':
            b_beta = sweep_value
        elif sweep_variable == 'prior_out_j_alpha':
            j_alpha = sweep_value
        elif sweep_variable == 'prior_out_j_beta':
            j_beta = sweep_value
    
    return ((w_alpha, w_beta), (b_alpha, b_beta), (j_alpha, j_beta))


def build_model_context(config, sweep_variable=None, sweep_value=None):
    """
    Build the ModelContext from config, optionally overriding GRID_SIZE or BETA_NAIVE.
    
    Args:
        config: The full configuration dict
        sweep_variable: The variable being swept (or None)
        sweep_value: The value to use for the sweep variable (or None)
    
    Returns:
        ModelContext instance
    """
    grid_size = config['GRID_SIZE']
    beta_naive = config['BETA_NAIVE']
    
    # Override if sweeping a model context variable
    if sweep_value is not None:
        if sweep_variable == 'GRID_SIZE':
            grid_size = int(sweep_value)
        elif sweep_variable == 'BETA_NAIVE':
            beta_naive = sweep_value
    
    return ModelContext(grid_size, beta_naive)


# -----------------------------------------------------------------------------
# PLOTTING HELPERS
# -----------------------------------------------------------------------------

def plot_belief_ribbon(ax, rounds, mean, std, color, label):
    """Plot a belief trajectory with ribbon for standard deviation."""
    ax.plot(rounds, mean, 'o-', color=color, label=label, markersize=4)
    ax.fill_between(rounds, mean - std, mean + std, color=color, alpha=0.2)


def plot_action_probabilities(ax, df):
    """Plot action probabilities on a twin axis."""
    ax2 = ax.twinx()
    rounds = df['round']
    ax2.plot(rounds, df['prob_harsh'], '--', color='black', alpha=0.6, label='P(Harsh)', linewidth=1.5)
    ax2.plot(rounds, df['prob_mild'], '-.', color='gray', alpha=0.6, label='P(Mild)', linewidth=1.5)
    ax2.plot(rounds, df['prob_none'], ':', color='dimgray', alpha=0.6, label='P(None)', linewidth=1.5)
    ax2.set_ylim(-0.1, 1.1)
    return ax2


# -----------------------------------------------------------------------------
# EXPERIMENT SWEEP
# -----------------------------------------------------------------------------

def run_conflict_sweep(config, debug=False):
    """
    Run the information asymmetry sweep experiment.
    
    The sweep variable is determined by config['sweep_variable'], which can be
    any of the following:
        - Weight variables: w_J_in, w_J_out, w_B_in, w_B_out, scale_int, scale_rep, beta_strat
        - Model context variables: GRID_SIZE, BETA_NAIVE
        - True state variables: true_state_w, true_state_b, true_state_j
        - Prior variables: prior_in_w_alpha, prior_in_w_beta, prior_in_b_alpha, 
                          prior_in_b_beta, prior_in_j_alpha, prior_in_j_beta,
                          prior_out_w_alpha, prior_out_w_beta, prior_out_b_alpha,
                          prior_out_b_beta, prior_out_j_alpha, prior_out_j_beta
    
    Args:
        config: Dict with all configuration parameters
        debug: Whether to print debug information
    """
    sweep_variable = config['sweep_variable']
    sweep_values = config['sweep_values']
    
    print(f"Running Sweep over '{sweep_variable}'...")
    print(f"  Sweep values: {sweep_values}")
    
    # Determine if we need to recreate ModelContext for each sweep value
    recreate_model_ctx = sweep_variable in MODEL_CONTEXT_VARIABLES
    
    # Create base model context (will be reused if not sweeping GRID_SIZE or BETA_NAIVE)
    if not recreate_model_ctx:
        model_ctx = build_model_context(config)
    
    # row 1: wrongness
    # row 2: bias
    # row 3: justice
    fig, axes = plt.subplots(3, 3, figsize=(15, 8))
    
    # Pre-build tensors for debug (only if not sweeping model context or priors)
    if debug and not recreate_model_ctx:
        priors_in_debug = build_priors_in(config)
        priors_out_debug = build_priors_out(config)
        t_in = model_ctx.create_prior_tensor(*priors_in_debug)
        t_out = model_ctx.create_prior_tensor(*priors_out_debug)
    
    for i, sweep_value in enumerate(sweep_values):
        # Build all parameters, with the sweep variable overridden
        if recreate_model_ctx:
            model_ctx = build_model_context(config, sweep_variable, sweep_value)
        
        weights = build_weights(config, sweep_variable, sweep_value)
        true_state = build_true_state(config, sweep_variable, sweep_value)
        priors_in = build_priors_in(config, sweep_variable, sweep_value)
        priors_out = build_priors_out(config, sweep_variable, sweep_value)
        
        # Debug output
        if debug:
            if recreate_model_ctx or sweep_variable in PRIOR_IN_VARIABLES + PRIOR_OUT_VARIABLES:
                # Need to rebuild debug tensors
                t_in = model_ctx.create_prior_tensor(*priors_in)
                t_out = model_ctx.create_prior_tensor(*priors_out)
            model_ctx.debug_utilities(t_in, t_out, true_state['w'], true_state['b'], true_state['j'], weights)
        
        # Run simulation
        df = run_simulation(config['rounds'], true_state, priors_in, priors_out, weights, model_ctx)
        
        rounds = df['round']
        
        # ==========================================
        # ROW 1: WRONGNESS BELIEFS (NEW)
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
            ax_wrong.legend(loc='lower right', fontsize=8)

        # ==========================================
        # ROW 2: BIAS BELIEFS (Shifted down)
        # ==========================================
        ax_bias = axes[1, i] # Was axes[0, i]
        
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
            ax2_bias.legend(lines + lines2, labels + labels2, loc='lower right', fontsize=8)
        
        # ==========================================
        # ROW 3: JUSTICE BELIEFS (Shifted down)
        # ==========================================
        ax_justice = axes[2, i] # Was axes[1, i]
        
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
            ax_justice.legend(loc='lower right', fontsize=8)

    # Update Row Labels
    fig.text(0.02, 0.88, 'Wrongness\nBeliefs', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.62, 'Bias\nBeliefs', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    fig.text(0.02, 0.28, 'Justice\nBeliefs', ha='center', va='center', fontsize=12, fontweight='bold', rotation=90)
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08)
    
    # Save to output file
    output_path = config['plot_figure_save_path']
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved sweep results to {os.path.abspath(output_path)}")
