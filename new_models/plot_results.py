import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import time
import argparse
from datetime import datetime

from models import ModelContext
from experiments import build_model_context

# -----------------------------------------------------------------------------
# LOGGING SETUP
# -----------------------------------------------------------------------------

def setup_logging(log_dir="logs", log_level=logging.INFO):
    """Configure logging to output to both console and a file in log_dir."""
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"simulation_{timestamp}.log")
    
    # Create logger
    logger = logging.getLogger("plot_results")
    logger.setLevel(log_level)
    
    # Clear existing handlers to avoid duplicates on repeated calls
    logger.handlers.clear()
    
    # File handler
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(log_level)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_filename}")
    return logger

# Global logger instance
logger = setup_logging()

# -----------------------------------------------------------------------------
# HELPER: Compute Probabilities for One Step
# -----------------------------------------------------------------------------

def get_action_probs(model_ctx, priors_in, priors_out, true_w, true_b, true_j, weights):
    """Run the strategic planner for one step and return the action probabilities."""
    curr_in = model_ctx.create_prior_tensor(*priors_in)
    curr_out = model_ctx.create_prior_tensor(*priors_out)
    probs = model_ctx.get_strategic_action_probs(
        curr_in, curr_out, true_w, true_b, true_j, weights
    )
    return np.array(probs)

def get_utility_components(model_ctx, priors_in, priors_out, true_w, true_b, true_j, weights):
    """Run the utility component calculator and return utility breakdown for each action."""
    curr_in = model_ctx.create_prior_tensor(*priors_in)
    curr_out = model_ctx.create_prior_tensor(*priors_out)
    utilities = model_ctx.get_utility_components(
        curr_in, curr_out, true_w, true_b, true_j, weights
    )
    return utilities

# -----------------------------------------------------------------------------
# AGENT CONFIGURATIONS
# -----------------------------------------------------------------------------

def get_agent_weights(agent_type):
    """Returns the weights dictionary for the specific agent type."""
    weights = {
        'scale_int': 1.0, 'scale_rep': 0.0, 'scale_comm': 0.0,
        'w_J_in': 0.0, 'w_J_out': 0.0,
        'w_B_in_dir': 0.0, 'w_B_out_dir': 0.0,
        'w_B_in_neu': 0.0, 'w_B_out_neu': 0.0,
        'beta_strat': 10.0 
    }
    
    if agent_type == 'naive':
        pass 
    elif agent_type == 'reputation':
        weights['scale_rep'] = 5.0
        val = 2.0
        weights['w_J_in'] = val
        weights['w_J_out'] = val
        weights['w_B_in_neu'] = val
        weights['w_B_out_neu'] = val
    elif agent_type == 'communicative':
        weights['scale_comm'] = 20.0 
        
    return weights

# -----------------------------------------------------------------------------
# PARAMETER GENERATORS
# -----------------------------------------------------------------------------

def get_beta_params(mean, concentration=10.0):
    alpha = mean * concentration + 1.0
    beta = (1.0 - mean) * concentration + 1.0
    return (alpha, beta)

def get_interpolated_params(p, start_mean, end_mean, concentration=20.0):
    """Linearly interpolate between two means while keeping concentration fixed.
    
    Args:
        p: Interpolation parameter in [0, 1]. p=0 gives start_mean, p=1 gives end_mean.
        start_mean: Starting mean value.
        end_mean: Ending mean value.
        concentration: Fixed concentration parameter (default 20.0).
    
    Returns:
        Tuple of (alpha, beta) for the interpolated beta distribution.
    """
    curr_mean = (1.0 - p) * start_mean + p * end_mean
    return get_beta_params(curr_mean, concentration)

# -----------------------------------------------------------------------------
# AGENT TYPE CONFIGURATIONS
# -----------------------------------------------------------------------------

# Four agent types defined by (J, B) combinations
AGENT_TYPES = [
    {'j': 0.9, 'b': -0.4, 'label': 'High J, Anti-B'},
    {'j': 0.9, 'b': 0.4,  'label': 'High J, Pro-B'},
    {'j': 0.1, 'b': -0.4, 'label': 'Low J, Anti-B'},
    {'j': 0.1, 'b': 0.4,  'label': 'Low J, Pro-B'},
]

# -----------------------------------------------------------------------------
# SUBPLOT SIMULATION HELPERS
# -----------------------------------------------------------------------------

def run_subplot_polarized_motives(model_ctx, ax, agent_type, agents, colors, w_values, punishment_mode='mild'):
    """Subplot: Polarized Motives, Uncertain Wrongness - sweep authority's W belief."""
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_in_j = get_beta_params(0.9, 20)
    p_in_b = get_beta_params(0.5, 20)
    p_in_w = (1.0, 1.0)
    p_out_j = get_beta_params(0.1, 20)
    bias_mean_out = 0.1 if true_b < 0 else 0.9
    p_out_b = get_beta_params(bias_mean_out, 20)
    p_out_w = (1.0, 1.0)
    priors_in = (p_in_w, p_in_b, p_in_j)
    priors_out = (p_out_w, p_out_b, p_out_j)
    
    for ag in agents:
        weights = get_agent_weights(ag)
        y_primary = []
        y_secondary = []
        for w in w_values:
            probs = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights)
            if punishment_mode == 'total-punishment':
                y_primary.append(probs[1] + probs[2])  # P(Mild) + P(Harsh)
            else:
                y_primary.append(probs[2])  # P(Harsh)
                if punishment_mode == 'mild':
                    y_secondary.append(probs[1])  # P(Mild)
                else:  # punishment_mode == 'none'
                    y_secondary.append(probs[0])  # P(None)
        
        if punishment_mode == 'total-punishment':
            ax.plot(w_values, y_primary, color=colors[ag], linestyle='-', label=f"{ag.capitalize()}")
        else:
            secondary_label = "Mild" if punishment_mode == 'mild' else "None"
            ax.plot(w_values, y_primary, color=colors[ag], linestyle='-', label=f"{ag.capitalize()} (Harsh)")
            ax.plot(w_values, y_secondary, color=colors[ag], linestyle='--', label=f"{ag.capitalize()} ({secondary_label})")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("P(Punishment)" if punishment_mode == 'total-punishment' else "P(Action)")

def run_subplot_polarized_wrongness(model_ctx, ax, agent_type, agents, colors, w_values, punishment_mode='mild'):
    """Subplot: Polarized Wrongness, Uncertain Motives - sweep authority's W belief."""
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_j_unc = (1.0, 1.0)
    p_b_unc = (1.0, 1.0)
    p_in_w = get_beta_params(0.9, 20)
    p_out_w = get_beta_params(0.1, 20)
    priors_in = (p_in_w, p_b_unc, p_j_unc)
    priors_out = (p_out_w, p_b_unc, p_j_unc)
    
    for ag in agents:
        weights = get_agent_weights(ag)
        y_primary = []
        y_secondary = []
        for w in w_values:
            probs = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights)
            if punishment_mode == 'total-punishment':
                y_primary.append(probs[1] + probs[2])
            else:
                y_primary.append(probs[2])
                if punishment_mode == 'mild':
                    y_secondary.append(probs[1])
                else:
                    y_secondary.append(probs[0])
        
        if punishment_mode == 'total-punishment':
            ax.plot(w_values, y_primary, color=colors[ag], linestyle='-', label=f"{ag.capitalize()}")
        else:
            secondary_label = "Mild" if punishment_mode == 'mild' else "None"
            ax.plot(w_values, y_primary, color=colors[ag], linestyle='-', label=f"{ag.capitalize()} (Harsh)")
            ax.plot(w_values, y_secondary, color=colors[ag], linestyle='--', label=f"{ag.capitalize()} ({secondary_label})")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("P(Punishment)" if punishment_mode == 'total-punishment' else "P(Action)")

def run_subplot_wrongness_polarization(model_ctx, ax, agent_type, agents, colors, pol_values, auth_w, punishment_mode='mild'):
    """Subplot: Wrongness Polarization - sweep polarization level at fixed authority W."""
    true_j, true_b = agent_type['j'], agent_type['b']
    p_motives = ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
    
    for ag in agents:
        weights = get_agent_weights(ag)
        y_primary = []
        y_secondary = []
        for p in pol_values:
            in_mean = 0.5 + p * 0.45
            out_mean = 0.5 - p * 0.45
            p_in = (get_beta_params(in_mean, 20), p_motives[1], p_motives[2])
            p_out = (get_beta_params(out_mean, 20), p_motives[1], p_motives[2])
            probs = get_action_probs(model_ctx, p_in, p_out, auth_w, true_b, true_j, weights)
            if punishment_mode == 'total-punishment':
                y_primary.append(probs[1] + probs[2])
            else:
                y_primary.append(probs[2])
                if punishment_mode == 'mild':
                    y_secondary.append(probs[1])
                else:
                    y_secondary.append(probs[0])
        
        if punishment_mode == 'total-punishment':
            ax.plot(pol_values, y_primary, color=colors[ag], linestyle='-', label=f"{ag.capitalize()}")
        else:
            secondary_label = "Mild" if punishment_mode == 'mild' else "None"
            ax.plot(pol_values, y_primary, color=colors[ag], linestyle='-', label=f"{ag.capitalize()} (Harsh)")
            ax.plot(pol_values, y_secondary, color=colors[ag], linestyle='--', label=f"{ag.capitalize()} ({secondary_label})")
    
    ax.set_xlabel("Polarization Level")
    ax.set_ylabel("P(Punishment)" if punishment_mode == 'total-punishment' else "P(Action)")

def run_subplot_trust_polarization(model_ctx, ax, agent_type, agents, colors, pol_values, auth_w, punishment_mode='mild'):
    """Subplot: Trust Polarization - sweep out-group distrust at fixed authority W."""
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_w_unc = (1.0, 1.0)
    # In-group priors (fixed): high justice, neutral bias
    in_j_mean = 0.9
    in_b_mean = 0.5
    in_j_params = get_beta_params(in_j_mean, 20)
    in_b_params = get_beta_params(in_b_mean, 20)
    p_in_trust = (p_w_unc, in_b_params, in_j_params)
    
    # Out-group endpoint means: low justice, biased
    out_j_mean = 0.1
    out_b_mean = 0.1 if true_b < 0 else 0.9
    
    for ag in agents:
        weights = get_agent_weights(ag)
        y_primary = []
        y_secondary = []
        for p in pol_values:
            # Interpolate means only, keeping concentration fixed at 20
            curr_j = get_interpolated_params(p, in_j_mean, out_j_mean, concentration=20.0)
            curr_b = get_interpolated_params(p, in_b_mean, out_b_mean, concentration=20.0)
            priors_out = (p_w_unc, curr_b, curr_j)
            probs = get_action_probs(model_ctx, p_in_trust, priors_out, auth_w, true_b, true_j, weights)
            if punishment_mode == 'total-punishment':
                y_primary.append(probs[1] + probs[2])
            else:
                y_primary.append(probs[2])
                if punishment_mode == 'mild':
                    y_secondary.append(probs[1])
                else:
                    y_secondary.append(probs[0])
        
        if punishment_mode == 'total-punishment':
            ax.plot(pol_values, y_primary, color=colors[ag], linestyle='-', label=f"{ag.capitalize()}")
        else:
            secondary_label = "Mild" if punishment_mode == 'mild' else "None"
            ax.plot(pol_values, y_primary, color=colors[ag], linestyle='-', label=f"{ag.capitalize()} (Harsh)")
            ax.plot(pol_values, y_secondary, color=colors[ag], linestyle='--', label=f"{ag.capitalize()} ({secondary_label})")
    
    ax.set_xlabel("Distrust Level")
    ax.set_ylabel("P(Punishment)" if punishment_mode == 'total-punishment' else "P(Action)")

# -----------------------------------------------------------------------------
# UTILITY-MODE SUBPLOT HELPERS
# -----------------------------------------------------------------------------

# Colors for utility components
UTILITY_COLORS = {
    'u_int': 'green',
    'u_rep': 'blue', 
    'u_comm': 'red'
}

UTILITY_LABELS = {
    'u_int': 'Intrinsic',
    'u_rep': 'Reputational',
    'u_comm': 'Communicative'
}

def run_subplot_polarized_motives_utility(model_ctx, ax, agent_type, w_values):
    """Utility-mode: Polarized Motives, Uncertain Wrongness - sweep authority's W belief."""
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_in_j = get_beta_params(0.9, 20)
    p_in_b = get_beta_params(0.5, 20)
    p_in_w = (1.0, 1.0)
    p_out_j = get_beta_params(0.1, 20)
    bias_mean_out = 0.1 if true_b < 0 else 0.9
    p_out_b = get_beta_params(bias_mean_out, 20)
    p_out_w = (1.0, 1.0)
    priors_in = (p_in_w, p_in_b, p_in_j)
    priors_out = (p_out_w, p_out_b, p_out_j)
    
    # Use reputation weights to get meaningful u_rep values
    weights_rep = get_agent_weights('reputation')
    # Use communicative weights to get meaningful u_comm values
    weights_comm = get_agent_weights('communicative')
    
    # Data storage: action -> component -> list of values
    y_harsh = {'u_int': [], 'u_rep': [], 'u_comm': []}
    y_none = {'u_int': [], 'u_rep': [], 'u_comm': []}
    
    for w in w_values:
        # Get utilities for reputational agent (has u_int and u_rep)
        utils_rep = get_utility_components(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_rep)
        # Get utilities for communicative agent (has u_int and u_comm)
        utils_comm = get_utility_components(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_comm)
        
        # Harsh (action 2)
        y_harsh['u_int'].append(utils_rep[2]['u_int'])  # Intrinsic is same for both
        y_harsh['u_rep'].append(utils_rep[2]['u_rep'])
        y_harsh['u_comm'].append(utils_comm[2]['u_comm'])
        
        # None (action 0)
        y_none['u_int'].append(utils_rep[0]['u_int'])
        y_none['u_rep'].append(utils_rep[0]['u_rep'])
        y_none['u_comm'].append(utils_comm[0]['u_comm'])
    
    # Plot: solid for Harsh, dashed for None
    for comp in ['u_int', 'u_rep', 'u_comm']:
        ax.plot(w_values, y_harsh[comp], color=UTILITY_COLORS[comp], linestyle='-', 
                label=f"{UTILITY_LABELS[comp]} (Harsh)")
        ax.plot(w_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle='--',
                label=f"{UTILITY_LABELS[comp]} (None)")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("Utility")

def run_subplot_polarized_wrongness_utility(model_ctx, ax, agent_type, w_values):
    """Utility-mode: Polarized Wrongness, Uncertain Motives - sweep authority's W belief."""
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_j_unc = (1.0, 1.0)
    p_b_unc = (1.0, 1.0)
    p_in_w = get_beta_params(0.9, 20)
    p_out_w = get_beta_params(0.1, 20)
    priors_in = (p_in_w, p_b_unc, p_j_unc)
    priors_out = (p_out_w, p_b_unc, p_j_unc)
    
    weights_rep = get_agent_weights('reputation')
    weights_comm = get_agent_weights('communicative')
    
    y_harsh = {'u_int': [], 'u_rep': [], 'u_comm': []}
    y_mild = {'u_int': [], 'u_rep': [], 'u_comm': []}
    
    for w in w_values:
        utils_rep = get_utility_components(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_rep)
        utils_comm = get_utility_components(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_comm)
        
        y_harsh['u_int'].append(utils_rep[2]['u_int'])
        y_harsh['u_rep'].append(utils_rep[2]['u_rep'])
        y_harsh['u_comm'].append(utils_comm[2]['u_comm'])
        
        y_mild['u_int'].append(utils_rep[1]['u_int'])
        y_mild['u_rep'].append(utils_rep[1]['u_rep'])
        y_mild['u_comm'].append(utils_comm[1]['u_comm'])
    
    for comp in ['u_int', 'u_rep', 'u_comm']:
        ax.plot(w_values, y_harsh[comp], color=UTILITY_COLORS[comp], linestyle='-',
                label=f"{UTILITY_LABELS[comp]} (Harsh)")
        ax.plot(w_values, y_mild[comp], color=UTILITY_COLORS[comp], linestyle='--',
                label=f"{UTILITY_LABELS[comp]} (Mild)")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("Utility")

def run_subplot_wrongness_polarization_utility(model_ctx, ax, agent_type, pol_values, auth_w):
    """Utility-mode: Wrongness Polarization - sweep polarization level at fixed authority W."""
    true_j, true_b = agent_type['j'], agent_type['b']
    p_motives = ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
    
    weights_rep = get_agent_weights('reputation')
    weights_comm = get_agent_weights('communicative')
    
    y_harsh = {'u_int': [], 'u_rep': [], 'u_comm': []}
    y_none = {'u_int': [], 'u_rep': [], 'u_comm': []}
    
    for p in pol_values:
        in_mean = 0.5 + p * 0.45
        out_mean = 0.5 - p * 0.45
        p_in = (get_beta_params(in_mean, 20), p_motives[1], p_motives[2])
        p_out = (get_beta_params(out_mean, 20), p_motives[1], p_motives[2])
        
        utils_rep = get_utility_components(model_ctx, p_in, p_out, auth_w, true_b, true_j, weights_rep)
        utils_comm = get_utility_components(model_ctx, p_in, p_out, auth_w, true_b, true_j, weights_comm)
        
        y_harsh['u_int'].append(utils_rep[2]['u_int'])
        y_harsh['u_rep'].append(utils_rep[2]['u_rep'])
        y_harsh['u_comm'].append(utils_comm[2]['u_comm'])
        
        y_none['u_int'].append(utils_rep[0]['u_int'])
        y_none['u_rep'].append(utils_rep[0]['u_rep'])
        y_none['u_comm'].append(utils_comm[0]['u_comm'])
    
    for comp in ['u_int', 'u_rep', 'u_comm']:
        ax.plot(pol_values, y_harsh[comp], color=UTILITY_COLORS[comp], linestyle='-',
                label=f"{UTILITY_LABELS[comp]} (Harsh)")
        ax.plot(pol_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle='--',
                label=f"{UTILITY_LABELS[comp]} (None)")
    
    ax.set_xlabel("Polarization Level")
    ax.set_ylabel("Utility")

def run_subplot_trust_polarization_utility(model_ctx, ax, agent_type, pol_values, auth_w):
    """Utility-mode: Trust Polarization - sweep out-group distrust at fixed authority W."""
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_w_unc = (1.0, 1.0)
    in_j_mean = 0.9
    in_b_mean = 0.5
    in_j_params = get_beta_params(in_j_mean, 20)
    in_b_params = get_beta_params(in_b_mean, 20)
    p_in_trust = (p_w_unc, in_b_params, in_j_params)
    
    out_j_mean = 0.1
    out_b_mean = 0.1 if true_b < 0 else 0.9
    
    weights_rep = get_agent_weights('reputation')
    weights_comm = get_agent_weights('communicative')
    
    y_harsh = {'u_int': [], 'u_rep': [], 'u_comm': []}
    y_none = {'u_int': [], 'u_rep': [], 'u_comm': []}
    
    for p in pol_values:
        curr_j = get_interpolated_params(p, in_j_mean, out_j_mean, concentration=20.0)
        curr_b = get_interpolated_params(p, in_b_mean, out_b_mean, concentration=20.0)
        priors_out = (p_w_unc, curr_b, curr_j)
        
        utils_rep = get_utility_components(model_ctx, p_in_trust, priors_out, auth_w, true_b, true_j, weights_rep)
        utils_comm = get_utility_components(model_ctx, p_in_trust, priors_out, auth_w, true_b, true_j, weights_comm)
        
        y_harsh['u_int'].append(utils_rep[2]['u_int'])
        y_harsh['u_rep'].append(utils_rep[2]['u_rep'])
        y_harsh['u_comm'].append(utils_comm[2]['u_comm'])
        
        y_none['u_int'].append(utils_rep[0]['u_int'])
        y_none['u_rep'].append(utils_rep[0]['u_rep'])
        y_none['u_comm'].append(utils_comm[0]['u_comm'])
    
    for comp in ['u_int', 'u_rep', 'u_comm']:
        ax.plot(pol_values, y_harsh[comp], color=UTILITY_COLORS[comp], linestyle='-',
                label=f"{UTILITY_LABELS[comp]} (Harsh)")
        ax.plot(pol_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle='--',
                label=f"{UTILITY_LABELS[comp]} (None)")
    
    ax.set_xlabel("Distrust Level")
    ax.set_ylabel("Utility")

# -----------------------------------------------------------------------------
# MAIN FIGURE FUNCTIONS
# -----------------------------------------------------------------------------

def run_figure_1(model_ctx, filename="fig1_polarized_beliefs.png", punishment_mode='mild'):
    """
    Figure 1: Polarized Beliefs
    Row 1: Polarized Motives (Uncertain W)
    Row 2: Polarized Wrongness (Uncertain Motives)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 1: Polarized Beliefs")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    title_suffix = "Total Punishment" if punishment_mode == 'total-punishment' else "Punishment"
    fig.suptitle(f"Polarized Beliefs: {title_suffix} Probability vs Authority's Belief (W)", fontsize=16, fontweight='bold')
    
    agents = ['naive', 'reputation', 'communicative']
    colors = {'naive': 'green', 'reputation': 'blue', 'communicative': 'red'}
    w_values = np.linspace(0.0, 1.0, 20)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Polarized Motives
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nPolarized Motives (Uncertain W)", fontsize=10)
        run_subplot_polarized_motives(model_ctx, ax1, agent_type, agents, colors, w_values, punishment_mode)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=7)
        
        # Row 2: Polarized Wrongness
        ax2 = axes[1, col]
        ax2.set_title(f"Polarized Wrongness (Uncertain Motives)", fontsize=10)
        run_subplot_polarized_wrongness(model_ctx, ax2, agent_type, agents, colors, w_values, punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 1 saved to {filename} in {time.time() - figure_start:.2f}s")

def run_figure_2(model_ctx, filename="fig2_wrongness_polarization.png", punishment_mode='mild'):
    """
    Figure 2: Wrongness Polarization
    Row 1: Authority believes WRONG (W=1.0)
    Row 2: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 2: Wrongness Polarization")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    title_suffix = "Total Punishment" if punishment_mode == 'total-punishment' else "Punishment"
    fig.suptitle(f"Wrongness Polarization: {title_suffix} Probability vs Audience Polarization Level", fontsize=16, fontweight='bold')
    
    agents = ['naive', 'reputation', 'communicative']
    colors = {'naive': 'green', 'reputation': 'blue', 'communicative': 'red'}
    pol_values = np.linspace(0.0, 1.0, 20)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Auth believes WRONG (W=1.0)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAuthority: W=1.0 (Wrong)", fontsize=10)
        run_subplot_wrongness_polarization(model_ctx, ax1, agent_type, agents, colors, pol_values, auth_w=1.0, punishment_mode=punishment_mode)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=7)
        
        # Row 2: Auth believes NOT WRONG (W=0.0)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_wrongness_polarization(model_ctx, ax2, agent_type, agents, colors, pol_values, auth_w=0.0, punishment_mode=punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 2 saved to {filename} in {time.time() - figure_start:.2f}s")

def run_figure_3(model_ctx, filename="fig3_trust_polarization.png", punishment_mode='mild'):
    """
    Figure 3: Trust Polarization
    Row 1: Authority believes WRONG (W=1.0)
    Row 2: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 3: Trust Polarization")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    title_suffix = "Total Punishment" if punishment_mode == 'total-punishment' else "Punishment"
    fig.suptitle(f"Trust Polarization: {title_suffix} Probability vs Out-Group Distrust Level", fontsize=16, fontweight='bold')
    
    agents = ['naive', 'reputation', 'communicative']
    colors = {'naive': 'green', 'reputation': 'blue', 'communicative': 'red'}
    pol_values = np.linspace(0.0, 1.0, 20)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Auth believes WRONG (W=1.0)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAuthority: W=1.0 (Wrong)", fontsize=10)
        run_subplot_trust_polarization(model_ctx, ax1, agent_type, agents, colors, pol_values, auth_w=1.0, punishment_mode=punishment_mode)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=7)
        
        # Row 2: Auth believes NOT WRONG (W=0.0)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_trust_polarization(model_ctx, ax2, agent_type, agents, colors, pol_values, auth_w=0.0, punishment_mode=punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 3 saved to {filename} in {time.time() - figure_start:.2f}s")

# -----------------------------------------------------------------------------
# UTILITY-MODE FIGURE FUNCTIONS
# -----------------------------------------------------------------------------

def run_figure_1_utility(model_ctx, filename="fig1_utility_polarized_beliefs.png"):
    """
    Figure 1 (Utility Mode): Polarized Beliefs
    Row 1: Polarized Motives (Uncertain W)
    Row 2: Polarized Wrongness (Uncertain Motives)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 1 (Utility): Polarized Beliefs")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Polarized Beliefs: Utility Components vs Authority's Belief (W)", fontsize=16, fontweight='bold')
    
    w_values = np.linspace(0.0, 1.0, 20)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Polarized Motives
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nPolarized Motives (Uncertain W)", fontsize=10)
        run_subplot_polarized_motives_utility(model_ctx, ax1, agent_type, w_values)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=6)
        
        # Row 2: Polarized Wrongness
        ax2 = axes[1, col]
        ax2.set_title(f"Polarized Wrongness (Uncertain Motives)", fontsize=10)
        run_subplot_polarized_wrongness_utility(model_ctx, ax2, agent_type, w_values)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 1 (Utility) saved to {filename} in {time.time() - figure_start:.2f}s")

def run_figure_2_utility(model_ctx, filename="fig2_utility_wrongness_polarization.png"):
    """
    Figure 2 (Utility Mode): Wrongness Polarization
    Row 1: Authority believes WRONG (W=1.0)
    Row 2: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 2 (Utility): Wrongness Polarization")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Wrongness Polarization: Utility Components vs Audience Polarization Level", fontsize=16, fontweight='bold')
    
    pol_values = np.linspace(0.0, 1.0, 20)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Auth believes WRONG (W=1.0)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAuthority: W=1.0 (Wrong)", fontsize=10)
        run_subplot_wrongness_polarization_utility(model_ctx, ax1, agent_type, pol_values, auth_w=1.0)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=6)
        
        # Row 2: Auth believes NOT WRONG (W=0.0)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_wrongness_polarization_utility(model_ctx, ax2, agent_type, pol_values, auth_w=0.0)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 2 (Utility) saved to {filename} in {time.time() - figure_start:.2f}s")

def run_figure_3_utility(model_ctx, filename="fig3_utility_trust_polarization.png"):
    """
    Figure 3 (Utility Mode): Trust Polarization
    Row 1: Authority believes WRONG (W=1.0)
    Row 2: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 3 (Utility): Trust Polarization")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Trust Polarization: Utility Components vs Out-Group Distrust Level", fontsize=16, fontweight='bold')
    
    pol_values = np.linspace(0.0, 1.0, 20)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Auth believes WRONG (W=1.0)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAuthority: W=1.0 (Wrong)", fontsize=10)
        run_subplot_trust_polarization_utility(model_ctx, ax1, agent_type, pol_values, auth_w=1.0)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=6)
        
        # Row 2: Auth believes NOT WRONG (W=0.0)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_trust_polarization_utility(model_ctx, ax2, agent_type, pol_values, auth_w=0.0)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 3 (Utility) saved to {filename} in {time.time() - figure_start:.2f}s")

# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def main(log_dir="logs", utility_mode=False, punishment_mode='mild', save_dir=None):
    """Run all figure simulations with logging.
    
    Args:
        log_dir: Directory to store log files (default: "logs")
        utility_mode: If True, generate utility component plots instead of probability plots
        punishment_mode: For probability plots, controls what the dashed line shows:
            'mild' - P(Mild), 'none' - P(None), 'total-punishment' - only P(Mild)+P(Harsh)
        save_dir: Optional directory to save figures. If None, saves to current directory.
    """
    global logger
    logger = setup_logging(log_dir=log_dir)
    
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("SIMULATION RUN STARTED")
    logger.info(f"Mode: {'UTILITY' if utility_mode else 'PROBABILITY'}")
    if not utility_mode:
        logger.info(f"Punishment Mode: {punishment_mode}")
    if save_dir:
        logger.info(f"Save Directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    logger.info("=" * 60)
    
    logger.info("Building model context...")
    model_start = time.time()
    config = {'GRID_SIZE': 200, 'BETA_NAIVE': 10.0}
    logger.info(f"Config: {config}")
    model_ctx = build_model_context(config)
    logger.info(f"Model context built in {time.time() - model_start:.2f}s")
    
    # Run all three figures
    logger.info("Running 3 figure simulations...")
    
    def get_save_path(fname):
        if save_dir:
            return os.path.join(save_dir, fname)
        return fname

    if utility_mode:
        logger.info("[Figure 1/3] (Utility Mode)")
        run_figure_1_utility(model_ctx, get_save_path("fig1_utility_polarized_beliefs.png"))
        
        logger.info("[Figure 2/3] (Utility Mode)")
        run_figure_2_utility(model_ctx, get_save_path("fig2_utility_wrongness_polarization.png"))
        
        logger.info("[Figure 3/3] (Utility Mode)")
        run_figure_3_utility(model_ctx, get_save_path("fig3_utility_trust_polarization.png"))
    else:
        logger.info("[Figure 1/3]")
        run_figure_1(model_ctx, get_save_path("fig1_polarized_beliefs.png"), punishment_mode=punishment_mode)
        
        logger.info("[Figure 2/3]")
        run_figure_2(model_ctx, get_save_path("fig2_wrongness_polarization.png"), punishment_mode=punishment_mode)
        
        logger.info("[Figure 3/3]")
        run_figure_3(model_ctx, get_save_path("fig3_trust_polarization.png"), punishment_mode=punishment_mode)
    
    logger.info("=" * 60)
    logger.info(f"ALL SIMULATIONS COMPLETED in {time.time() - total_start:.2f}s total")
    logger.info("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate punishment model visualizations")
    parser.add_argument(
        "--utility-mode",
        action="store_true",
        default=False,
        help="Generate utility component plots instead of probability plots"
    )
    parser.add_argument(
        "--punishment",
        type=str,
        choices=['mild', 'none', 'total-punishment'],
        default='mild',
        help="For probability plots: 'mild' shows P(Mild) as dashed, 'none' shows P(None) as dashed, "
             "'total-punishment' shows only P(Mild)+P(Harsh) as solid (default: mild)"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs",
        help="Directory to store log files (default: logs)"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="Directory to save generated figures (default: current directory)"
    )
    
    args = parser.parse_args()
    main(log_dir=args.log_dir, utility_mode=args.utility_mode, punishment_mode=args.punishment, save_dir=args.save_dir)