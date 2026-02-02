import argparse
import sys

# -----------------------------------------------------------------------------
# DEVICE CONFIGURATION (must happen before other JAX imports)
# -----------------------------------------------------------------------------

def configure_device(device='cpu'):
    """Configure JAX to use the specified device.
    
    Args:
        device: 'cpu' for CPU, 'gpu' for GPU (uses first available GPU)
    
    Must be called BEFORE importing any modules that use JAX.
    Returns the actual device being used (may differ if GPU unavailable).
    """
    import jax
    
    if device not in ('cpu', 'gpu'):
        raise ValueError(f"Unknown device: {device}. Use 'cpu' or 'gpu'.")
    
    actual_device = device
    
    if device == 'gpu':
        # Attempt to configure GPU - set platform first, then verify it works
        try:
            jax.config.update('jax_platform_name', 'gpu')
            # Force JAX to actually initialize the backend by querying devices
            # This will raise an exception if GPU isn't available
            gpu_devices = jax.devices()
            print(f"JAX configured to use: gpu")
            print(f"Available GPU devices: {gpu_devices}")
        except Exception as e:
            # GPU not available (e.g., CUDA-enabled jaxlib not installed)
            print(f"WARNING: GPU not available: {e}")
            print("Falling back to CPU.")
            actual_device = 'cpu'
            jax.config.update('jax_platform_name', 'cpu')
            print(f"JAX configured to use: cpu")
            print(f"Available CPU devices: {jax.devices()}")
    else:
        # CPU requested
        jax.config.update('jax_platform_name', 'cpu')
        print(f"JAX configured to use: cpu")
        print(f"Available CPU devices: {jax.devices()}")
    
    return actual_device

# Parse device argument early, before other imports
def parse_device_arg():
    """Parse just the --device argument for early JAX configuration."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'gpu'],
                        help="Device to use for computation: 'cpu' or 'gpu' (default: cpu)")
    args, _ = parser.parse_known_args()
    return args.device

# Configure device before importing JAX-dependent modules
if __name__ == "__main__":
    _device = parse_device_arg()
    configure_device(_device)

# Now import JAX and other modules
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import logging
import time
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
        weights['scale_comm'] = 5.0 
        
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

def run_subplot_polarized_motives_utility(model_ctx, ax, agent_type, w_values, punishment_mode='none'):
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
    y_mild = {'u_int': [], 'u_rep': [], 'u_comm': []}
    y_none = {'u_int': [], 'u_rep': [], 'u_comm': []}
    
    for w in w_values:
        # Get utilities for reputational agent (has u_int and u_rep)
        utils_rep = get_utility_components(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_rep)
        # Get utilities for communicative agent (has u_int and u_comm)
        utils_comm = get_utility_components(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_comm)
        
        # Harsh (action 2)
        y_harsh['u_int'].append(utils_rep[2]['u_int'])
        y_harsh['u_rep'].append(utils_rep[2]['u_rep'])
        y_harsh['u_comm'].append(utils_comm[2]['u_comm'])
        
        # Mild (action 1)
        y_mild['u_int'].append(utils_rep[1]['u_int'])
        y_mild['u_rep'].append(utils_rep[1]['u_rep'])
        y_mild['u_comm'].append(utils_comm[1]['u_comm'])
        
        # None (action 0)
        y_none['u_int'].append(utils_rep[0]['u_int'])
        y_none['u_rep'].append(utils_rep[0]['u_rep'])
        y_none['u_comm'].append(utils_comm[0]['u_comm'])
    
    # Plot: solid for Harsh, dashed for Mild (or None), dotted for None (if 'all' mode)
    for comp in ['u_int', 'u_rep', 'u_comm']:
        ax.plot(w_values, y_harsh[comp], color=UTILITY_COLORS[comp], linestyle='-', 
                label=f"{UTILITY_LABELS[comp]} (Harsh)")
        if punishment_mode == 'all':
            ax.plot(w_values, y_mild[comp], color=UTILITY_COLORS[comp], linestyle='--',
                    label=f"{UTILITY_LABELS[comp]} (Mild)")
            ax.plot(w_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle=':',
                    label=f"{UTILITY_LABELS[comp]} (None)")
        else:
            ax.plot(w_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle='--',
                    label=f"{UTILITY_LABELS[comp]} (None)")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("Utility")

def run_subplot_polarized_wrongness_utility(model_ctx, ax, agent_type, w_values, punishment_mode='none'):
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
    y_none = {'u_int': [], 'u_rep': [], 'u_comm': []}
    
    for w in w_values:
        utils_rep = get_utility_components(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_rep)
        utils_comm = get_utility_components(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_comm)
        
        y_harsh['u_int'].append(utils_rep[2]['u_int'])
        y_harsh['u_rep'].append(utils_rep[2]['u_rep'])
        y_harsh['u_comm'].append(utils_comm[2]['u_comm'])
        
        y_mild['u_int'].append(utils_rep[1]['u_int'])
        y_mild['u_rep'].append(utils_rep[1]['u_rep'])
        y_mild['u_comm'].append(utils_comm[1]['u_comm'])
        
        y_none['u_int'].append(utils_rep[0]['u_int'])
        y_none['u_rep'].append(utils_rep[0]['u_rep'])
        y_none['u_comm'].append(utils_comm[0]['u_comm'])
    
    for comp in ['u_int', 'u_rep', 'u_comm']:
        ax.plot(w_values, y_harsh[comp], color=UTILITY_COLORS[comp], linestyle='-',
                label=f"{UTILITY_LABELS[comp]} (Harsh)")
        if punishment_mode == 'all':
            ax.plot(w_values, y_mild[comp], color=UTILITY_COLORS[comp], linestyle='--',
                    label=f"{UTILITY_LABELS[comp]} (Mild)")
            ax.plot(w_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle=':',
                    label=f"{UTILITY_LABELS[comp]} (None)")
        else:
            ax.plot(w_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle='--',
                    label=f"{UTILITY_LABELS[comp]} (None)")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("Utility")

def run_subplot_wrongness_polarization_utility(model_ctx, ax, agent_type, pol_values, auth_w, punishment_mode='none'):
    """Utility-mode: Wrongness Polarization - sweep polarization level at fixed authority W."""
    true_j, true_b = agent_type['j'], agent_type['b']
    p_motives = ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
    
    weights_rep = get_agent_weights('reputation')
    weights_comm = get_agent_weights('communicative')
    
    y_harsh = {'u_int': [], 'u_rep': [], 'u_comm': []}
    y_mild = {'u_int': [], 'u_rep': [], 'u_comm': []}
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
        
        y_mild['u_int'].append(utils_rep[1]['u_int'])
        y_mild['u_rep'].append(utils_rep[1]['u_rep'])
        y_mild['u_comm'].append(utils_comm[1]['u_comm'])
        
        y_none['u_int'].append(utils_rep[0]['u_int'])
        y_none['u_rep'].append(utils_rep[0]['u_rep'])
        y_none['u_comm'].append(utils_comm[0]['u_comm'])
    
    for comp in ['u_int', 'u_rep', 'u_comm']:
        ax.plot(pol_values, y_harsh[comp], color=UTILITY_COLORS[comp], linestyle='-',
                label=f"{UTILITY_LABELS[comp]} (Harsh)")
        if punishment_mode == 'all':
            ax.plot(pol_values, y_mild[comp], color=UTILITY_COLORS[comp], linestyle='--',
                    label=f"{UTILITY_LABELS[comp]} (Mild)")
            ax.plot(pol_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle=':',
                    label=f"{UTILITY_LABELS[comp]} (None)")
        else:
            ax.plot(pol_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle='--',
                    label=f"{UTILITY_LABELS[comp]} (None)")
    
    ax.set_xlabel("Polarization Level")
    ax.set_ylabel("Utility")

def run_subplot_trust_polarization_utility(model_ctx, ax, agent_type, pol_values, auth_w, punishment_mode='none'):
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
    y_mild = {'u_int': [], 'u_rep': [], 'u_comm': []}
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
        
        y_mild['u_int'].append(utils_rep[1]['u_int'])
        y_mild['u_rep'].append(utils_rep[1]['u_rep'])
        y_mild['u_comm'].append(utils_comm[1]['u_comm'])
        
        y_none['u_int'].append(utils_rep[0]['u_int'])
        y_none['u_rep'].append(utils_rep[0]['u_rep'])
        y_none['u_comm'].append(utils_comm[0]['u_comm'])
    
    for comp in ['u_int', 'u_rep', 'u_comm']:
        ax.plot(pol_values, y_harsh[comp], color=UTILITY_COLORS[comp], linestyle='-',
                label=f"{UTILITY_LABELS[comp]} (Harsh)")
        if punishment_mode == 'all':
            ax.plot(pol_values, y_mild[comp], color=UTILITY_COLORS[comp], linestyle='--',
                    label=f"{UTILITY_LABELS[comp]} (Mild)")
            ax.plot(pol_values, y_none[comp], color=UTILITY_COLORS[comp], linestyle=':',
                    label=f"{UTILITY_LABELS[comp]} (None)")
        else:
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
    w_values = np.linspace(0.0, 1.0, 100) # changed to be more fine-grained
    
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
    Row 2: Authority believes UNCERTAIN (W=0.5)
    Row 3: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 2: Wrongness Polarization")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    title_suffix = "Total Punishment" if punishment_mode == 'total-punishment' else "Punishment"
    fig.suptitle(f"Wrongness Polarization: {title_suffix} Probability vs Audience Polarization Level", fontsize=16, fontweight='bold')
    
    agents = ['naive', 'reputation', 'communicative']
    colors = {'naive': 'green', 'reputation': 'blue', 'communicative': 'red'}
    pol_values = np.linspace(0.0, 1.0, 100)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Auth believes WRONG (W=1.0)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAuthority: W=1.0 (Wrong)", fontsize=10)
        run_subplot_wrongness_polarization(model_ctx, ax1, agent_type, agents, colors, pol_values, auth_w=1.0, punishment_mode=punishment_mode)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=7)
        
        # Row 2: Auth believes UNCERTAIN (W=0.5)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.5 (Uncertain)", fontsize=10)
        run_subplot_wrongness_polarization(model_ctx, ax2, agent_type, agents, colors, pol_values, auth_w=0.5, punishment_mode=punishment_mode)
        
        # Row 3: Auth believes NOT WRONG (W=0.0)
        ax3 = axes[2, col]
        ax3.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_wrongness_polarization(model_ctx, ax3, agent_type, agents, colors, pol_values, auth_w=0.0, punishment_mode=punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 2 saved to {filename} in {time.time() - figure_start:.2f}s")

def run_figure_3(model_ctx, filename="fig3_trust_polarization.png", punishment_mode='mild'):
    """
    Figure 3: Trust Polarization
    Row 1: Authority believes WRONG (W=1.0)
    Row 2: Authority believes UNCERTAIN (W=0.5)
    Row 3: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 3: Trust Polarization")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    title_suffix = "Total Punishment" if punishment_mode == 'total-punishment' else "Punishment"
    fig.suptitle(f"Trust Polarization: {title_suffix} Probability vs Out-Group Distrust Level", fontsize=16, fontweight='bold')
    
    agents = ['naive', 'reputation', 'communicative']
    colors = {'naive': 'green', 'reputation': 'blue', 'communicative': 'red'}
    pol_values = np.linspace(0.0, 1.0, 100)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Auth believes WRONG (W=1.0)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAuthority: W=1.0 (Wrong)", fontsize=10)
        run_subplot_trust_polarization(model_ctx, ax1, agent_type, agents, colors, pol_values, auth_w=1.0, punishment_mode=punishment_mode)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=7)
        
        # Row 2: Auth believes UNCERTAIN (W=0.5)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.5 (Uncertain)", fontsize=10)
        run_subplot_trust_polarization(model_ctx, ax2, agent_type, agents, colors, pol_values, auth_w=0.5, punishment_mode=punishment_mode)
        
        # Row 3: Auth believes NOT WRONG (W=0.0)
        ax3 = axes[2, col]
        ax3.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_trust_polarization(model_ctx, ax3, agent_type, agents, colors, pol_values, auth_w=0.0, punishment_mode=punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 3 saved to {filename} in {time.time() - figure_start:.2f}s")

# -----------------------------------------------------------------------------
# UTILITY-MODE FIGURE FUNCTIONS
# -----------------------------------------------------------------------------

def run_figure_1_utility(model_ctx, filename="fig1_utility_polarized_beliefs.png", punishment_mode='none'):
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
    title_suffix = " (All Actions)" if punishment_mode == 'all' else ""
    fig.suptitle(f"Polarized Beliefs: Utility Components vs Authority's Belief (W){title_suffix}", fontsize=16, fontweight='bold')
    
    w_values = np.linspace(0.0, 1.0, 100) # changed to be more fine-grained
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Polarized Motives
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nPolarized Motives (Uncertain W)", fontsize=10)
        run_subplot_polarized_motives_utility(model_ctx, ax1, agent_type, w_values, punishment_mode)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=5 if punishment_mode == 'all' else 6)
        
        # Row 2: Polarized Wrongness
        ax2 = axes[1, col]
        ax2.set_title(f"Polarized Wrongness (Uncertain Motives)", fontsize=10)
        run_subplot_polarized_wrongness_utility(model_ctx, ax2, agent_type, w_values, punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 1 (Utility) saved to {filename} in {time.time() - figure_start:.2f}s")

def run_figure_2_utility(model_ctx, filename="fig2_utility_wrongness_polarization.png", punishment_mode='none'):
    """
    Figure 2 (Utility Mode): Wrongness Polarization
    Row 1: Authority believes WRONG (W=1.0)
    Row 2: Authority believes UNCERTAIN (W=0.5)
    Row 3: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 2 (Utility): Wrongness Polarization")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    title_suffix = " (All Actions)" if punishment_mode == 'all' else ""
    fig.suptitle(f"Wrongness Polarization: Utility Components vs Audience Polarization Level{title_suffix}", fontsize=16, fontweight='bold')
    
    pol_values = np.linspace(0.0, 1.0, 100)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Auth believes WRONG (W=1.0)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAuthority: W=1.0 (Wrong)", fontsize=10)
        run_subplot_wrongness_polarization_utility(model_ctx, ax1, agent_type, pol_values, auth_w=1.0, punishment_mode=punishment_mode)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=5 if punishment_mode == 'all' else 6)
        
        # Row 2: Auth believes UNCERTAIN (W=0.5)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.5 (Uncertain)", fontsize=10)
        run_subplot_wrongness_polarization_utility(model_ctx, ax2, agent_type, pol_values, auth_w=0.5, punishment_mode=punishment_mode)
        
        # Row 3: Auth believes NOT WRONG (W=0.0)
        ax3 = axes[2, col]
        ax3.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_wrongness_polarization_utility(model_ctx, ax3, agent_type, pol_values, auth_w=0.0, punishment_mode=punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 2 (Utility) saved to {filename} in {time.time() - figure_start:.2f}s")

def run_figure_3_utility(model_ctx, filename="fig3_utility_trust_polarization.png", punishment_mode='none'):
    """
    Figure 3 (Utility Mode): Trust Polarization
    Row 1: Authority believes WRONG (W=1.0)
    Row 2: Authority believes UNCERTAIN (W=0.5)
    Row 3: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 3 (Utility): Trust Polarization")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    title_suffix = " (All Actions)" if punishment_mode == 'all' else ""
    fig.suptitle(f"Trust Polarization: Utility Components vs Out-Group Distrust Level{title_suffix}", fontsize=16, fontweight='bold')
    
    pol_values = np.linspace(0.0, 1.0, 100)
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: Auth believes WRONG (W=1.0)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAuthority: W=1.0 (Wrong)", fontsize=10)
        run_subplot_trust_polarization_utility(model_ctx, ax1, agent_type, pol_values, auth_w=1.0, punishment_mode=punishment_mode)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=5 if punishment_mode == 'all' else 6)
        
        # Row 2: Auth believes UNCERTAIN (W=0.5)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.5 (Uncertain)", fontsize=10)
        run_subplot_trust_polarization_utility(model_ctx, ax2, agent_type, pol_values, auth_w=0.5, punishment_mode=punishment_mode)
        
        # Row 3: Auth believes NOT WRONG (W=0.0)
        ax3 = axes[2, col]
        ax3.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_trust_polarization_utility(model_ctx, ax3, agent_type, pol_values, auth_w=0.0, punishment_mode=punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 3 (Utility) saved to {filename} in {time.time() - figure_start:.2f}s")

# -----------------------------------------------------------------------------
# BELIEF UPDATE FIGURE FUNCTIONS
# -----------------------------------------------------------------------------

def run_subplot_all_actions_polarized_wrongness(model_ctx, ax, agent_type, agents, colors, w_values):
    """Subplot: Polarized Wrongness, Uncertain Motives - all 3 actions per agent (9 lines total)."""
    true_j, true_b = agent_type['j'], agent_type['b']
    
    # Polarized wrongness, uncertain motives priors (same as run_subplot_polarized_wrongness)
    p_j_unc = (1.0, 1.0)
    p_b_unc = (1.0, 1.0)
    p_in_w = get_beta_params(0.9, 20)
    p_out_w = get_beta_params(0.1, 20)
    priors_in = (p_in_w, p_b_unc, p_j_unc)
    priors_out = (p_out_w, p_b_unc, p_j_unc)
    
    action_styles = {0: ':', 1: '--', 2: '-'}  # None: dotted, Mild: dashed, Harsh: solid
    action_labels = {0: 'None', 1: 'Mild', 2: 'Harsh'}
    
    for ag in agents:
        weights = get_agent_weights(ag)
        y_actions = {0: [], 1: [], 2: []}
        
        for w in w_values:
            probs = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights)
            for a_idx in [0, 1, 2]:
                y_actions[a_idx].append(probs[a_idx])
        
        for a_idx in [0, 1, 2]:
            ax.plot(w_values, y_actions[a_idx], color=colors[ag], linestyle=action_styles[a_idx],
                    label=f"{ag.capitalize()} ({action_labels[a_idx]})")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("P(Action)")


def run_subplot_belief_update(model_ctx, ax, agent_type, action_idx, group_type):
    """
    Subplot: Show prior and posterior belief about W after observing an action.
    
    Args:
        model_ctx: The model context
        ax: The matplotlib axis
        agent_type: Dict with 'j', 'b', 'label' keys (used for context/title)
        action_idx: 0 (None), 1 (Mild), or 2 (Harsh)
        group_type: 'in' for in-group, 'out' for out-group, or 'both'
    """
    # Polarized wrongness, uncertain motives priors (same as the main plot)
    p_j_unc = (1.0, 1.0)
    p_b_unc = (1.0, 1.0)
    p_in_w = get_beta_params(0.9, 20)
    p_out_w = get_beta_params(0.1, 20)
    
    # Create prior tensors
    priors_in = (p_in_w, p_b_unc, p_j_unc)
    priors_out = (p_out_w, p_b_unc, p_j_unc)
    
    prior_tensor_in = model_ctx.create_prior_tensor(*priors_in)
    prior_tensor_out = model_ctx.create_prior_tensor(*priors_out)
    
    # Get the W grid for x-axis
    W_GRID = np.array(model_ctx.W_GRID)
    
    # Compute prior P(W) by marginalizing over B and J
    prior_w_in = np.array(jnp.sum(prior_tensor_in, axis=(1, 2)))
    prior_w_out = np.array(jnp.sum(prior_tensor_out, axis=(1, 2)))
    
    # Perform Bayesian update after observing the action
    posterior_tensor_in = model_ctx.observer_update(prior_tensor_in, action_idx)
    posterior_tensor_out = model_ctx.observer_update(prior_tensor_out, action_idx)
    
    # Compute posterior P(W | action)
    posterior_w_in = np.array(jnp.sum(posterior_tensor_in, axis=(1, 2)))
    posterior_w_out = np.array(jnp.sum(posterior_tensor_out, axis=(1, 2)))
    
    # Colors for in-group and out-group
    color_in = 'blue'
    color_out = 'red'
    
    # Plot priors (dashed) and posteriors (solid)
    ax.plot(W_GRID, prior_w_in, color=color_in, linestyle='--', linewidth=1.5, 
            label='In-group Prior', alpha=0.7)
    ax.plot(W_GRID, posterior_w_in, color=color_in, linestyle='-', linewidth=2, 
            label='In-group Posterior')
    ax.plot(W_GRID, prior_w_out, color=color_out, linestyle='--', linewidth=1.5, 
            label='Out-group Prior', alpha=0.7)
    ax.plot(W_GRID, posterior_w_out, color=color_out, linestyle='-', linewidth=2, 
            label='Out-group Posterior')
    
    ax.set_xlabel("Wrongness (W)")
    ax.set_ylabel("P(W)")
    ax.set_xlim(0, 1)


def run_figure_belief_update(model_ctx, filename="fig_belief_update.png"):
    """
    Figure: Belief Update Visualization
    
    Row 1: Action probabilities (None, Mild, Harsh) for all agents - Polarized Wrongness scenario
    Row 2: Belief update P(W) after observing None action
    Row 3: Belief update P(W) after observing Mild action  
    Row 4: Belief update P(W) after observing Harsh action
    
    Columns: 4 agent types (High J Anti-B, High J Pro-B, Low J Anti-B, Low J Pro-B)
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Belief Update Figure")
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 20))
    fig.suptitle("Belief Update: Action Probabilities and Observer Inference\n(Polarized Wrongness, Uncertain Motives)", 
                 fontsize=16, fontweight='bold')
    
    agents = ['naive', 'reputation', 'communicative']
    colors = {'naive': 'green', 'reputation': 'blue', 'communicative': 'red'}
    w_values = np.linspace(0.0, 1.0, 100)
    
    action_names = {0: 'None', 1: 'Mild', 2: 'Harsh'}
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: All action probabilities (9 lines)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nAction Probabilities", fontsize=10)
        run_subplot_all_actions_polarized_wrongness(model_ctx, ax1, agent_type, agents, colors, w_values)
        if col == 0:
            ax1.legend(loc='upper left', fontsize=5)
        
        # Rows 2-4: Belief updates for each action
        for row, action_idx in enumerate([0, 1, 2], start=1):
            ax = axes[row, col]
            ax.set_title(f"Belief Update After {action_names[action_idx]} Action", fontsize=10)
            run_subplot_belief_update(model_ctx, ax, agent_type, action_idx, 'both')
            if col == 0:
                ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Belief Update Figure saved to {filename} in {time.time() - figure_start:.2f}s")


# -----------------------------------------------------------------------------
# WASSERSTEIN-1 FIGURE FUNCTIONS
# -----------------------------------------------------------------------------

def compute_wasserstein1_point_mass(p_w, w_true, W_GRID):
    """
    Compute the Wasserstein-1 distance between a discrete distribution p_w
    and a point mass at w_true.
    
    For a point mass, W_1(P, delta_{w_true}) = E_P[|W - w_true|]
    = sum_i p(w_i) * |w_i - w_true|
    
    Args:
        p_w: Probability distribution over W_GRID (should sum to 1)
        w_true: Authority's true wrongness belief (scalar)
        W_GRID: The grid of W values
    
    Returns:
        The Wasserstein-1 distance (scalar)
    """
    # Ensure distribution is normalized
    p_w = p_w / (np.sum(p_w) + 1e-10)
    # Compute absolute distances from the point mass
    abs_distances = np.abs(W_GRID - w_true)
    # Expected absolute distance under distribution p_w
    return np.sum(p_w * abs_distances)


def run_subplot_w1_vs_authority_belief(model_ctx, ax, agent_type, group_type, w_values):
    """
    Subplot: W-1 distance between group's posterior P(W) and authority's belief vs authority's W.
    
    Args:
        model_ctx: The model context
        ax: The matplotlib axis
        agent_type: Dict with 'j', 'b', 'label' keys
        group_type: 'in' for in-group (high-W prior), 'out' for out-group (low-W prior)
        w_values: Array of authority wrongness beliefs to sweep over
    """
    # Polarized wrongness, uncertain motives priors
    p_j_unc = (1.0, 1.0)
    p_b_unc = (1.0, 1.0)
    p_in_w = get_beta_params(0.9, 20)  # In-group: high W prior
    p_out_w = get_beta_params(0.1, 20)  # Out-group: low W prior
    
    # Create prior tensors
    priors_in = (p_in_w, p_b_unc, p_j_unc)
    priors_out = (p_out_w, p_b_unc, p_j_unc)
    
    prior_tensor_in = model_ctx.create_prior_tensor(*priors_in)
    prior_tensor_out = model_ctx.create_prior_tensor(*priors_out)
    
    # Select which group to analyze
    if group_type == 'in':
        prior_tensor = prior_tensor_in
    else:
        prior_tensor = prior_tensor_out
    
    # Get the W grid
    W_GRID = np.array(model_ctx.W_GRID)
    
    action_names = {0: 'None', 1: 'Mild', 2: 'Harsh'}
    action_styles = {0: ':', 1: '--', 2: '-'}  # None: dotted, Mild: dashed, Harsh: solid
    
    # For each action, compute W-1 at each authority W value
    for action_idx in [0, 1, 2]:
        w1_values = []
        
        # Compute posterior after observing this action
        posterior_tensor = model_ctx.observer_update(prior_tensor, action_idx)
        posterior_w = np.array(jnp.sum(posterior_tensor, axis=(1, 2)))
        
        for auth_w in w_values:
            # Compute W-1 between posterior and authority's point belief
            w1_val = compute_wasserstein1_point_mass(posterior_w, auth_w, W_GRID)
            w1_values.append(w1_val)
        
        if group_type == 'in':
            color = 'blue'
        elif group_type == 'out':
            color = 'red'
            
        ax.plot(w_values, w1_values, color=color, linestyle=action_styles[action_idx],
                linewidth=2, label=f"After {action_names[action_idx]}")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("W-1(Posterior, Authority)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)


def run_figure_w1(model_ctx, filename="fig_w1.png"):
    """
    Figure: Wasserstein-1 Distance Visualization
    
    Row 1: W-1 between in-group posterior and authority belief vs authority W
           (3 blue lines: None, Mild, Harsh actions)
    Row 2: W-1 between out-group posterior and authority belief vs authority W
           (3 red lines: None, Mild, Harsh actions)
    Row 3: Belief update P(W) after observing None action
    Row 4: Belief update P(W) after observing Mild action  
    Row 5: Belief update P(W) after observing Harsh action
    
    Columns: 4 agent types (High J Anti-B, High J Pro-B, Low J Anti-B, Low J Pro-B)
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting W-1 Figure")
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle("Wasserstein-1 Analysis: Posterior vs Authority Belief\n(Polarized Wrongness, Uncertain Motives)", 
                 fontsize=16, fontweight='bold')
    
    w_values = np.linspace(0.0, 1.0, 100)
    action_names = {0: 'None', 1: 'Mild', 2: 'Harsh'}
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: W-1 for in-group (high-W prior)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nIn-Group W-1 (High-W Prior)", fontsize=10)
        run_subplot_w1_vs_authority_belief(model_ctx, ax1, agent_type, 'in', w_values)
        if col == 0:
            ax1.legend(loc='upper right', fontsize=8)
        
        # Row 2: W-1 for out-group (low-W prior)
        ax2 = axes[1, col]
        ax2.set_title(f"Out-Group W-1 (Low-W Prior)", fontsize=10)
        run_subplot_w1_vs_authority_belief(model_ctx, ax2, agent_type, 'out', w_values)
        if col == 0:
            ax2.legend(loc='upper right', fontsize=8)
        
        # Rows 3-5: Belief updates for each action
        for row, action_idx in enumerate([0, 1, 2], start=2):
            ax = axes[row, col]
            ax.set_title(f"Belief Update After {action_names[action_idx]} Action", fontsize=10)
            run_subplot_belief_update(model_ctx, ax, agent_type, action_idx, 'both')
            if col == 0:
                ax.legend(loc='upper right', fontsize=7)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"W-1 Figure saved to {filename} in {time.time() - figure_start:.2f}s")

# -----------------------------------------------------------------------------
# FINAL PLOTS HELPER FUNCTIONS
# -----------------------------------------------------------------------------

# Reordered agent types for final plots: High J Anti-B, Low J Anti-B, High J Pro-B, Low J Pro-B
FINAL_AGENT_ORDER = [0, 2, 1, 3]  # Indices into AGENT_TYPES

# Style constants for final plots - MODIFY THESE TO ADJUST FONT SIZES
FINAL_LINEWIDTH = 3
FINAL_INSET_LINEWIDTH = 2
FINAL_TITLE_FONTSIZE = 16          # Title font size
FINAL_LABEL_FONTSIZE = 15          # Axis label font size
FINAL_TICK_FONTSIZE = 13           # Tick label font size
FINAL_LEGEND_FONTSIZE = 13         # Legend font size
FINAL_INSET_TICK_FONTSIZE = 11     # Inset tick font size
FINAL_SUBPLOT_LABEL_FONTSIZE = 18  # Subplot label (A, B, C...) font size
FINAL_SUPTITLE_FONTSIZE = 20       # Figure suptitle font size

# Multiplier for Figure 2 Alternative (1.25x larger fonts)
FIG2_ALT_FONT_MULTIPLIER = 1.25

# Color for naive agent in final plots (dark grey instead of green)
NAIVE_COLOR = '#404040'  # Dark grey

# Line styles for utility plots
LINESTYLE_HARSH = '-'       # Solid
LINESTYLE_MILD = ':'        # Dotted
LINESTYLE_NONE = '-.'       # Dash-dot


def run_subplot_polarized_wrongness_final(model_ctx, ax, agent_type, w_values, show_ylabel=True):
    """
    Subplot: Polarized Wrongness, Uncertain Motives - reputational and naive agents.
    Shows P(Harsh) and P(Mild) for both agents.
    """
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_j_unc = (1.0, 1.0)
    p_b_unc = (1.0, 1.0)
    p_in_w = get_beta_params(0.9, 20)
    p_out_w = get_beta_params(0.1, 20)
    priors_in = (p_in_w, p_b_unc, p_j_unc)
    priors_out = (p_out_w, p_b_unc, p_j_unc)
    
    # Reputational agent (blue)
    weights_rep = get_agent_weights('reputation')
    y_harsh_rep, y_mild_rep = [], []
    
    # Naive agent (dark grey)
    weights_naive = get_agent_weights('naive')
    y_harsh_naive, y_mild_naive = [], []
    
    for w in w_values:
        probs_rep = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_rep)
        y_harsh_rep.append(probs_rep[2])
        y_mild_rep.append(probs_rep[1])
        
        probs_naive = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_naive)
        y_harsh_naive.append(probs_naive[2])
        y_mild_naive.append(probs_naive[1])
    
    # Plot reputational (blue)
    ax.plot(w_values, y_harsh_rep, color='blue', linestyle='-', linewidth=FINAL_LINEWIDTH, label='Rep. Harsh')
    ax.plot(w_values, y_mild_rep, color='blue', linestyle='--', linewidth=FINAL_LINEWIDTH, label='Rep. Mild')
    
    # Plot naive (dark grey)
    ax.plot(w_values, y_harsh_naive, color=NAIVE_COLOR, linestyle='-', linewidth=FINAL_LINEWIDTH, label='Naive Harsh')
    ax.plot(w_values, y_mild_naive, color=NAIVE_COLOR, linestyle='--', linewidth=FINAL_LINEWIDTH, label='Naive Mild')
    
    ax.set_xlabel("Authority's Belief (W)", fontsize=FINAL_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("P(Action)", fontsize=FINAL_LABEL_FONTSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.1)  # Extended y-axis to show full lines at 0 and 1
    ax.tick_params(axis='both', which='major', labelsize=FINAL_TICK_FONTSIZE)


def run_subplot_wrongness_polarization_utility_final(model_ctx, ax, agent_type, pol_values, auth_w, show_ylabel=True):
    """
    Subplot: Wrongness Polarization Utility - only reputational agent, all 3 actions.
    Shows utility for None, Mild, and Harsh actions.
    Uses consistent line styles: Harsh=solid, Mild=dotted, None=dash-dot.
    """
    true_j, true_b = agent_type['j'], agent_type['b']
    p_motives = ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
    
    weights_rep = get_agent_weights('reputation')
    
    y_harsh, y_mild, y_none = [], [], []
    
    for p in pol_values:
        in_mean = 0.5 + p * 0.45
        out_mean = 0.5 - p * 0.45
        p_in = (get_beta_params(in_mean, 20), p_motives[1], p_motives[2])
        p_out = (get_beta_params(out_mean, 20), p_motives[1], p_motives[2])
        
        utils_rep = get_utility_components(model_ctx, p_in, p_out, auth_w, true_b, true_j, weights_rep)
        
        y_harsh.append(utils_rep[2]['u_rep'])
        y_mild.append(utils_rep[1]['u_rep'])
        y_none.append(utils_rep[0]['u_rep'])
    
    ax.plot(pol_values, y_harsh, color='blue', linestyle=LINESTYLE_HARSH, linewidth=FINAL_LINEWIDTH, label='Harsh')
    ax.plot(pol_values, y_mild, color='blue', linestyle=LINESTYLE_MILD, linewidth=FINAL_LINEWIDTH, label='Mild')
    ax.plot(pol_values, y_none, color='blue', linestyle=LINESTYLE_NONE, linewidth=FINAL_LINEWIDTH, label='None')
    
    ax.set_xlabel("Polarization Level", fontsize=FINAL_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("Reputational Utility", fontsize=FINAL_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FINAL_TICK_FONTSIZE)


def run_subplot_polarized_motives_final(model_ctx, ax, agent_type, w_values, show_separate=False, show_ylabel=True):
    """
    Subplot: Polarized Motives, Uncertain Wrongness - reputational and naive agents.
    
    Args:
        show_separate: If False, show P(Punishment) = P(Harsh) + P(Mild) as single curve.
                       If True, show P(Harsh) and P(Mild) separately.
    """
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
    
    # Reputational agent (blue)
    weights_rep = get_agent_weights('reputation')
    y_total_rep, y_harsh_rep, y_mild_rep = [], [], []
    
    # Naive agent (dark grey)
    weights_naive = get_agent_weights('naive')
    y_total_naive, y_harsh_naive, y_mild_naive = [], [], []
    
    for w in w_values:
        probs_rep = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_rep)
        y_total_rep.append(probs_rep[1] + probs_rep[2])
        y_harsh_rep.append(probs_rep[2])
        y_mild_rep.append(probs_rep[1])
        
        probs_naive = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_naive)
        y_total_naive.append(probs_naive[1] + probs_naive[2])
        y_harsh_naive.append(probs_naive[2])
        y_mild_naive.append(probs_naive[1])
    
    if show_separate:
        # Reputational (blue)
        ax.plot(w_values, y_harsh_rep, color='blue', linestyle='-', linewidth=FINAL_LINEWIDTH, label='Rep. Harsh')
        ax.plot(w_values, y_mild_rep, color='blue', linestyle='--', linewidth=FINAL_LINEWIDTH, label='Rep. Mild')
        # Naive (dark grey)
        ax.plot(w_values, y_harsh_naive, color=NAIVE_COLOR, linestyle='-', linewidth=FINAL_LINEWIDTH, label='Naive Harsh')
        ax.plot(w_values, y_mild_naive, color=NAIVE_COLOR, linestyle='--', linewidth=FINAL_LINEWIDTH, label='Naive Mild')
        if show_ylabel:
            ax.set_ylabel("P(Action)", fontsize=FINAL_LABEL_FONTSIZE)
    else:
        ax.plot(w_values, y_total_rep, color='blue', linestyle='-', linewidth=FINAL_LINEWIDTH, label='Reputational')
        ax.plot(w_values, y_total_naive, color=NAIVE_COLOR, linestyle='-', linewidth=FINAL_LINEWIDTH, label='Naive')
        if show_ylabel:
            ax.set_ylabel("P(Punishment)", fontsize=FINAL_LABEL_FONTSIZE)
    
    ax.set_xlabel("Authority's Belief (W)", fontsize=FINAL_LABEL_FONTSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=FINAL_TICK_FONTSIZE)


def run_subplot_polarized_motives_final_v2(model_ctx, ax, agent_type, w_values, show_total=True, show_ylabel=True, font_mult=1.0):
    """
    Subplot: Polarized Motives, Uncertain Wrongness - reputational and naive agents.
    Version 2 for Figure 2 Alternative.
    
    Args:
        show_total: If True, show P(Punishment) dashed. If False, show P(Harsh) solid and P(Mild) dotted.
        font_mult: Font size multiplier (default 1.0).
    """
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
    
    # Reputational agent (blue)
    weights_rep = get_agent_weights('reputation')
    y_total_rep, y_harsh_rep, y_mild_rep = [], [], []
    
    # Naive agent (dark grey)
    weights_naive = get_agent_weights('naive')
    y_total_naive, y_harsh_naive, y_mild_naive = [], [], []
    
    for w in w_values:
        probs_rep = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_rep)
        y_total_rep.append(probs_rep[1] + probs_rep[2])
        y_harsh_rep.append(probs_rep[2])
        y_mild_rep.append(probs_rep[1])
        
        probs_naive = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_naive)
        y_total_naive.append(probs_naive[1] + probs_naive[2])
        y_harsh_naive.append(probs_naive[2])
        y_mild_naive.append(probs_naive[1])
    
    label_fs = int(FINAL_LABEL_FONTSIZE * font_mult)
    tick_fs = int(FINAL_TICK_FONTSIZE * font_mult)
    
    if show_total:
        # P(Punishment) dashed
        ax.plot(w_values, y_total_rep, color='blue', linestyle='--', linewidth=FINAL_LINEWIDTH, label='Rep. P(Punish)')
        ax.plot(w_values, y_total_naive, color=NAIVE_COLOR, linestyle='--', linewidth=FINAL_LINEWIDTH, label='Naive P(Punish)')
        if show_ylabel:
            ax.set_ylabel("P(Punishment)", fontsize=label_fs)
    else:
        # P(Harsh) solid, P(Mild) dotted
        ax.plot(w_values, y_harsh_rep, color='blue', linestyle=LINESTYLE_HARSH, linewidth=FINAL_LINEWIDTH, label='Rep. Harsh')
        ax.plot(w_values, y_mild_rep, color='blue', linestyle=LINESTYLE_MILD, linewidth=FINAL_LINEWIDTH, label='Rep. Mild')
        ax.plot(w_values, y_harsh_naive, color=NAIVE_COLOR, linestyle=LINESTYLE_HARSH, linewidth=FINAL_LINEWIDTH, label='Naive Harsh')
        ax.plot(w_values, y_mild_naive, color=NAIVE_COLOR, linestyle=LINESTYLE_MILD, linewidth=FINAL_LINEWIDTH, label='Naive Mild')
        if show_ylabel:
            ax.set_ylabel("P(Action)", fontsize=label_fs)
    
    ax.set_xlabel("Authority's Belief (W)", fontsize=label_fs)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)


def run_subplot_trust_polarization_utility_final_with_insets(model_ctx, ax, agent_type, pol_values, auth_w, show_ylabel=True, inset_bottom=False, font_mult=1.0):
    """
    Subplot: Trust Polarization Utility - only reputational agent, all 3 actions.
    Includes inset zoom boxes for the extremes (0-0.1 and 0.9-1.0).
    
    Args:
        inset_bottom: If True, place left inset at bottom-left to avoid covering main lines.
        font_mult: Font size multiplier (default 1.0).
    """
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
    
    y_harsh, y_mild, y_none = [], [], []
    
    for p in pol_values:
        curr_j = get_interpolated_params(p, in_j_mean, out_j_mean, concentration=20.0)
        curr_b = get_interpolated_params(p, in_b_mean, out_b_mean, concentration=20.0)
        priors_out = (p_w_unc, curr_b, curr_j)
        
        utils_rep = get_utility_components(model_ctx, p_in_trust, priors_out, auth_w, true_b, true_j, weights_rep)
        
        y_harsh.append(utils_rep[2]['u_rep'])
        y_mild.append(utils_rep[1]['u_rep'])
        y_none.append(utils_rep[0]['u_rep'])
    
    # Convert to numpy arrays for easier slicing
    pol_values = np.array(pol_values)
    y_harsh = np.array(y_harsh)
    y_mild = np.array(y_mild)
    y_none = np.array(y_none)
    
    label_fs = int(FINAL_LABEL_FONTSIZE * font_mult)
    tick_fs = int(FINAL_TICK_FONTSIZE * font_mult)
    inset_tick_fs = int(FINAL_INSET_TICK_FONTSIZE * font_mult)
    
    # Main plot - using consistent line styles: Harsh=solid, Mild=dotted, None=dash-dot
    ax.plot(pol_values, y_harsh, color='blue', linestyle=LINESTYLE_HARSH, linewidth=FINAL_LINEWIDTH, label='Harsh')
    ax.plot(pol_values, y_mild, color='blue', linestyle=LINESTYLE_MILD, linewidth=FINAL_LINEWIDTH, label='Mild')
    ax.plot(pol_values, y_none, color='blue', linestyle=LINESTYLE_NONE, linewidth=FINAL_LINEWIDTH, label='None')
    
    ax.set_xlabel("Distrust Level", fontsize=label_fs)
    if show_ylabel:
        ax.set_ylabel("Reputational Utility", fontsize=label_fs)
    ax.set_xlim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=tick_fs)
    
    # Inset 1: Left side (0.0 to 0.1) - position depends on inset_bottom flag
    if inset_bottom:
        axins1 = ax.inset_axes([0.12, 0.08, 0.28, 0.38])  # Bottom-left position
    else:
        axins1 = ax.inset_axes([0.12, 0.52, 0.28, 0.38])  # Upper-left position
    
    # Filter data for left inset
    mask_left = pol_values <= 0.1
    if np.any(mask_left):
        axins1.plot(pol_values[mask_left], y_harsh[mask_left], color='blue', linestyle=LINESTYLE_HARSH, linewidth=FINAL_INSET_LINEWIDTH)
        axins1.plot(pol_values[mask_left], y_mild[mask_left], color='blue', linestyle=LINESTYLE_MILD, linewidth=FINAL_INSET_LINEWIDTH)
        axins1.plot(pol_values[mask_left], y_none[mask_left], color='blue', linestyle=LINESTYLE_NONE, linewidth=FINAL_INSET_LINEWIDTH)
        axins1.set_xlim(0.0, 0.1)
        # Auto-scale y based on data range
        all_y_left = np.concatenate([y_harsh[mask_left], y_mild[mask_left], y_none[mask_left]])
        y_margin = (all_y_left.max() - all_y_left.min()) * 0.1
        axins1.set_ylim(all_y_left.min() - y_margin, all_y_left.max() + y_margin)
        axins1.tick_params(axis='both', which='major', labelsize=inset_tick_fs)
        ax.indicate_inset_zoom(axins1, edgecolor="black", linewidth=1.5)
    
    # Inset 2: Right side (0.9 to 1.0)
    axins2 = ax.inset_axes([0.62, 0.52, 0.28, 0.38])
    
    # Filter data for right inset
    mask_right = pol_values >= 0.9
    if np.any(mask_right):
        axins2.plot(pol_values[mask_right], y_harsh[mask_right], color='blue', linestyle=LINESTYLE_HARSH, linewidth=FINAL_INSET_LINEWIDTH)
        axins2.plot(pol_values[mask_right], y_mild[mask_right], color='blue', linestyle=LINESTYLE_MILD, linewidth=FINAL_INSET_LINEWIDTH)
        axins2.plot(pol_values[mask_right], y_none[mask_right], color='blue', linestyle=LINESTYLE_NONE, linewidth=FINAL_INSET_LINEWIDTH)
        axins2.set_xlim(0.9, 1.0)
        # Auto-scale y based on data range
        all_y_right = np.concatenate([y_harsh[mask_right], y_mild[mask_right], y_none[mask_right]])
        y_margin = (all_y_right.max() - all_y_right.min()) * 0.1
        axins2.set_ylim(all_y_right.min() - y_margin, all_y_right.max() + y_margin)
        axins2.tick_params(axis='both', which='major', labelsize=inset_tick_fs)
        ax.indicate_inset_zoom(axins2, edgecolor="black", linewidth=1.5)


# Helper functions for Final Figure 3
def run_subplot_polarized_wrongness_naive_comm(model_ctx, ax, agent_type, w_values, show_ylabel=True):
    """
    Subplot: Polarized Wrongness, Uncertain Motives - naive and communicative agents only.
    Shows P(Harsh) and P(Mild) for both agents.
    """
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_j_unc = (1.0, 1.0)
    p_b_unc = (1.0, 1.0)
    p_in_w = get_beta_params(0.9, 20)
    p_out_w = get_beta_params(0.1, 20)
    priors_in = (p_in_w, p_b_unc, p_j_unc)
    priors_out = (p_out_w, p_b_unc, p_j_unc)
    
    # Naive agent (dark grey)
    weights_naive = get_agent_weights('naive')
    y_harsh_naive, y_mild_naive = [], []
    
    # Communicative agent (red)
    weights_comm = get_agent_weights('communicative')
    y_harsh_comm, y_mild_comm = [], []
    
    for w in w_values:
        probs_naive = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_naive)
        y_harsh_naive.append(probs_naive[2])
        y_mild_naive.append(probs_naive[1])
        
        probs_comm = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_comm)
        y_harsh_comm.append(probs_comm[2])
        y_mild_comm.append(probs_comm[1])
    
    # Plot naive (dark grey)
    ax.plot(w_values, y_harsh_naive, color=NAIVE_COLOR, linestyle='-', linewidth=FINAL_LINEWIDTH, label='Naive Harsh')
    ax.plot(w_values, y_mild_naive, color=NAIVE_COLOR, linestyle='--', linewidth=FINAL_LINEWIDTH, label='Naive Mild')
    
    # Plot communicative (red)
    ax.plot(w_values, y_harsh_comm, color='red', linestyle='-', linewidth=FINAL_LINEWIDTH, label='Comm. Harsh')
    ax.plot(w_values, y_mild_comm, color='red', linestyle='--', linewidth=FINAL_LINEWIDTH, label='Comm. Mild')
    
    ax.set_xlabel("Authority's Belief (W)", fontsize=FINAL_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("P(Action)", fontsize=FINAL_LABEL_FONTSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=FINAL_TICK_FONTSIZE)


def run_subplot_polarized_motives_naive_comm(model_ctx, ax, agent_type, w_values, show_ylabel=True):
    """
    Subplot: Polarized Motives, Uncertain Wrongness - naive and communicative agents only.
    Shows P(Harsh) and P(Mild) for both agents.
    """
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
    
    # Naive agent (dark grey)
    weights_naive = get_agent_weights('naive')
    y_harsh_naive, y_mild_naive = [], []
    
    # Communicative agent (red)
    weights_comm = get_agent_weights('communicative')
    y_harsh_comm, y_mild_comm = [], []
    
    for w in w_values:
        probs_naive = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_naive)
        y_harsh_naive.append(probs_naive[2])
        y_mild_naive.append(probs_naive[1])
        
        probs_comm = get_action_probs(model_ctx, priors_in, priors_out, w, true_b, true_j, weights_comm)
        y_harsh_comm.append(probs_comm[2])
        y_mild_comm.append(probs_comm[1])
    
    # Plot naive (dark grey)
    ax.plot(w_values, y_harsh_naive, color=NAIVE_COLOR, linestyle='-', linewidth=FINAL_LINEWIDTH, label='Naive Harsh')
    ax.plot(w_values, y_mild_naive, color=NAIVE_COLOR, linestyle='--', linewidth=FINAL_LINEWIDTH, label='Naive Mild')
    
    # Plot communicative (red)
    ax.plot(w_values, y_harsh_comm, color='red', linestyle='-', linewidth=FINAL_LINEWIDTH, label='Comm. Harsh')
    ax.plot(w_values, y_mild_comm, color='red', linestyle='--', linewidth=FINAL_LINEWIDTH, label='Comm. Mild')
    
    ax.set_xlabel("Authority's Belief (W)", fontsize=FINAL_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("P(Action)", fontsize=FINAL_LABEL_FONTSIZE)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis='both', which='major', labelsize=FINAL_TICK_FONTSIZE)


def run_subplot_wrongness_polarization_utility_comm(model_ctx, ax, agent_type, pol_values, auth_w, show_ylabel=True):
    """
    Subplot: Wrongness Polarization Utility - communicative agent only, all 3 actions.
    Shows utility for None, Mild, and Harsh actions.
    Uses consistent line styles: Harsh=solid, Mild=dotted, None=dash-dot.
    """
    true_j, true_b = agent_type['j'], agent_type['b']
    p_motives = ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
    
    weights_comm = get_agent_weights('communicative')
    
    y_harsh, y_mild, y_none = [], [], []
    
    for p in pol_values:
        in_mean = 0.5 + p * 0.45
        out_mean = 0.5 - p * 0.45
        p_in = (get_beta_params(in_mean, 20), p_motives[1], p_motives[2])
        p_out = (get_beta_params(out_mean, 20), p_motives[1], p_motives[2])
        
        utils_comm = get_utility_components(model_ctx, p_in, p_out, auth_w, true_b, true_j, weights_comm)
        
        y_harsh.append(utils_comm[2]['u_comm'])
        y_mild.append(utils_comm[1]['u_comm'])
        y_none.append(utils_comm[0]['u_comm'])
    
    ax.plot(pol_values, y_harsh, color='red', linestyle=LINESTYLE_HARSH, linewidth=FINAL_LINEWIDTH, label='Harsh')
    ax.plot(pol_values, y_mild, color='red', linestyle=LINESTYLE_MILD, linewidth=FINAL_LINEWIDTH, label='Mild')
    ax.plot(pol_values, y_none, color='red', linestyle=LINESTYLE_NONE, linewidth=FINAL_LINEWIDTH, label='None')
    
    ax.set_xlabel("Polarization Level", fontsize=FINAL_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("Communicative Utility", fontsize=FINAL_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FINAL_TICK_FONTSIZE)


def run_subplot_trust_polarization_utility_comm(model_ctx, ax, agent_type, pol_values, auth_w, show_ylabel=True):
    """
    Subplot: Trust Polarization Utility - communicative agent only, all 3 actions.
    Shows utility for None, Mild, and Harsh actions.
    Uses consistent line styles: Harsh=solid, Mild=dotted, None=dash-dot.
    """
    true_j, true_b = agent_type['j'], agent_type['b']
    
    p_w_unc = (1.0, 1.0)
    in_j_mean = 0.9
    in_b_mean = 0.5
    in_j_params = get_beta_params(in_j_mean, 20)
    in_b_params = get_beta_params(in_b_mean, 20)
    p_in_trust = (p_w_unc, in_b_params, in_j_params)
    
    out_j_mean = 0.1
    out_b_mean = 0.1 if true_b < 0 else 0.9
    
    weights_comm = get_agent_weights('communicative')
    
    y_harsh, y_mild, y_none = [], [], []
    
    for p in pol_values:
        curr_j = get_interpolated_params(p, in_j_mean, out_j_mean, concentration=20.0)
        curr_b = get_interpolated_params(p, in_b_mean, out_b_mean, concentration=20.0)
        priors_out = (p_w_unc, curr_b, curr_j)
        
        utils_comm = get_utility_components(model_ctx, p_in_trust, priors_out, auth_w, true_b, true_j, weights_comm)
        
        y_harsh.append(utils_comm[2]['u_comm'])
        y_mild.append(utils_comm[1]['u_comm'])
        y_none.append(utils_comm[0]['u_comm'])
    
    ax.plot(pol_values, y_harsh, color='red', linestyle=LINESTYLE_HARSH, linewidth=FINAL_LINEWIDTH, label='Harsh')
    ax.plot(pol_values, y_mild, color='red', linestyle=LINESTYLE_MILD, linewidth=FINAL_LINEWIDTH, label='Mild')
    ax.plot(pol_values, y_none, color='red', linestyle=LINESTYLE_NONE, linewidth=FINAL_LINEWIDTH, label='None')
    
    ax.set_xlabel("Distrust Level", fontsize=FINAL_LABEL_FONTSIZE)
    if show_ylabel:
        ax.set_ylabel("Communicative Utility", fontsize=FINAL_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=FINAL_TICK_FONTSIZE)


# -----------------------------------------------------------------------------
# FINAL PLOTS FIGURE FUNCTIONS
# -----------------------------------------------------------------------------

def run_final_figure_1(model_ctx, filename="final_fig1.png"):
    """
    Final Figure 1:
    
    Row 1 (4 plots): Polarized Wrongness (Uncertain Motives) - Reputational and Naive agents
        - X-axis: Authority's Belief (W)
        - Y-axis: P(Action) with extended range (-0.1 to 1.1)
        - 4 curves per plot: Rep Harsh/Mild (blue), Naive Harsh/Mild (dark grey)
        - Columns: High J Anti-B, Low J Anti-B, High J Pro-B, Low J Pro-B
    
    Row 2 (1 centered plot): Wrongness Polarization Utility - Reputational agent only
        - X-axis: Polarization Level
        - Y-axis: Reputational Utility
        - 3 blue curves: Harsh (solid), Mild (dashed), None (dotted)
        - Configuration: w=1.0, High Justice, Anti-Bias
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Final Figure 1")
    
    # Create figure with custom layout
    fig = plt.figure(figsize=(24, 12))
    
    # Row 1: 4 subplots
    axes_row1 = []
    for i in range(4):
        ax = fig.add_subplot(2, 4, i + 1)
        axes_row1.append(ax)
    
    # Row 2: 1 centered subplot spanning the middle 2 positions
    ax_row2 = fig.add_subplot(2, 4, (6, 7))
    
    fig.suptitle("Reputational Agent Behavior Analysis", fontsize=FINAL_SUPTITLE_FONTSIZE, fontweight='bold')
    
    w_values = np.linspace(0.0, 1.0, 100)
    pol_values = np.linspace(0.0, 1.0, 100)
    
    # Unique alphabetical labels: Row 1 = A-D, Row 2 = E
    row1_labels = ['A', 'B', 'C', 'D']
    
    # Row 1: Polarized Wrongness for each agent type (reordered)
    for col, agent_idx in enumerate(FINAL_AGENT_ORDER):
        agent_type = AGENT_TYPES[agent_idx]
        logger.info(f"  Processing Row 1, column {col+1}/4: {agent_type['label']}")
        ax = axes_row1[col]
        ax.set_title(f"{agent_type['label']}\nPolarized Wrongness (Uncertain Motives)", fontsize=FINAL_TITLE_FONTSIZE)
        show_ylabel = (col == 0)
        run_subplot_polarized_wrongness_final(model_ctx, ax, agent_type, w_values, show_ylabel=show_ylabel)
        if col == 0:
            ax.legend(loc='upper left', fontsize=FINAL_LEGEND_FONTSIZE)
        # Add unique alphabetical label
        ax.text(-0.12, 1.05, row1_labels[col], transform=ax.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    # Row 2: Wrongness Polarization Utility (High J, Anti-B, W=1.0)
    logger.info("  Processing Row 2: Wrongness Polarization Utility")
    agent_type_for_utility = AGENT_TYPES[0]  # High J, Anti-B
    ax_row2.set_title("Wrongness Polarization: Reputational Utility Components", fontsize=FINAL_TITLE_FONTSIZE)
    run_subplot_wrongness_polarization_utility_final(model_ctx, ax_row2, agent_type_for_utility, pol_values, auth_w=1.0)
    ax_row2.legend(loc='upper right', fontsize=FINAL_LEGEND_FONTSIZE)
    # Add unique alphabetical label
    ax_row2.text(-0.06, 1.05, 'E', transform=ax_row2.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Final Figure 1 saved to {filename} in {time.time() - figure_start:.2f}s")


def run_final_figure_2(model_ctx, filename="final_fig2.png"):
    """
    Final Figure 2:
    
    Row 1 (4 plots): Polarized Motives (Uncertain Wrongness) - Reputational and Naive agents
        - X-axis: Authority's Belief (W)
        - Y-axis: P(Punishment) = P(Harsh) + P(Mild)
        - 2 curves per plot: Reputational (blue), Naive (green)
        - Columns: High J Anti-B, Low J Anti-B, High J Pro-B, Low J Pro-B
    
    Row 2 (4 plots): Same as Row 1 but P(Harsh) and P(Mild) shown separately
        - X-axis: Authority's Belief (W)
        - Y-axis: P(Action)
        - 4 curves per plot: Rep Harsh/Mild (blue), Naive Harsh/Mild (green)
    
    Row 3 (2 centered plots): Trust Polarization Utility with inset zooms
        - X-axis: Distrust Level
        - Y-axis: Reputational Utility
        - 3 blue curves: Harsh (solid), Mild (dashed), None (dotted)
        - Plot 1: High J, Anti-B, W=1.0 (labelled "Anti-Bias")
        - Plot 2: High J, Pro-B, W=1.0 (labelled "Pro-Bias")
        - Each plot has zoom insets for 0.0-0.1 and 0.9-1.0
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Final Figure 2")
    
    # Create figure with gridspec for complex layout
    fig = plt.figure(figsize=(24, 18))
    
    # Row 1: 4 subplots (positions 1-4)
    axes_row1 = []
    for i in range(4):
        ax = fig.add_subplot(3, 4, i + 1)
        axes_row1.append(ax)
    
    # Row 2: 4 subplots (positions 5-8)
    axes_row2 = []
    for i in range(4):
        ax = fig.add_subplot(3, 4, i + 5)
        axes_row2.append(ax)
    
    # Row 3: 2 centered subplots (positions 10 and 11)
    ax_row3_left = fig.add_subplot(3, 4, 10)
    ax_row3_right = fig.add_subplot(3, 4, 11)
    
    fig.suptitle("Reputational Agent: Punishment Probabilities and Utility Analysis", fontsize=FINAL_SUPTITLE_FONTSIZE, fontweight='bold')
    
    w_values = np.linspace(0.0, 1.0, 100)
    pol_values = np.linspace(0.0, 1.0, 200)  # More points for smoother inset zooms
    
    # Unique alphabetical labels: Row 1 = A-D, Row 2 = E-H, Row 3 = I-J
    row1_labels = ['A', 'B', 'C', 'D']
    row2_labels = ['E', 'F', 'G', 'H']
    
    # Row 1: Polarized Motives - Total Punishment (reordered)
    for col, agent_idx in enumerate(FINAL_AGENT_ORDER):
        agent_type = AGENT_TYPES[agent_idx]
        logger.info(f"  Processing Row 1, column {col+1}/4: {agent_type['label']}")
        ax = axes_row1[col]
        ax.set_title(f"{agent_type['label']}\nPolarized Motives (Uncertain Wrongness)", fontsize=FINAL_TITLE_FONTSIZE)
        show_ylabel = (col == 0)
        run_subplot_polarized_motives_final(model_ctx, ax, agent_type, w_values, show_separate=False, show_ylabel=show_ylabel)
        if col == 0:
            ax.legend(loc='upper left', fontsize=FINAL_LEGEND_FONTSIZE)
        # Add unique alphabetical label
        ax.text(-0.12, 1.05, row1_labels[col], transform=ax.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    # Row 2: Polarized Motives - Harsh and Mild separate (reordered)
    for col, agent_idx in enumerate(FINAL_AGENT_ORDER):
        agent_type = AGENT_TYPES[agent_idx]
        logger.info(f"  Processing Row 2, column {col+1}/4: {agent_type['label']}")
        ax = axes_row2[col]
        ax.set_title(f"P(Harsh) and P(Mild) Separately", fontsize=FINAL_TITLE_FONTSIZE)
        show_ylabel = (col == 0)
        run_subplot_polarized_motives_final(model_ctx, ax, agent_type, w_values, show_separate=True, show_ylabel=show_ylabel)
        if col == 0:
            ax.legend(loc='upper left', fontsize=FINAL_LEGEND_FONTSIZE)
        # Add unique alphabetical label
        ax.text(-0.12, 1.05, row2_labels[col], transform=ax.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    # Row 3: Trust Polarization Utility with insets
    # Left plot: High J, Anti-B (AGENT_TYPES[0])
    logger.info("  Processing Row 3, left: Anti-Bias")
    agent_type_anti = AGENT_TYPES[0]  # High J, Anti-B
    ax_row3_left.set_title("Anti-Bias", fontsize=FINAL_TITLE_FONTSIZE + 2, fontweight='bold')
    run_subplot_trust_polarization_utility_final_with_insets(model_ctx, ax_row3_left, agent_type_anti, pol_values, auth_w=1.0)
    ax_row3_left.legend(loc='lower left', fontsize=FINAL_LEGEND_FONTSIZE)
    # Add unique alphabetical label
    ax_row3_left.text(-0.12, 1.05, 'I', transform=ax_row3_left.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    # Right plot: High J, Pro-B (AGENT_TYPES[1])
    logger.info("  Processing Row 3, right: Pro-Bias")
    agent_type_pro = AGENT_TYPES[1]  # High J, Pro-B
    ax_row3_right.set_title("Pro-Bias", fontsize=FINAL_TITLE_FONTSIZE + 2, fontweight='bold')
    run_subplot_trust_polarization_utility_final_with_insets(model_ctx, ax_row3_right, agent_type_pro, pol_values, auth_w=1.0, show_ylabel=False)
    ax_row3_right.legend(loc='lower left', fontsize=FINAL_LEGEND_FONTSIZE)
    # Add unique alphabetical label
    ax_row3_right.text(-0.12, 1.05, 'J', transform=ax_row3_right.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.94)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Final Figure 2 saved to {filename} in {time.time() - figure_start:.2f}s")


def run_final_figure_1_alt(model_ctx, filename="final_fig1_alt.png"):
    """
    Alternative Final Figure 1: Single row layout (1 row x 5 columns)
    
    Columns 1-4: Polarized Wrongness plots for each agent type (labelled A-D)
    Column 5: Wrongness Polarization Utility plot (labelled E)
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Final Figure 1 (Alternative)")
    
    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    
    fig.suptitle("Reputational Agent Behavior Analysis", fontsize=FINAL_SUPTITLE_FONTSIZE, fontweight='bold', y=0.98)
    
    w_values = np.linspace(0.0, 1.0, 100)
    pol_values = np.linspace(0.0, 1.0, 100)
    
    subplot_labels = ['A', 'B', 'C', 'D', 'E']
    
    # Columns 1-4: Polarized Wrongness for each agent type (reordered)
    for col, agent_idx in enumerate(FINAL_AGENT_ORDER):
        agent_type = AGENT_TYPES[agent_idx]
        logger.info(f"  Processing column {col+1}/5: {agent_type['label']}")
        ax = axes[col]
        ax.set_title(f"{agent_type['label']}\nPolarized Wrongness", fontsize=FINAL_TITLE_FONTSIZE)
        show_ylabel = (col == 0)
        run_subplot_polarized_wrongness_final(model_ctx, ax, agent_type, w_values, show_ylabel=show_ylabel)
        if col == 0:
            ax.legend(loc='upper left', fontsize=FINAL_LEGEND_FONTSIZE)
        # Add unique alphabetical label
        ax.text(-0.12, 1.05, subplot_labels[col], transform=ax.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    # Column 5: Wrongness Polarization Utility (High J, Anti-B, W=1.0)
    logger.info("  Processing column 5/5: Utility")
    agent_type_for_utility = AGENT_TYPES[0]  # High J, Anti-B
    ax = axes[4]
    ax.set_title("Wrongness Polarization\nReputational Utility", fontsize=FINAL_TITLE_FONTSIZE)
    run_subplot_wrongness_polarization_utility_final(model_ctx, ax, agent_type_for_utility, pol_values, auth_w=1.0, show_ylabel=True)
    ax.legend(loc='upper right', fontsize=FINAL_LEGEND_FONTSIZE)
    # Add unique alphabetical label
    ax.text(-0.12, 1.05, subplot_labels[4], transform=ax.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Final Figure 1 (Alternative) saved to {filename} in {time.time() - figure_start:.2f}s")


def run_final_figure_2_alt(model_ctx, filename="final_fig2_alt.png"):
    """
    Alternative Final Figure 2: 2 rows x 5 columns layout (1.25x larger fonts)
    
    Columns 1-4: Polarized Motives plots
        Row 1: P(Punishment) dashed (labelled A-D)
        Row 2: P(Harsh) solid, P(Mild) dotted (labelled F-I)
    Column 5: Trust Polarization Utility
        Row 1: Anti-Bias (labelled E)
        Row 2: Pro-Bias (labelled J)
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Final Figure 2 (Alternative)")
    
    # Use 1.25x larger fonts for this figure
    fm = FIG2_ALT_FONT_MULTIPLIER
    title_fs = int(FINAL_TITLE_FONTSIZE * fm)
    suptitle_fs = int(FINAL_SUPTITLE_FONTSIZE * fm)
    legend_fs = int(FINAL_LEGEND_FONTSIZE * fm)
    label_fs = int(FINAL_SUBPLOT_LABEL_FONTSIZE * fm)
    
    fig, axes = plt.subplots(2, 5, figsize=(30, 12))
    
    fig.suptitle("Reputational Agent: Punishment Probabilities and Utility Analysis", fontsize=suptitle_fs, fontweight='bold', y=0.98)
    
    w_values = np.linspace(0.0, 1.0, 100)
    pol_values = np.linspace(0.0, 1.0, 200)
    
    # Alphabetical labels: Row 1 = A-E, Row 2 = F-J
    # Layout: Row 1 cols 0-3 = A-D, col 4 = E; Row 2 cols 0-3 = F-I, col 4 = J
    row1_labels = ['A', 'B', 'C', 'D', 'E']
    row2_labels = ['F', 'G', 'H', 'I', 'J']
    
    # Row 1: P(Punishment) dashed for all 4 agent types
    for col, agent_idx in enumerate(FINAL_AGENT_ORDER):
        agent_type = AGENT_TYPES[agent_idx]
        logger.info(f"  Row 1, column {col+1}/5: {agent_type['label']}")
        ax = axes[0, col]
        ax.set_title(f"{agent_type['label']}\nPolarized Motives", fontsize=title_fs)
        show_ylabel = (col == 0)
        run_subplot_polarized_motives_final_v2(model_ctx, ax, agent_type, w_values, show_total=True, show_ylabel=show_ylabel, font_mult=fm)
        if col == 0:
            ax.legend(loc='upper left', fontsize=legend_fs)
        # Add unique alphabetical label
        ax.text(-0.12, 1.05, row1_labels[col], transform=ax.transAxes, fontsize=label_fs, fontweight='bold', va='bottom')
    
    # Row 1, Column 5: Anti-Bias Utility
    logger.info("  Row 1, column 5/5: Anti-Bias Utility")
    ax = axes[0, 4]
    ax.set_title("Anti-Bias\nTrust Polarization Utility", fontsize=title_fs)
    run_subplot_trust_polarization_utility_final_with_insets(model_ctx, ax, AGENT_TYPES[0], pol_values, auth_w=1.0, show_ylabel=True, inset_bottom=True, font_mult=fm)
    ax.legend(loc='lower right', fontsize=legend_fs - 1)
    # Add unique alphabetical label
    ax.text(-0.12, 1.05, row1_labels[4], transform=ax.transAxes, fontsize=label_fs, fontweight='bold', va='bottom')
    
    # Row 2: P(Harsh) solid, P(Mild) dotted for all 4 agent types
    for col, agent_idx in enumerate(FINAL_AGENT_ORDER):
        agent_type = AGENT_TYPES[agent_idx]
        logger.info(f"  Row 2, column {col+1}/5: {agent_type['label']}")
        ax = axes[1, col]
        ax.set_title(f"P(Harsh) and P(Mild)", fontsize=title_fs)
        show_ylabel = (col == 0)
        run_subplot_polarized_motives_final_v2(model_ctx, ax, agent_type, w_values, show_total=False, show_ylabel=show_ylabel, font_mult=fm)
        if col == 0:
            ax.legend(loc='upper left', fontsize=legend_fs)
        # Add unique alphabetical label
        ax.text(-0.12, 1.05, row2_labels[col], transform=ax.transAxes, fontsize=label_fs, fontweight='bold', va='bottom')
    
    # Row 2, Column 5: Pro-Bias Utility
    logger.info("  Row 2, column 5/5: Pro-Bias Utility")
    ax = axes[1, 4]
    ax.set_title("Pro-Bias\nTrust Polarization Utility", fontsize=title_fs)
    run_subplot_trust_polarization_utility_final_with_insets(model_ctx, ax, AGENT_TYPES[1], pol_values, auth_w=1.0, show_ylabel=True, inset_bottom=True, font_mult=fm)
    ax.legend(loc='lower right', fontsize=legend_fs - 1)
    # Add unique alphabetical label
    ax.text(-0.12, 1.05, row2_labels[4], transform=ax.transAxes, fontsize=label_fs, fontweight='bold', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Final Figure 2 (Alternative) saved to {filename} in {time.time() - figure_start:.2f}s")


def run_final_figure_3(model_ctx, filename="final_fig3.png"):
    """
    Final Figure 3: Communicative agent analysis
    
    Layout: 2 rows x 4 columns with whitespace
    
    Row 1: [Polarized Wrongness Anti-B (A)] [Polarized Wrongness Pro-B (B)] [empty] [Wrongness Utility (C)]
    Row 2: [Polarized Motives Anti-B (D)] [Polarized Motives Pro-B (E)] [Trust Utility Anti-B (F)] [Trust Utility Pro-B (G)]
    
    Part A (columns 0-1): Policy plots with Naive (dark grey) and Communicative (red)
    Part B (columns 2-3): Utility plots for Communicative agent only
    
    Top utility subplot is same size as others, centered with whitespace in column 2.
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Final Figure 3")
    
    # Use GridSpec for complex layout
    from matplotlib.gridspec import GridSpec
    
    fig = plt.figure(figsize=(28, 14))
    gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 1])
    
    w_values = np.linspace(0.0, 1.0, 100)
    pol_values = np.linspace(0.0, 1.0, 100)
    
    fig.suptitle("Communicative Agent Analysis", fontsize=FINAL_SUPTITLE_FONTSIZE, fontweight='bold')
    
    # High Justice agent types: High J Anti-B (index 0), High J Pro-B (index 1)
    high_j_agents = [AGENT_TYPES[0], AGENT_TYPES[1]]
    
    # Unique alphabetical labels for all 7 subplots
    # Row 1: A, B, (empty), C
    # Row 2: D, E, F, G
    
    # Part A - Row 1: Polarized Wrongness (Naive + Communicative)
    row1_policy_labels = ['A', 'B']
    for col, agent_type in enumerate(high_j_agents):
        logger.info(f"  Part A Row 1, column {col+1}/2: {agent_type['label']}")
        ax = fig.add_subplot(gs[0, col])
        ax.set_title(f"{agent_type['label']}\nPolarized Wrongness", fontsize=FINAL_TITLE_FONTSIZE)
        show_ylabel = (col == 0)
        run_subplot_polarized_wrongness_naive_comm(model_ctx, ax, agent_type, w_values, show_ylabel=show_ylabel)
        if col == 0:
            ax.legend(loc='upper left', fontsize=FINAL_LEGEND_FONTSIZE)
        # Add unique alphabetical label
        ax.text(-0.12, 1.05, row1_policy_labels[col], transform=ax.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    # Part A - Row 2: Polarized Motives (Naive + Communicative)
    row2_policy_labels = ['D', 'E']
    for col, agent_type in enumerate(high_j_agents):
        logger.info(f"  Part A Row 2, column {col+1}/2: {agent_type['label']}")
        ax = fig.add_subplot(gs[1, col])
        ax.set_title(f"Polarized Motives", fontsize=FINAL_TITLE_FONTSIZE)
        show_ylabel = (col == 0)
        run_subplot_polarized_motives_naive_comm(model_ctx, ax, agent_type, w_values, show_ylabel=show_ylabel)
        if col == 0:
            ax.legend(loc='upper left', fontsize=FINAL_LEGEND_FONTSIZE)
        # Add unique alphabetical label
        ax.text(-0.12, 1.05, row2_policy_labels[col], transform=ax.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    # Part B - Top: Wrongness Polarization Utility (Communicative only, W=0.5, High J Anti-B)
    # Centered over columns 2-3 (left side of the whitespace area)
    logger.info("  Part B Top: Wrongness Polarization Utility")
    ax_b_top = fig.add_subplot(gs[0, 2])  # Column 2, leaving column 3 as whitespace (centered/left of whitespace)
    ax_b_top.set_title("Wrongness Polarization\nCommunicative Utility", fontsize=FINAL_TITLE_FONTSIZE)
    run_subplot_wrongness_polarization_utility_comm(model_ctx, ax_b_top, AGENT_TYPES[0], pol_values, auth_w=0.5, show_ylabel=True)
    ax_b_top.legend(loc='upper right', fontsize=FINAL_LEGEND_FONTSIZE)
    # Add unique alphabetical label
    ax_b_top.text(-0.12, 1.05, 'C', transform=ax_b_top.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    # Part B - Bottom: Trust Polarization Utility (Communicative only, W=0.5)
    # Two separate plots: High J Anti-B and High J Pro-B
    logger.info("  Part B Bottom-Left: Anti-Bias Trust Utility")
    ax_b_bl = fig.add_subplot(gs[1, 2])
    ax_b_bl.set_title("Anti-Bias\nTrust Polarization", fontsize=FINAL_TITLE_FONTSIZE)
    run_subplot_trust_polarization_utility_comm(model_ctx, ax_b_bl, AGENT_TYPES[0], pol_values, auth_w=0.5, show_ylabel=True)
    ax_b_bl.legend(loc='lower left', fontsize=FINAL_LEGEND_FONTSIZE)
    # Add unique alphabetical label
    ax_b_bl.text(-0.12, 1.05, 'F', transform=ax_b_bl.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    logger.info("  Part B Bottom-Right: Pro-Bias Trust Utility")
    ax_b_br = fig.add_subplot(gs[1, 3])
    ax_b_br.set_title("Pro-Bias\nTrust Polarization", fontsize=FINAL_TITLE_FONTSIZE)
    run_subplot_trust_polarization_utility_comm(model_ctx, ax_b_br, AGENT_TYPES[1], pol_values, auth_w=0.5, show_ylabel=False)
    ax_b_br.legend(loc='lower left', fontsize=FINAL_LEGEND_FONTSIZE)
    # Add unique alphabetical label
    ax_b_br.text(-0.12, 1.05, 'G', transform=ax_b_br.transAxes, fontsize=FINAL_SUBPLOT_LABEL_FONTSIZE, fontweight='bold', va='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Final Figure 3 saved to {filename} in {time.time() - figure_start:.2f}s")


# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def main(log_dir="logs", utility_mode=False, punishment_mode='mild', save_dir=None, belief_update=False, w1_mode=False, final_plots=False):
    """Run all figure simulations with logging.
    
    Args:
        log_dir: Directory to store log files (default: "logs")
        utility_mode: If True, generate utility component plots instead of probability plots
        punishment_mode: For probability plots, controls what the dashed line shows:
            'mild' - P(Mild), 'none' - P(None), 'total-punishment' - only P(Mild)+P(Harsh)
            For utility plots with 'all': shows 9 lines (all 3 actions)
        save_dir: Optional directory to save figures. If None, saves to current directory.
        belief_update: If True, generate belief update visualization figure.
        w1_mode: If True, generate Wasserstein-1 analysis figure.
        final_plots: If True, generate the final publication-ready figures.
    """
    global logger
    logger = setup_logging(log_dir=log_dir)
    
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("SIMULATION RUN STARTED")
    logger.info(f"Device: {jax.devices()[0].platform} ({jax.devices()})")
    mode_str = 'FINAL PLOTS' if final_plots else ('W-1' if w1_mode else ('BELIEF UPDATE' if belief_update else ('UTILITY' if utility_mode else 'PROBABILITY')))
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Punishment Mode: {punishment_mode}")
    if save_dir:
        logger.info(f"Save Directory: {save_dir}")
        os.makedirs(save_dir, exist_ok=True)
    logger.info("=" * 60)
    
    logger.info("Building model context...")
    model_start = time.time()
    config = {'GRID_SIZE': 400, 'BETA_NAIVE': 10.0}
    logger.info(f"Config: {config}")
    model_ctx = build_model_context(config)
    logger.info(f"Model context built in {time.time() - model_start:.2f}s")
    
    def get_save_path(fname):
        if save_dir:
            return os.path.join(save_dir, fname)
        return fname

    if final_plots:
        # Generate the final publication-ready figures
        logger.info("Running final plots simulations...")
        logger.info("[Figure 1/5] (Final Plots Mode)")
        run_final_figure_1(model_ctx, get_save_path("final_fig1.png"))
        logger.info("[Figure 2/5] (Final Plots Mode)")
        run_final_figure_2(model_ctx, get_save_path("final_fig2.png"))
        logger.info("[Figure 3/5] (Final Plots Mode)")
        run_final_figure_3(model_ctx, get_save_path("final_fig3.png"))
        logger.info("[Figure 4/5] (Final Plots Mode - Alternative)")
        run_final_figure_1_alt(model_ctx, get_save_path("final_fig1_alt.png"))
        logger.info("[Figure 5/5] (Final Plots Mode - Alternative)")
        run_final_figure_2_alt(model_ctx, get_save_path("final_fig2_alt.png"))
    elif w1_mode:
        # Generate only the Wasserstein-1 analysis figure
        logger.info("Running W-1 figure simulation...")
        logger.info("[Figure 1/1] (W-1 Mode)")
        run_figure_w1(model_ctx, get_save_path("fig_w1.png"))
    elif belief_update:
        # Generate only the belief update figure
        logger.info("Running belief update figure simulation...")
        logger.info("[Figure 1/1] (Belief Update Mode)")
        run_figure_belief_update(model_ctx, get_save_path("fig_belief_update.png"))
    elif utility_mode:
        # Run all three utility figures
        logger.info("Running 3 figure simulations...")
        
        logger.info(f"[Figure 1/3] (Utility Mode, punishment_mode={punishment_mode})")
        run_figure_1_utility(model_ctx, get_save_path("fig1_utility_polarized_beliefs.png"), punishment_mode=punishment_mode)
        
        logger.info(f"[Figure 2/3] (Utility Mode, punishment_mode={punishment_mode})")
        run_figure_2_utility(model_ctx, get_save_path("fig2_utility_wrongness_polarization.png"), punishment_mode=punishment_mode)
        
        logger.info(f"[Figure 3/3] (Utility Mode, punishment_mode={punishment_mode})")
        run_figure_3_utility(model_ctx, get_save_path("fig3_utility_trust_polarization.png"), punishment_mode=punishment_mode)
    else:
        # Run all three probability figures
        logger.info("Running 3 figure simulations...")
        
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
        "--device",
        type=str,
        choices=['cpu', 'gpu'],
        default='cpu',
        help="Device to use for computation: 'cpu' or 'gpu' (default: cpu). "
             "Note: Device is configured early before imports."
    )
    parser.add_argument(
        "--utility-mode",
        action="store_true",
        default=False,
        help="Generate utility component plots instead of probability plots"
    )
    parser.add_argument(
        "--belief-update",
        action="store_true",
        default=False,
        help="Generate belief update visualization showing action probabilities and "
             "how observers update their beliefs about wrongness after observing each action type. "
             "Creates a 4-row figure: Row 1 shows all action probs (9 lines), "
             "Rows 2-4 show prior/posterior P(W) for None/Mild/Harsh actions."
    )
    parser.add_argument(
        "--w1",
        action="store_true",
        default=False,
        help="Generate Wasserstein-1 analysis figure showing how W-1 distance between "
             "observer posteriors and authority's point belief changes across different actions. "
             "Creates a 5-row figure: Row 1 shows W-1 for in-group (high-W prior), "
             "Row 2 shows W-1 for out-group (low-W prior), "
             "Rows 3-5 show prior/posterior P(W) for None/Mild/Harsh actions."
    )
    parser.add_argument(
        "--final-plots",
        action="store_true",
        default=False,
        help="Generate final publication-ready figures. "
             "Figure 1: Row 1 shows polarized wrongness (uncertain motives) for all 4 agent types "
             "with only reputational agent (2 curves: Harsh/Mild). Row 2 shows wrongness polarization "
             "utility for reputational agent (3 curves). "
             "Figure 2: Row 1 shows polarized motives with P(Punishment), Row 2 shows P(Harsh)/P(Mild) separately, "
             "Row 3 shows trust polarization utility with zoom insets for Anti-Bias and Pro-Bias agents."
    )
    parser.add_argument(
        "--punishment",
        type=str,
        choices=['mild', 'none', 'total-punishment', 'all'],
        default='mild',
        help="For probability plots: 'mild' shows P(Mild) as dashed, 'none' shows P(None) as dashed, "
             "'total-punishment' shows only P(Mild)+P(Harsh) as solid. "
             "'all' (requires --utility-mode) shows utilities for all 3 actions (9 lines). (default: mild)"
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
    
    # Validate that 'all' punishment mode requires utility-mode
    if args.punishment == 'all' and not args.utility_mode:
        parser.error("--punishment all requires --utility-mode to be enabled")
    
    # Validate mutually exclusive modes
    special_modes = sum([args.belief_update, args.utility_mode, args.w1, args.final_plots])
    if special_modes > 1:
        parser.error("--belief-update, --utility-mode, --w1, and --final-plots are mutually exclusive")
    
    main(log_dir=args.log_dir, utility_mode=args.utility_mode, punishment_mode=args.punishment, 
         save_dir=args.save_dir, belief_update=args.belief_update, w1_mode=args.w1, final_plots=args.final_plots)