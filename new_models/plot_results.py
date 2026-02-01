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
    pol_values = np.linspace(0.0, 1.0, 100)
    
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
    pol_values = np.linspace(0.0, 1.0, 100)
    
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
    Row 2: Authority believes NOT WRONG (W=0.0)
    Columns: 4 agent types
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting Figure 2 (Utility): Wrongness Polarization")
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
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
        
        # Row 2: Auth believes NOT WRONG (W=0.0)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_wrongness_polarization_utility(model_ctx, ax2, agent_type, pol_values, auth_w=0.0, punishment_mode=punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(filename, dpi=150)
    plt.close(fig)
    logger.info(f"Figure 2 (Utility) saved to {filename} in {time.time() - figure_start:.2f}s")

def run_figure_3_utility(model_ctx, filename="fig3_utility_trust_polarization.png", punishment_mode='none'):
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
        
        # Row 2: Auth believes NOT WRONG (W=0.0)
        ax2 = axes[1, col]
        ax2.set_title(f"Authority: W=0.0 (Not Wrong)", fontsize=10)
        run_subplot_trust_polarization_utility(model_ctx, ax2, agent_type, pol_values, auth_w=0.0, punishment_mode=punishment_mode)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
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
# JSD FIGURE FUNCTIONS
# -----------------------------------------------------------------------------

def compute_jsd(p, q):
    """Compute Jensen-Shannon Divergence between two distributions p and q."""
    # Ensure distributions are normalized
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    
    m = 0.5 * (p + q)
    
    # KL(P || M) with numerical stability
    kl_pm = np.sum(p * np.log((p + 1e-10) / (m + 1e-10)))
    # KL(Q || M)
    kl_qm = np.sum(q * np.log((q + 1e-10) / (m + 1e-10)))
    
    return 0.5 * kl_pm + 0.5 * kl_qm


def make_authority_belief_dist_numpy(true_w, W_GRID):
    """
    Constructs a sharp probability distribution Q centered on the authority's
    true_w scalar. NumPy version for plotting.
    """
    from scipy.stats import beta as scipy_beta
    
    concentration = 100.0
    w_safe = np.clip(true_w, 0.001, 0.999)
    
    alpha = w_safe * concentration + 1.0
    beta_param = (1.0 - w_safe) * concentration + 1.0
    
    dist = scipy_beta.pdf(W_GRID, alpha, beta_param)
    return dist / (np.sum(dist) + 1e-10)


def run_subplot_jsd_vs_authority_belief(model_ctx, ax, agent_type, group_type, w_values):
    """
    Subplot: JSD between group's posterior P(W) and authority's belief vs authority's W.
    
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
    
    # For each action, compute JSD at each authority W value
    for action_idx in [0, 1, 2]:
        jsd_values = []
        
        # Compute posterior after observing this action
        posterior_tensor = model_ctx.observer_update(prior_tensor, action_idx)
        posterior_w = np.array(jnp.sum(posterior_tensor, axis=(1, 2)))
        
        for auth_w in w_values:
            # Get authority's belief distribution (delta-like centered at auth_w)
            authority_dist = make_authority_belief_dist_numpy(auth_w, W_GRID)
            
            # Compute JSD between posterior and authority belief
            jsd_val = compute_jsd(posterior_w, authority_dist)
            jsd_values.append(jsd_val)
        
        if group_type == 'in':
            color = 'blue'
        elif group_type == 'out':
            color = 'red'
            
        ax.plot(w_values, jsd_values, color=color, linestyle=action_styles[action_idx],
                linewidth=2, label=f"After {action_names[action_idx]}")
    
    ax.set_xlabel("Authority's Belief (W)")
    ax.set_ylabel("JSD(Posterior, Authority)")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)


def run_figure_jsd(model_ctx, filename="fig_jsd.png"):
    """
    Figure: JSD Visualization
    
    Row 1: JSD between in-group posterior and authority belief vs authority W
           (3 blue lines: None, Mild, Harsh actions)
    Row 2: JSD between out-group posterior and authority belief vs authority W
           (3 blue lines: None, Mild, Harsh actions)
    Row 3: Belief update P(W) after observing None action
    Row 4: Belief update P(W) after observing Mild action  
    Row 5: Belief update P(W) after observing Harsh action
    
    Columns: 4 agent types (High J Anti-B, High J Pro-B, Low J Anti-B, Low J Pro-B)
    """
    figure_start = time.time()
    logger.info("=" * 60)
    logger.info("Starting JSD Figure")
    
    fig, axes = plt.subplots(5, 4, figsize=(20, 25))
    fig.suptitle("JSD Analysis: Posterior vs Authority Belief\n(Polarized Wrongness, Uncertain Motives)", 
                 fontsize=16, fontweight='bold')
    
    w_values = np.linspace(0.0, 1.0, 100)
    action_names = {0: 'None', 1: 'Mild', 2: 'Harsh'}
    
    for col, agent_type in enumerate(AGENT_TYPES):
        logger.info(f"  Processing column {col+1}/4: {agent_type['label']}")
        
        # Row 1: JSD for in-group (high-W prior)
        ax1 = axes[0, col]
        ax1.set_title(f"{agent_type['label']}\nIn-Group JSD (High-W Prior)", fontsize=10)
        run_subplot_jsd_vs_authority_belief(model_ctx, ax1, agent_type, 'in', w_values)
        if col == 0:
            ax1.legend(loc='upper right', fontsize=8)
        
        # Row 2: JSD for out-group (low-W prior)
        ax2 = axes[1, col]
        ax2.set_title(f"Out-Group JSD (Low-W Prior)", fontsize=10)
        run_subplot_jsd_vs_authority_belief(model_ctx, ax2, agent_type, 'out', w_values)
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
    logger.info(f"JSD Figure saved to {filename} in {time.time() - figure_start:.2f}s")

# -----------------------------------------------------------------------------
# MAIN ENTRY POINT
# -----------------------------------------------------------------------------

def main(log_dir="logs", utility_mode=False, punishment_mode='mild', save_dir=None, belief_update=False, jsd_mode=False):
    """Run all figure simulations with logging.
    
    Args:
        log_dir: Directory to store log files (default: "logs")
        utility_mode: If True, generate utility component plots instead of probability plots
        punishment_mode: For probability plots, controls what the dashed line shows:
            'mild' - P(Mild), 'none' - P(None), 'total-punishment' - only P(Mild)+P(Harsh)
            For utility plots with 'all': shows 9 lines (all 3 actions)
        save_dir: Optional directory to save figures. If None, saves to current directory.
        belief_update: If True, generate belief update visualization figure.
        jsd_mode: If True, generate JSD analysis figure.
    """
    global logger
    logger = setup_logging(log_dir=log_dir)
    
    total_start = time.time()
    logger.info("=" * 60)
    logger.info("SIMULATION RUN STARTED")
    logger.info(f"Device: {jax.devices()[0].platform} ({jax.devices()})")
    mode_str = 'JSD' if jsd_mode else ('BELIEF UPDATE' if belief_update else ('UTILITY' if utility_mode else 'PROBABILITY'))
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

    if jsd_mode:
        # Generate only the JSD analysis figure
        logger.info("Running JSD figure simulation...")
        logger.info("[Figure 1/1] (JSD Mode)")
        run_figure_jsd(model_ctx, get_save_path("fig_jsd.png"))
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
        "--jsd",
        action="store_true",
        default=False,
        help="Generate JSD analysis figure showing how Jensen-Shannon Divergence between "
             "observer posteriors and authority belief changes across different actions. "
             "Creates a 5-row figure: Row 1 shows JSD for in-group (high-W prior), "
             "Row 2 shows JSD for out-group (low-W prior), "
             "Rows 3-5 show prior/posterior P(W) for None/Mild/Harsh actions."
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
    special_modes = sum([args.belief_update, args.utility_mode, args.jsd])
    if special_modes > 1:
        parser.error("--belief-update, --utility-mode, and --jsd are mutually exclusive")
    
    main(log_dir=args.log_dir, utility_mode=args.utility_mode, punishment_mode=args.punishment, 
         save_dir=args.save_dir, belief_update=args.belief_update, jsd_mode=args.jsd)