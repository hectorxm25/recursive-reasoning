#!/usr/bin/env python3
"""
Master runner script that parses config.yaml and runs experiments.
"""

import argparse
import os
import yaml
import numpy as np

from experiments import run_conflict_sweep, ALL_SWEEPABLE_VARIABLES


def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config):
    """Validate the configuration parameters."""
    # Assert that sweep_values has exactly 3 values
    assert len(config['sweep_values']) == 3, \
        f"sweep_values must have exactly 3 values, got {len(config['sweep_values'])}"
    
    # Validate required keys exist
    required_keys = [
        'GRID_SIZE', 'BETA_NAIVE',
        'w_J_in', 'w_J_out', 
        'w_B_in_dir', 'w_B_out_dir',   # Directional bias weights
        'w_B_in_neu', 'w_B_out_neu',   # Neutrality bias weights
        'scale_int', 'scale_rep', 'beta_strat',
        'true_state_w', 'true_state_b', 'true_state_j',
        'rounds', 'sweep_values', 'sweep_variable', 'plot_figure_save_path',
        'prior_in_w_alpha', 'prior_in_w_beta',
        'prior_in_b_alpha', 'prior_in_b_beta',
        'prior_in_j_alpha', 'prior_in_j_beta',
        'prior_out_w_alpha', 'prior_out_w_beta',
        'prior_out_b_alpha', 'prior_out_b_beta',
        'prior_out_j_alpha', 'prior_out_j_beta',
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    # Validate sweep_variable is one of the allowed values
    sweep_variable = config['sweep_variable']
    assert sweep_variable in ALL_SWEEPABLE_VARIABLES, \
        f"sweep_variable must be one of {ALL_SWEEPABLE_VARIABLES}, got '{sweep_variable}'"
    
    return True


def set_random_seed(config):
    """Set the random seed if specified in config."""
    seed = config.get('random_seed', None)
    if seed is not None:
        np.random.seed(seed)
        print(f"  Random seed: {seed}")
        return seed
    else:
        print("  Random seed: None (non-deterministic)")
        return None


def main():
    parser = argparse.ArgumentParser(
        description='Run experiment sweeps with configuration from YAML file.'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default='config.yaml',
        help='Path to the configuration YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--debug', '-d',
        action='store_true',
        help='Enable debug output'
    )
    
    args = parser.parse_args()
    
    # Resolve config path relative to script location if not absolute
    if not os.path.isabs(args.config):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config
    
    # Load and validate config
    print(f"Loading configuration from: {config_path}")
    config = load_config(config_path)
    validate_config(config)
    
    # Resolve plot_figure_save_path relative to script location if not absolute
    if not os.path.isabs(config['plot_figure_save_path']):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        config['plot_figure_save_path'] = os.path.join(
            script_dir, config['plot_figure_save_path']
        )
    
    # Run the experiment sweep
    print("Configuration loaded successfully.")
    print(f"  GRID_SIZE: {config['GRID_SIZE']}")
    print(f"  BETA_NAIVE: {config['BETA_NAIVE']}")
    print(f"  Rounds: {config['rounds']}")
    set_random_seed(config)
    print(f"  Sweep variable: {config['sweep_variable']}")
    print(f"  Sweep values: {config['sweep_values']}")
    print(f"  Output path: {config['plot_figure_save_path']}")
    print()
    
    run_conflict_sweep(config, debug=args.debug)


if __name__ == "__main__":
    main()
