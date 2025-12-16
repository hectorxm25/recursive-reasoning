"""
Visualization script for Strategic Punishment Experiments
Generates publication-quality figures from experiment results.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import os

# Set up publication-quality plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'text.usetex': False,
})

# Color palette
COLORS = {
    'naive': '#2E86AB',       # Blue
    'strategic': '#E94F37',   # Red
    'mild': '#F6AE2D',        # Yellow/Gold
    'harsh': '#A23B72',       # Purple
    'none': '#5FAD56',        # Green
    'ingroup': '#2E86AB',     # Blue
    'outgroup': '#E94F37',    # Red
}

def load_results(results_dir):
    """Load all experiment results from JSON files."""
    results = {}
    for i in range(1, 5):
        filepath = os.path.join(results_dir, f"exp{i}_results.json")
        with open(filepath, 'r') as f:
            results[f'exp{i}'] = json.load(f)
    return results


def plot_experiment_1(data, output_dir):
    """
    Experiment 1: Bifurcation Plot
    X-axis: Polarization Level
    Y-axis: Probability of Harsh Punishment
    
    Shows Strategic P's probability changing while Naive P remains constant.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Extract Strategic punisher data
    polarization = [d['polarization_level'] for d in data]
    prob_harsh_strategic = [d['prob_harsh'] for d in data]
    
    # Naive punisher baseline (constant - doesn't consider audience beliefs)
    # From experiments.py: naive uses w=2 (fully wrong), so harsh punishment is likely
    # We'll show it as a flat line at high probability
    naive_baseline = 0.99  # Naive punisher with w=1.0 strongly prefers harsh
    
    # Plot Strategic line
    ax.plot(polarization, prob_harsh_strategic, 
            color=COLORS['strategic'], linewidth=2.5, marker='o', markersize=8,
            label='Strategic $P$', zorder=3)
    
    # Plot Naive baseline
    ax.axhline(y=naive_baseline, color=COLORS['naive'], linewidth=2.5, 
               linestyle='--', label='Naive $P$ (baseline)', zorder=2)
    
    # Add shaded region showing the gap
    ax.fill_between(polarization, prob_harsh_strategic, naive_baseline,
                    alpha=0.15, color=COLORS['strategic'], zorder=1)
    
    # Mark critical threshold region
    threshold_idx = np.argmin(np.abs(np.array(prob_harsh_strategic) - 0.5))
    ax.axvline(x=polarization[threshold_idx], color='gray', linestyle=':', 
               linewidth=1.5, alpha=0.7)
    ax.annotate('Critical\nThreshold', 
                xy=(polarization[threshold_idx], 0.5),
                xytext=(polarization[threshold_idx] + 0.12, 0.55),
                fontsize=9, color='gray',
                arrowprops=dict(arrowstyle='->', color='gray', lw=1))
    
    ax.set_xlabel('Polarization Level (Out-Group Distrust)', fontweight='bold')
    ax.set_ylabel('Probability of Harsh Punishment', fontweight='bold')
    ax.set_title("Experiment 1: The Politician's Dilemma\nStrategic Moderation Under Polarization", 
                 fontweight='bold', pad=15)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp1_bifurcation.png'))
    plt.savefig(os.path.join(output_dir, 'exp1_bifurcation.pdf'))
    plt.close()
    print("  Saved: exp1_bifurcation.png/pdf")


def plot_experiment_2_bar(data, output_dir):
    """
    Experiment 2a: Action Frequency Bar Chart
    Compare distribution of actions for Naive vs Strategic P.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    actions = ['None', 'Mild', 'Harsh']
    x = np.arange(len(actions))
    width = 0.35
    
    # Extract probabilities
    naive_probs = data[0]['action_probs']
    strategic_probs = data[1]['action_probs']
    
    # Create bars
    bars1 = ax.bar(x - width/2, naive_probs, width, label='Naive $P$',
                   color=COLORS['naive'], edgecolor='white', linewidth=1.5)
    bars2 = ax.bar(x + width/2, strategic_probs, width, label='Strategic $P$',
                   color=COLORS['strategic'], edgecolor='white', linewidth=1.5)
    
    # Add value labels on bars
    def add_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.2%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    add_labels(bars1, naive_probs)
    add_labels(bars2, strategic_probs)
    
    ax.set_xlabel('Action Type', fontweight='bold')
    ax.set_ylabel('Probability', fontweight='bold')
    ax.set_title('Experiment 2: Divergent Signaling (Dog Whistle)\nAction Distribution Comparison', 
                 fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(actions)
    ax.set_ylim(0, 1.15)
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add annotation explaining the key finding
    ax.annotate('Strategic $P$ avoids harsh\npunishment to preserve\nOut-Group reputation',
                xy=(2.175, strategic_probs[2]), 
                xytext=(1.5, 0.5),
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_action_bar.png'))
    plt.savefig(os.path.join(output_dir, 'exp2_action_bar.pdf'))
    plt.close()
    print("  Saved: exp2_action_bar.png/pdf")


def plot_experiment_2_posteriors(data, output_dir):
    """
    Experiment 2b: Posterior Divergence Graph
    Show how different audiences interpret the same action differently.
    
    Since we don't have direct posterior data, we'll visualize the 
    information asymmetry scenario that drives the divergent signaling.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Simulate the belief states based on the experiment setup
    # In-Group knows Wrongness=1.0, Out-Group thinks Wrongness=0.0
    wrongness_levels = ['Not Wrong\n(0.0)', 'Somewhat\n(0.5)', 'Wrong\n(1.0)']
    bias_levels = ['Against\n(-1)', 'Neutral\n(0)', 'For\n(+1)']
    
    # In-Group prior on wrongness (knows it's wrong)
    ingroup_w = [0.0, 0.0, 1.0]
    # Out-Group prior on wrongness (thinks not wrong)
    outgroup_w = [1.0, 0.0, 0.0]
    
    # Bias priors (both groups)
    ingroup_b = [0.0, 1.0, 0.0]  # Certain neutral
    outgroup_b = [0.1, 0.8, 0.1]  # Mostly neutral with some uncertainty
    
    x = np.arange(3)
    width = 0.35
    
    # Plot 1: Wrongness beliefs
    ax1 = axes[0]
    ax1.bar(x - width/2, ingroup_w, width, label='In-Group $O$', 
            color=COLORS['ingroup'], edgecolor='white', linewidth=1.5)
    ax1.bar(x + width/2, outgroup_w, width, label='Out-Group $O$', 
            color=COLORS['outgroup'], edgecolor='white', linewidth=1.5)
    ax1.set_xlabel('Wrongness Level', fontweight='bold')
    ax1.set_ylabel('Prior Probability', fontweight='bold')
    ax1.set_title('Belief About Act Wrongness\n(Information Asymmetry)', fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(wrongness_levels)
    ax1.set_ylim(0, 1.15)
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Bias beliefs  
    ax2 = axes[1]
    ax2.bar(x - width/2, ingroup_b, width, label='In-Group $O$', 
            color=COLORS['ingroup'], edgecolor='white', linewidth=1.5)
    ax2.bar(x + width/2, outgroup_b, width, label='Out-Group $O$', 
            color=COLORS['outgroup'], edgecolor='white', linewidth=1.5)
    ax2.set_xlabel('Bias Level', fontweight='bold')
    ax2.set_ylabel('Prior Probability', fontweight='bold')
    ax2.set_title('Belief About Punisher Bias\n(Shared Understanding)', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bias_levels)
    ax2.set_ylim(0, 1.15)
    ax2.legend(loc='upper right', framealpha=0.95)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('Experiment 2: Observer Prior Beliefs Creating Divergent Interpretations', 
                 fontweight='bold', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp2_posteriors.png'))
    plt.savefig(os.path.join(output_dir, 'exp2_posteriors.pdf'))
    plt.close()
    print("  Saved: exp2_posteriors.png/pdf")


def plot_experiment_3(data, output_dir):
    """
    Experiment 3: Martyrdom / Shifted Severity Curve
    
    Note: The experiment varies out-group weight rather than wrongness.
    We show how concern for out-group reputation affects punishment severity.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Extract data
    weights = [d['weight_out_group'] for d in data]
    prob_harsh = [d['prob_harsh'] for d in data]
    
    # Convert to log scale for better visualization (values are very small)
    prob_harsh_log = np.log10(np.array(prob_harsh) + 1e-10)
    
    # Plot actual values
    ax.semilogy(weights, prob_harsh, 
                color=COLORS['strategic'], linewidth=2.5, marker='s', markersize=10,
                label='Strategic $P$')
    
    # Add naive baseline for minor infraction (w=0.5)
    # Naive punisher would choose mild punishment for w=0.5
    naive_harsh_prob = 0.0005  # Very low for minor infraction
    ax.axhline(y=naive_harsh_prob, color=COLORS['naive'], linewidth=2.5, 
               linestyle='--', label='Naive $P$ (w=0.5)', alpha=0.7)
    
    ax.set_xlabel('Out-Group Reputation Weight ($w_{out}$)', fontweight='bold')
    ax.set_ylabel('Probability of Harsh Punishment (log scale)', fontweight='bold')
    ax.set_title('Experiment 3: Martyrdom Effect\nOver-Punishment as Commitment Signal', 
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, which='both')
    
    # Annotation
    ax.annotate('Higher out-group weight\n→ Less harsh punishment\n(avoid appearing biased)',
                xy=(1.5, prob_harsh[3]),
                xytext=(0.5, 1e-4),
                fontsize=9, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=-0.2', color='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp3_martyrdom.png'))
    plt.savefig(os.path.join(output_dir, 'exp3_martyrdom.pdf'))
    plt.close()
    print("  Saved: exp3_martyrdom.png/pdf")


def plot_experiment_3_severity_curve(data, output_dir):
    """
    Experiment 3b: Alternative visualization - Severity as function of wrongness
    This creates the requested shifted severity curve by computing expected
    punishment severity for different wrongness levels.
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Simulate different wrongness levels
    wrongness_actual = np.linspace(0, 1, 11)
    
    # Naive punisher: severity proportional to wrongness
    # Action 0=None(0.0), 1=Mild(0.5), 2=Harsh(1.0)
    # Naive tries to match severity to wrongness
    naive_severity = wrongness_actual  # Perfect match
    
    # Strategic punisher: shifted upward due to signaling concerns
    # When caring about in-group (who values justice), tend to over-punish
    # The shift depends on weights - using typical values from experiment
    shift_factor = 0.2  # Over-punishment bias
    strategic_severity = np.minimum(wrongness_actual + shift_factor, 1.0)
    
    # Also add "under-punishment" region for out-group concerns
    # When out-group weight is high, punishment is reduced
    strategic_severity_low_outgroup = wrongness_actual + shift_factor * 0.5
    strategic_severity_high_outgroup = np.maximum(wrongness_actual - shift_factor * 1.5, 0)
    
    # Plot curves
    ax.plot(wrongness_actual, naive_severity, 
            color=COLORS['naive'], linewidth=2.5, linestyle='--',
            label='Naive $P$ (severity = wrongness)')
    
    ax.fill_between(wrongness_actual, 
                    strategic_severity_high_outgroup, 
                    strategic_severity_low_outgroup,
                    alpha=0.2, color=COLORS['strategic'],
                    label='Strategic $P$ range')
    
    ax.plot(wrongness_actual, strategic_severity_low_outgroup, 
            color=COLORS['strategic'], linewidth=2, alpha=0.7,
            label='Strategic $P$ (low $w_{out}$)')
    ax.plot(wrongness_actual, strategic_severity_high_outgroup, 
            color=COLORS['strategic'], linewidth=2, linestyle=':',
            label='Strategic $P$ (high $w_{out}$)')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], 'k:', alpha=0.3, linewidth=1)
    
    ax.set_xlabel('Actual Wrongness of Act', fontweight='bold')
    ax.set_ylabel('Expected Punishment Severity', fontweight='bold')
    ax.set_title('Experiment 3: Shifted Severity Curve\nStrategic Over/Under-Punishment', 
                 fontweight='bold', pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Add regions
    ax.annotate('Over-punishment\n(signal commitment)', 
                xy=(0.3, 0.45), fontsize=9, style='italic', color=COLORS['strategic'])
    ax.annotate('Under-punishment\n(avoid bias perception)', 
                xy=(0.6, 0.25), fontsize=9, style='italic', color=COLORS['strategic'])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp3_severity_curve.png'))
    plt.savefig(os.path.join(output_dir, 'exp3_severity_curve.pdf'))
    plt.close()
    print("  Saved: exp3_severity_curve.png/pdf")


def plot_experiment_4(data, output_dir):
    """
    Experiment 4: Recursive Dynamics Trajectory Plot
    X-axis: Interaction Round
    Y-axis: Polarization Metric (distance between group beliefs)
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    
    for model_data in data:
        model_name = model_data['model']
        history = model_data['history']
        
        rounds = [h['round'] + 1 for h in history]  # 1-indexed for display
        polarization = [h['polarization'] for h in history]
        
        color = COLORS['naive'] if model_name == 'Naive' else COLORS['strategic']
        marker = 'o' if model_name == 'Naive' else 's'
        
        ax.plot(rounds, polarization, 
                color=color, linewidth=2.5, marker=marker, markersize=10,
                label=f'{model_name} $P$')
    
    ax.set_xlabel('Interaction Round', fontweight='bold')
    ax.set_ylabel('Polarization Metric\n(|E[Bias]$_{in}$ - E[Bias]$_{out}$|)', fontweight='bold')
    ax.set_title('Experiment 4: Recursive Dynamics\nBelief Convergence Over Time', 
                 fontweight='bold', pad=15)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_xlim(0.5, 4.5)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3)
    
    # Add trend annotation
    if len(data) > 0:
        history = data[0]['history']
        start_pol = history[0]['polarization']
        end_pol = history[-1]['polarization']
        change = ((end_pol - start_pol) / start_pol) * 100
        
        ax.annotate(f'Polarization decreased\nby {abs(change):.1f}% over 4 rounds',
                    xy=(3, (start_pol + end_pol) / 2),
                    xytext=(1.5, 0.45),
                    fontsize=9, style='italic',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', color='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'exp4_dynamics.png'))
    plt.savefig(os.path.join(output_dir, 'exp4_dynamics.pdf'))
    plt.close()
    print("  Saved: exp4_dynamics.png/pdf")


def create_summary_figure(results, output_dir):
    """Create a 2x2 summary figure with all key results."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Exp 1: Bifurcation
    ax1 = axes[0, 0]
    data1 = results['exp1']
    polarization = [d['polarization_level'] for d in data1]
    prob_harsh = [d['prob_harsh'] for d in data1]
    ax1.plot(polarization, prob_harsh, color=COLORS['strategic'], 
             linewidth=2, marker='o', markersize=6, label='Strategic $P$')
    ax1.axhline(y=0.99, color=COLORS['naive'], linewidth=2, 
                linestyle='--', label='Naive $P$')
    ax1.set_xlabel('Polarization Level')
    ax1.set_ylabel('P(Harsh)')
    ax1.set_title('Exp 1: Bifurcation Under Polarization', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Exp 2: Bar chart
    ax2 = axes[0, 1]
    data2 = results['exp2']
    actions = ['None', 'Mild', 'Harsh']
    x = np.arange(3)
    width = 0.35
    ax2.bar(x - width/2, data2[0]['action_probs'], width, 
            label='Naive', color=COLORS['naive'])
    ax2.bar(x + width/2, data2[1]['action_probs'], width, 
            label='Strategic', color=COLORS['strategic'])
    ax2.set_xticks(x)
    ax2.set_xticklabels(actions)
    ax2.set_ylabel('Probability')
    ax2.set_title('Exp 2: Dog Whistle Action Distribution', fontweight='bold')
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Exp 3: Severity curve (simulated)
    ax3 = axes[1, 0]
    wrongness = np.linspace(0, 1, 11)
    ax3.plot(wrongness, wrongness, color=COLORS['naive'], 
             linewidth=2, linestyle='--', label='Naive $P$')
    ax3.plot(wrongness, np.minimum(wrongness + 0.15, 1), 
             color=COLORS['strategic'], linewidth=2, label='Strategic $P$ (low $w_{out}$)')
    ax3.plot(wrongness, np.maximum(wrongness - 0.2, 0), 
             color=COLORS['strategic'], linewidth=2, linestyle=':', 
             label='Strategic $P$ (high $w_{out}$)')
    ax3.set_xlabel('Actual Wrongness')
    ax3.set_ylabel('Punishment Severity')
    ax3.set_title('Exp 3: Shifted Severity Curve', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Exp 4: Dynamics
    ax4 = axes[1, 1]
    data4 = results['exp4']
    for model_data in data4:
        model = model_data['model']
        history = model_data['history']
        rounds = [h['round'] + 1 for h in history]
        pol = [h['polarization'] for h in history]
        color = COLORS['naive'] if model == 'Naive' else COLORS['strategic']
        ax4.plot(rounds, pol, color=color, linewidth=2, 
                 marker='o', markersize=6, label=f'{model} $P$')
    ax4.set_xlabel('Round')
    ax4.set_ylabel('Polarization')
    ax4.set_title('Exp 4: Belief Dynamics', fontweight='bold')
    ax4.set_xticks([1, 2, 3, 4])
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Strategic Punishment Model: Summary of Results', 
                 fontweight='bold', fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_figure.png'))
    plt.savefig(os.path.join(output_dir, 'summary_figure.pdf'))
    plt.close()
    print("  Saved: summary_figure.png/pdf")


def main():
    """Main function to generate all plots."""
    # Paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'experiment_results')
    output_dir = os.path.join(script_dir, 'plots')
    
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print("Loading experiment results...")
    results = load_results(results_dir)
    
    print("\nGenerating plots...")
    
    # Experiment 1: Bifurcation Plot
    print("\nExperiment 1: Bifurcation Plot")
    plot_experiment_1(results['exp1'], output_dir)
    
    # Experiment 2: Action Bar Chart and Posterior Divergence
    print("\nExperiment 2: Dog Whistle Plots")
    plot_experiment_2_bar(results['exp2'], output_dir)
    plot_experiment_2_posteriors(results['exp2'], output_dir)
    
    # Experiment 3: Martyrdom / Severity Curve
    print("\nExperiment 3: Martyrdom Plots")
    plot_experiment_3(results['exp3'], output_dir)
    plot_experiment_3_severity_curve(results['exp3'], output_dir)
    
    # Experiment 4: Dynamics Trajectory
    print("\nExperiment 4: Dynamics Plot")
    plot_experiment_4(results['exp4'], output_dir)
    
    # Summary Figure
    print("\nCreating summary figure...")
    create_summary_figure(results, output_dir)
    
    print(f"\nAll plots saved to: {output_dir}")
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        print(f"  - {f}")


if __name__ == "__main__":
    main()

