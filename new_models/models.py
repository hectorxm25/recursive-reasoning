import jax
import jax.numpy as jnp
from jax import vmap
from jax.scipy.stats import beta

# Note: Device configuration (CPU/GPU) should be set by the calling script
# BEFORE importing this module. See plot_results.py for example.

# -----------------------------------------------------------------------------
# GRID SETUP FUNCTIONS
# -----------------------------------------------------------------------------

def setup_grids(grid_size):
    """Create the grids for W, B, J based on grid size."""
    W_GRID = jnp.linspace(0.0, 1.0, grid_size)
    B_GRID = jnp.linspace(-0.5, 0.5, grid_size)  
    J_GRID = jnp.linspace(0.0, 1.0, grid_size)
    return W_GRID, B_GRID, J_GRID

# Fixed action-related constants
ACTIONS = jnp.array([0, 1, 2])       # None, Mild, Harsh
SEVERITY = jnp.array([0.0, 0.5, 1.0])
# COST_TARGET = jnp.array([0.0, -0.5, -1.0])
COST_TARGET = jnp.array([0.0, -0.5, -0.8]) # this is for other settings, less cost on harsh
# COST_SELF = jnp.array([0.0, -0.1, -0.2])
COST_SELF = jnp.array([0.0, 0.0, 0.0]) # this is for other settings; not focusing on selfishness, see Radkani et al. 

# -----------------------------------------------------------------------------
# NAIVE UTILITY & LIKELIHOODS
# -----------------------------------------------------------------------------

@jax.jit
def compute_naive_utility(a_idx, w, b, j):
    u_bias = b * COST_TARGET[a_idx]
    dist = jnp.abs(SEVERITY[a_idx] - w)
    u_justice = j * (-dist)
    u_self = COST_SELF[a_idx]
    return u_bias + u_justice + u_self

def precompute_likelihoods(W_GRID, B_GRID, J_GRID, BETA_NAIVE):
    w_mesh, b_mesh, j_mesh = jnp.meshgrid(W_GRID, B_GRID, J_GRID, indexing='ij')
    def get_grid_utility(a_idx):
        return compute_naive_utility(a_idx, w_mesh, b_mesh, j_mesh)
    utilities = vmap(get_grid_utility)(ACTIONS)
    # Numerical stability: subtract max before exp
    utilities = utilities - jnp.max(utilities, axis=0)
    probs = jax.nn.softmax(BETA_NAIVE * utilities, axis=0)
    return probs

# -----------------------------------------------------------------------------
# OBSERVER INFERENCE & METRICS
# -----------------------------------------------------------------------------

def make_observer_update(LIKELIHOOD_TENSOR):
    """Create an observer_update function with the given likelihood tensor."""
    @jax.jit
    def observer_update(prior_tensor, action_idx):
        likelihood = LIKELIHOOD_TENSOR[action_idx]
        unnorm_post = prior_tensor * likelihood
        posterior =  unnorm_post / (jnp.sum(unnorm_post) + 1e-10)
        # add some jitter for numerical stability
        smooth_posterior = posterior * 0.99 + 0.01 * (1.0 / posterior.size)
        return smooth_posterior
    return observer_update

def make_get_metrics(W_GRID, B_GRID, J_GRID):
    """Create a get_metrics function with the given grids."""
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
    return get_metrics


# -----------------------------------------------------------------------------
# COMMUNICATIVE UTILITY HELPER
# -----------------------------------------------------------------------------

def make_authority_belief_dist(true_w, W_GRID):
    """
    Constructs a sharp probability distribution Q centered on the authority's
    true_w scalar. This allows us to compute KL(P || Q).
    """
    # sharp peak on true_w to convert to beta distribution
    concentration = 100.0 

    # clip true_w to avoid bounding errors
    w_safe = jnp.clip(true_w, 0.001, 0.999)

    # convert to alpha, beta parameters
    # 1.0 to ensure alpha, beta >= 1.0 for stability
    alpha = w_safe * concentration + 1.0
    beta_param = (1.0 - w_safe) * concentration + 1.0
    
    # Calculate PDF over the grid
    dist = beta.pdf(W_GRID, alpha, beta_param)
    
    # normalize to sum to 1.0
    return dist / (jnp.sum(dist) + 1e-10)

# -----------------------------------------------------------------------------
# STRATEGIC PLANNER
# -----------------------------------------------------------------------------

def make_get_strategic_action_probs(observer_update, get_metrics, W_GRID):
    """Create a strategic action probs function with the given observer and metrics functions.
    
    Args:
        observer_update: Function to update observer beliefs
        get_metrics: Function to compute metrics from posterior
        W_GRID: The W grid from the model context (used for communicative utility)
    """
    
    def _compute_utility_components(prior_in, prior_out, true_w, true_b, true_j, weights, a_idx):
        """Compute individual utility components for a single action."""
        # Pre-calculate Authority's Ideal Distribution (Q) for Communication
        q_w_dist = make_authority_belief_dist(true_w, W_GRID)
        
        # A. Intrinsic Utility
        u_int = compute_naive_utility(a_idx, true_w, true_b, true_j)
        
        # B. Reputational Utility
        post_in = observer_update(prior_in, a_idx)
        metrics_in = get_metrics(post_in)
        
        post_out = observer_update(prior_out, a_idx)
        metrics_out = get_metrics(post_out)

        # Maximizing Justice (e_j) and Impartiality (neutral e_b)
        u_rep = (
            weights['w_J_in'] * metrics_in['e_j'] + weights['w_J_out'] * metrics_out['e_j'] -
            weights['w_B_in_neu'] * jnp.abs(metrics_in['e_b']) - weights['w_B_out_neu'] * jnp.abs(metrics_out['e_b'])
        )
        
        # C. Communicative Utility (Jensen-Shannon Divergence)
        p_w_in = jnp.sum(post_in, axis=(1, 2))
        p_w_out = jnp.sum(post_out, axis=(1, 2))
        
        def jsd(p, q):
            m = 0.5 * (p + q)
            kl_pm = jnp.sum(p * jnp.log((p + 1e-10) / (m + 1e-10)))
            kl_qm = jnp.sum(q * jnp.log((q + 1e-10) / (m + 1e-10)))
            return 0.5 * kl_pm + 0.5 * kl_qm
        
        jsd_in = jsd(p_w_in, q_w_dist)
        jsd_out = jsd(p_w_out, q_w_dist)
        
        # Utility is NEGATIVE JSD (minimize distance = maximize utility)
        u_comm = -1.0 * (jsd_in + jsd_out)
        
        return u_int, u_rep, u_comm
    
    @jax.jit
    def get_strategic_action_probs(prior_in, prior_out, true_w, true_b, true_j, weights):
        
        # Pre-calculate Authority's Ideal Distribution (Q) for Communication
        q_w_dist = make_authority_belief_dist(true_w, W_GRID)
        
        def utility_for_action(a_idx):
            # A. Intrinsic Utility
            u_int = compute_naive_utility(a_idx, true_w, true_b, true_j)
            
            # B. Reputational Utility
            post_in = observer_update(prior_in, a_idx)
            metrics_in = get_metrics(post_in)
            
            post_out = observer_update(prior_out, a_idx)
            metrics_out = get_metrics(post_out)

            # --- 2. Reputational Utility ---
            # Maximizing Justice (e_j) and Impartiality (neutral e_b)
            u_rep = (
                weights['w_J_in'] * metrics_in['e_j'] + weights['w_J_out'] * metrics_out['e_j'] -        # Be seen as Just
                weights['w_B_in_neu'] * jnp.abs(metrics_in['e_b']) - weights['w_B_out_neu'] * jnp.abs(metrics_out['e_b']) # Be seen as Neutral
            )
            
            # C. Communicative Utility (Jensen-Shannon Divergence)
            # Minimize JSD(P || Q) where P is Audience Belief, Q is Authority Belief
            # JSD is symmetric and bounded [0, 1], more stable than KL divergence
            
            # Step 1: Marginalize Audience Posterior to get their W-belief (P)
            # post_in has shape (W, B, J). Sum over axis 1 (B) and 2 (J) to get W.
            p_w_in = jnp.sum(post_in, axis=(1, 2))
            p_w_out = jnp.sum(post_out, axis=(1, 2))
            
            # Step 2: Compute JSD(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)
            # where M = 0.5 * (P + Q) is the mixture distribution
            # We add 1e-10 to avoid division by zero or log(0)
            def jsd(p, q):
                m = 0.5 * (p + q)
                kl_pm = jnp.sum(p * jnp.log((p + 1e-10) / (m + 1e-10)))
                kl_qm = jnp.sum(q * jnp.log((q + 1e-10) / (m + 1e-10)))
                return 0.5 * kl_pm + 0.5 * kl_qm
            
            jsd_in = jsd(p_w_in, q_w_dist)
            jsd_out = jsd(p_w_out, q_w_dist)
            
            # Utility is NEGATIVE JSD (minimize distance = maximize utility)
            u_comm = -1.0 * (jsd_in + jsd_out)
            
            # Total Utility
            return (weights['scale_int'] * u_int + 
                    weights['scale_rep'] * u_rep + 
                    weights.get('scale_comm', 0.0) * u_comm)

        strat_utilities = vmap(utility_for_action)(ACTIONS)
        
        # Higher beta_strat makes the agent follow the utility more strictly
        return jax.nn.softmax(weights['beta_strat'] * strat_utilities)
    
    def get_utility_components(prior_in, prior_out, true_w, true_b, true_j, weights):
        """
        Returns the individual utility components for each action.
        
        Returns:
            dict with keys for each action (0=None, 1=Mild, 2=Harsh), each containing:
                'u_int': intrinsic utility
                'u_rep': reputational utility  
                'u_comm': communicative utility
        """
        results = {}
        for a_idx in [0, 1, 2]:
            u_int, u_rep, u_comm = _compute_utility_components(
                prior_in, prior_out, true_w, true_b, true_j, weights, a_idx
            )
            results[a_idx] = {
                'u_int': float(u_int),
                'u_rep': float(u_rep),
                'u_comm': float(u_comm)
            }
        return results
    
    return get_strategic_action_probs, get_utility_components

# -----------------------------------------------------------------------------
# DIAGNOSTIC HELPER
# -----------------------------------------------------------------------------

def debug_utilities(prior_in, prior_out, true_w, true_b, true_j, weights, observer_update, get_metrics):
    """
    Prints the raw utility values for Action 0 (None) vs Action 2 (Harsh).
    Call this ONCE at the start of a simulation to sanity check scales.
    """
    print(f"\n--- DEBUG UTILITIES (B_out_dir={weights['w_B_out_dir']}, B_out_neu={weights['w_B_out_neu']}) ---")
    
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
    
    # 3. Weighted Total (directional + neutrality terms)
    u_rep_none = weights['w_B_out_dir'] * b_none - weights['w_B_out_neu'] * jnp.abs(b_none)
    u_rep_harsh = weights['w_B_out_dir'] * b_harsh - weights['w_B_out_neu'] * jnp.abs(b_harsh)
    
    total_none = weights['scale_int'] * u_int_none + weights['scale_rep'] * u_rep_none
    total_harsh = weights['scale_int'] * u_int_harsh + weights['scale_rep'] * u_rep_harsh
    
    print(f"Total Util:     None={total_none:.4f} | Harsh={total_harsh:.4f}")
    print(f"WINNER:         {'HARSH' if total_harsh > total_none else 'NONE'}")
    print("------------------------------------------------\n")

# -----------------------------------------------------------------------------
# PRIOR TENSOR CREATION
# -----------------------------------------------------------------------------

def create_prior_tensor(w_params, b_params, j_params, W_GRID, B_GRID, J_GRID):
    pdf_w = beta.pdf(W_GRID, *w_params); pdf_w /= pdf_w.sum()
    
    # Shift B_GRID by +0.5 so it maps from [-0.5, 0.5] to [0.0, 1.0]
    pdf_b = beta.pdf(B_GRID + 0.5, *b_params); pdf_b /= pdf_b.sum()
    
    pdf_j = beta.pdf(J_GRID, *j_params); pdf_j /= pdf_j.sum()
    return jnp.einsum('i,j,k->ijk', pdf_w, pdf_b, pdf_j)


# -----------------------------------------------------------------------------
# MODEL CONTEXT CLASS
# -----------------------------------------------------------------------------

class ModelContext:
    """
    Holds all model state (grids, likelihood tensor, observer functions).
    This allows experiments to use different configurations without global state.
    """
    def __init__(self, grid_size, beta_naive):
        self.grid_size = grid_size
        self.beta_naive = beta_naive
        
        # Setup grids
        self.W_GRID, self.B_GRID, self.J_GRID = setup_grids(grid_size)
        
        # Precompute likelihoods
        self.LIKELIHOOD_TENSOR = precompute_likelihoods(
            self.W_GRID, self.B_GRID, self.J_GRID, beta_naive
        )
        
        # Create bound functions
        self.observer_update = make_observer_update(self.LIKELIHOOD_TENSOR)
        self.get_metrics = make_get_metrics(self.W_GRID, self.B_GRID, self.J_GRID)
        self.get_strategic_action_probs, self.get_utility_components = make_get_strategic_action_probs(
            self.observer_update, self.get_metrics, self.W_GRID
        )
    
    def create_prior_tensor(self, w_params, b_params, j_params):
        return create_prior_tensor(
            w_params, b_params, j_params,
            self.W_GRID, self.B_GRID, self.J_GRID
        )
    
    def debug_utilities(self, prior_in, prior_out, true_w, true_b, true_j, weights):
        debug_utilities(
            prior_in, prior_out, true_w, true_b, true_j, weights,
            self.observer_update, self.get_metrics
        )
