import numpy as np

def causal_information_bottleneck(joint_prob, p_sigma_plus, beta_values, num_clusters, 
                                 default_restarts=10, default_iter=300, 
                                 max_restarts=50, max_iter=500,
                                 enforce_monotonicity=True,
                                 convergence_threshold=1e-6,
                                 annealing=True,
                                 perturbation_size=0.1):
    """
    Implement the Causal Information Bottleneck (CIB) algorithm using equations 14-16
    from the paper.
    
    Parameters:
    joint_prob: Joint probability matrix p(σ+,σ-)
    p_sigma_plus: Marginal probability p(σ+)
    beta_values: List of beta values to compute the information curve
    num_clusters: Number of clusters to use in the representation
    default_restarts: Default number of random restarts
    default_iter: Default maximum number of iterations
    max_restarts: Maximum number of restarts for non-monotonic points
    max_iter: Maximum iterations for non-monotonic points
    enforce_monotonicity: Whether to enforce monotonicity of the information function
    convergence_threshold: Threshold for declaring convergence
    annealing: Whether to use annealing for larger beta values
    perturbation_size: Size of perturbation when using annealing
    
    Returns:
    tuple: (rate_values, distortion_values, feature_curve)
    """
    M = len(p_sigma_plus) - 1
    p_sigma_minus = np.sum(joint_prob, axis=0)  # Marginal p(σ-)
    p_sigma_minus_given_plus = conditional_probability_minus_given_plus(joint_prob, p_sigma_plus)
    
    rate_values = []       # I[R;σ+]
    distortion_values = []  # I[σ+;σ-|R] or equivalently I[σ+;σ-] - I[R;σ-]
    feature_curve = []     # (β, I[R;σ+]) pairs
    
    # Calculate total mutual information I[σ+;σ-]
    total_mi = np.sum(joint_prob[joint_prob > 0] * np.log2(joint_prob[joint_prob > 0] / 
                (p_sigma_plus[:, None] * p_sigma_minus)[joint_prob > 0]))
    
    print(f"Total mutual information I[σ+;σ-] = {total_mi:.6f} bits")
    
    # Process each beta value
    prev_rate = None
    prev_distortion = None
    
    total_betas = len(beta_values)
    best_pt_r_given_sigma_plus = None  # Store the best solution for annealing
    rate = None
    
    seed = 52
    
    for idx, beta in enumerate(beta_values):
        # Print progress percentage and current beta
        progress = (idx / total_betas) * 100
        print("###########################################################")
        print(f"[{progress:.1f}%] Processing β={beta:.6f}")
        
        # Default parameters for this beta
        num_restarts_for_beta = default_restarts
        max_iter_for_beta = default_iter
        
        # Run CIB for this beta, using annealing for larger beta values
        if annealing and beta > 1000 and best_pt_r_given_sigma_plus is not None:
            # Use annealing with previous solution
            rate, distortion, new_best_pt_r_given_sigma_plus = _run_cib_for_beta(
                beta, p_sigma_plus, p_sigma_minus, p_sigma_minus_given_plus, 
                total_mi, M, num_clusters, num_restarts_for_beta, max_iter_for_beta,
                convergence_threshold, seed=seed+idx, annealing=True, 
                initial_pt_r_given_sigma_plus=best_pt_r_given_sigma_plus,
                perturbation_size=perturbation_size
            )            
            best_pt_r_given_sigma_plus = new_best_pt_r_given_sigma_plus
        else:
            # Use random initialization for small beta or first run
            rate, distortion, new_best_pt_r_given_sigma_plus = _run_cib_for_beta(
                beta, p_sigma_plus, p_sigma_minus, p_sigma_minus_given_plus, 
                total_mi, M, num_clusters, num_restarts_for_beta, max_iter_for_beta,
                convergence_threshold, seed=seed+idx, annealing=False
            )
            best_pt_r_given_sigma_plus = new_best_pt_r_given_sigma_plus
        
        # Check monotonicity with previous point
        if enforce_monotonicity and prev_rate is not None and prev_distortion is not None and rate is not None:
            if rate < prev_rate:
                print(f"Non-monotonic point detected at β={beta}. Rerunning with increased parameters.")
                # Rerun with increased parameters
                if annealing and beta > 1000 and best_pt_r_given_sigma_plus is not None:
                    rate, distortion, new_best_pt_r_given_sigma_plus = _run_cib_for_beta(
                        beta, p_sigma_plus, p_sigma_minus, p_sigma_minus_given_plus, 
                        total_mi, M, num_clusters, max_restarts, max_iter,
                        convergence_threshold, seed=seed+idx, annealing=True,
                        initial_pt_r_given_sigma_plus=best_pt_r_given_sigma_plus,
                        perturbation_size=perturbation_size/2  # Reduced perturbation for more stability
                    )
                else:
                    rate, distortion, new_best_pt_r_given_sigma_plus = _run_cib_for_beta(
                        beta, p_sigma_plus, p_sigma_minus, p_sigma_minus_given_plus, 
                        total_mi, M, num_clusters, max_restarts, max_iter,
                        convergence_threshold, seed=seed+idx, annealing=False
                    )
                best_pt_r_given_sigma_plus = new_best_pt_r_given_sigma_plus
        
        rate_values.append(rate)
        distortion_values.append(distortion)
        feature_curve.append((beta, rate))
        
        # Update previous values for next iteration
        prev_rate = rate
        prev_distortion = distortion
    
    return np.array(rate_values), np.array(distortion_values), np.array(feature_curve)

def _run_cib_for_beta(beta, p_sigma_plus, p_sigma_minus, p_sigma_minus_given_plus, 
                     total_mi, M, num_clusters, num_restarts, max_iter, 
                     convergence_threshold, seed, annealing=False, initial_pt_r_given_sigma_plus=None, 
                     perturbation_size=0.1):
    """
    Helper function to run CIB algorithm for a single beta value.
    """
    best_objective = -np.inf
    best_h_sigma_plus_given_r = np.inf
    best_rate = None
    best_distortion = None
    best_pt_r_given_sigma_plus = None
    
    for restart in range(num_restarts):
        # Set a fixed random seed for reproducibility
        np.random.seed(seed + restart)
        
        # Initialize p_t(r|σ+) based on annealing option
        if annealing and initial_pt_r_given_sigma_plus is not None:
            # Use the provided solution with perturbation
            pt_r_given_sigma_plus = initial_pt_r_given_sigma_plus.copy()
            # Add random perturbation
            perturbation = np.random.uniform(-perturbation_size*np.mean(pt_r_given_sigma_plus), perturbation_size*np.mean(pt_r_given_sigma_plus), pt_r_given_sigma_plus.shape)
            pt_r_given_sigma_plus += perturbation
            # Ensure values are positive
            pt_r_given_sigma_plus = np.maximum(pt_r_given_sigma_plus, 1e-10)
            # Normalize
            pt_r_given_sigma_plus /= np.sum(pt_r_given_sigma_plus, axis=1, keepdims=True)
        else:
            # Initialize randomly
            pt_r_given_sigma_plus = np.random.random((M+1, num_clusters+1))
            pt_r_given_sigma_plus /= np.sum(pt_r_given_sigma_plus, axis=1, keepdims=True)
        
        # Initialize p_t(r)
        pt_r = np.sum(pt_r_given_sigma_plus * p_sigma_plus[:, None], axis=0)
        
        # Initialize p_t(σ-|r)
        pt_sigma_minus_given_r = np.zeros((num_clusters+1, M+1))
        for r in range(num_clusters+1):
            if pt_r[r] > 0:
                for sigma_minus in range(M+1):
                    value = np.sum(p_sigma_minus_given_plus[:, sigma_minus] * 
                                   (pt_r_given_sigma_plus[:, r] * p_sigma_plus / pt_r[r]))
                    pt_sigma_minus_given_r[r, sigma_minus] = value
        
        # Iterate until convergence
        for iteration in range(max_iter):
            # Equation 14: Update p_t(r|σ+)
            new_pt_r_given_sigma_plus = np.zeros((M+1, num_clusters+1))
            
            for sigma_plus in range(M+1):
                kl_values = np.zeros(num_clusters+1)
                for r in range(num_clusters+1):
                    # Normalize p_sigma_minus_given_plus
                    p_sigma_minus_given_plus_norm = p_sigma_minus_given_plus[sigma_plus, :] / np.sum(p_sigma_minus_given_plus[sigma_plus, :])
                    
                    # Check if sum is zero before normalizing
                    pt_sigma_minus_given_r_sum = np.sum(pt_sigma_minus_given_r[r, :])
                    if pt_sigma_minus_given_r_sum > 0:
                        pt_sigma_minus_given_r_norm = pt_sigma_minus_given_r[r, :] / pt_sigma_minus_given_r_sum
                    else:
                        # If sum is zero, continue with original (unnormalized) values
                        pt_sigma_minus_given_r_norm = pt_sigma_minus_given_r[r, :]
                    
                    kl_values[r] = np.sum(improved_kl_divergence(p_sigma_minus_given_plus_norm, pt_sigma_minus_given_r_norm)) / np.log(2)
                
                # Calculate unnormalized log probabilities
                # Replace zero values with a small epsilon to avoid log(0)
                
                pt_r_safe = np.copy(pt_r)
                pt_r_safe[pt_r_safe <= 0] = 1e-10  # Very small but non-zero value
                        
                # Calculate log probabilities
                log_unnorm_probs = np.log2(pt_r_safe) - beta * (kl_values)
                
                # Log-sum-exp trick for numerical stability
                max_log_prob = np.max(log_unnorm_probs)
                log_Z = max_log_prob + np.log2(np.sum(np.exp2(log_unnorm_probs - max_log_prob)))
                
                # Calculate normalized probabilities
                new_pt_r_given_sigma_plus[sigma_plus, :] = np.exp2(log_unnorm_probs - log_Z)

            
            # Equation 15: Update p_t(r)
            new_pt_r = np.sum(new_pt_r_given_sigma_plus * p_sigma_plus[:, None], axis=0)
            
            # Equation 16: Update p_t(σ-|r)
            new_pt_sigma_minus_given_r = np.zeros((num_clusters+1, M+1))
            for r in range(num_clusters+1):
                if new_pt_r[r] > 0:
                    for sigma_minus in range(M+1):
                        value = np.sum(p_sigma_minus_given_plus[:, sigma_minus] * 
                                       (new_pt_r_given_sigma_plus[:, r] * p_sigma_plus / new_pt_r[r]))
                        new_pt_sigma_minus_given_r[r, sigma_minus] = value
            
            pt_r_given_sigma_plus = new_pt_r_given_sigma_plus
            
            # Check that pt_r_given_sigma_plus is normalized for each sigma_plus
            if not np.allclose(np.sum(pt_r_given_sigma_plus, axis=1), 1.0, rtol=1e-5):
                print("Warning: pt_r_given_sigma_plus not normalized.")
            
            pt_r = new_pt_r
            pt_sigma_minus_given_r = new_pt_sigma_minus_given_r 
                                                           
        pt_r[pt_r <= 0] = 1e-8
        pt_r_given_sigma_plus[pt_r_given_sigma_plus <= 0] = 1e-8
        
        
        rate = 0
        for sigma_plus in range(M+1):
            for r in range(num_clusters+1):
                if pt_r_given_sigma_plus[sigma_plus, r] > 1e-8 and pt_r[r] > 1e-8:
                    rate += p_sigma_plus[sigma_plus] * pt_r_given_sigma_plus[sigma_plus, r] * \
                                        np.log2(pt_r_given_sigma_plus[sigma_plus, r] / (pt_r[r]))         
        
        # Calculate I[R;σ+] (rate)
        # Original version that may cause RuntimeWarning due to division by zero
        # rate = np.sum(p_sigma_plus[:, None] * pt_r_given_sigma_plus * 
        #               np.log2(pt_r_given_sigma_plus / pt_r[None, :]), where=(pt_r_given_sigma_plus > 1e-8) & (pt_r[None, :] > 1e-8))
        
        # # Revised version to handle potential division by zero and overflow
        # with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
        #     # Calculate the log term safely, setting invalid or overflow results to zero
        #     log_term = np.where((pt_r_given_sigma_plus > 1e-8) & (pt_r[None, :] > 1e-8),
        #                         np.log2(np.clip(pt_r_given_sigma_plus / pt_r[None, :], 1e-8, None)),
        #                         0)
        #     rate = np.sum(p_sigma_plus[:, None] * pt_r_given_sigma_plus * log_term)
        

        
        # #Compare the two rates and print if they differ significantly
        # if not np.isclose(rate, alternative_rate, rtol=1e-8):
        #     print(f"Discrepancy in I[R;σ+] calculation: rate={rate}, alternative_rate={alternative_rate}, difference={rate-alternative_rate} \n pt_r_given_sigma_plus={pt_r_given_sigma_plus} \n pt_r={pt_r}")
        
        # Calculate H[σ+] (Shannon entropy of σ+)
        h_sigma_plus = -np.sum(p_sigma_plus * np.log2(p_sigma_plus), where=(p_sigma_plus > 1e-8))
        
        # Calculate H[σ+|R] (conditional entropy)
        h_sigma_plus_given_r = 0
        for r in range(num_clusters+1):
            if pt_r[r] > 1e-8:
                p_sigma_plus_given_r = pt_r_given_sigma_plus[:, r] * p_sigma_plus / pt_r[r]
                # Avoid log2(0) by using np.where to filter out zero probabilities
                valid_indices = p_sigma_plus_given_r > 1e-8
                h_sigma_plus_given_r -= np.sum(pt_r[r] * p_sigma_plus_given_r[valid_indices] * 
                                               np.log2(p_sigma_plus_given_r[valid_indices]))
        
        # Calculate rate using alternative method
        rate_alt = h_sigma_plus - h_sigma_plus_given_r
        
        #Debug rate calculation differences
        if not np.isclose(rate, rate_alt, rtol=1e-6):
            print(f"Rate calculation discrepancy: rate={rate}, alt_rate={rate_alt}, difference={rate-rate_alt}")
            print(f"H[σ+]={h_sigma_plus}, H[σ+|R]={h_sigma_plus_given_r}")
        
        # Calculate I[R;σ-] (relevant information) using the first method
        info_r_sigma_minus = 0
        for r in range(num_clusters+1):
            for sigma_minus in range(M+1):
                if pt_r[r] > 1e-8 and pt_sigma_minus_given_r[r, sigma_minus] > 1e-8 and p_sigma_minus[sigma_minus] > 1e-8:
                    pt_r_sigma_minus = pt_r[r] * pt_sigma_minus_given_r[r, sigma_minus]
                    denominator = pt_r[r] * p_sigma_minus[sigma_minus]
                    info_r_sigma_minus += pt_r_sigma_minus * \
                                              np.log2(pt_r_sigma_minus / denominator) 
        
        pt_sigma_minus_given_r[pt_sigma_minus_given_r <= 0] = 1e-8
        
        # # Calculate I[R;σ-] (relevant information) using the second method
        # info_r_sigma_minus = np.sum(pt_r[:, None] * pt_sigma_minus_given_r * 
        #                             np.log2(pt_r[:, None]* pt_sigma_minus_given_r / (pt_r[:, None] * p_sigma_minus[None, :])), 
        #                             where=(pt_sigma_minus_given_r > 1e-8) & (pt_r[None, :] > 1e-8) & (p_sigma_minus[None, :] > 1e-8))
        
        # #Check if the two methods yield similar results
        # if not np.isclose(info_r_sigma_minus, alt_info_r_sigma_minus, rtol=1e-8):
        #     print(f"Discrepancy in I[R;σ-] calculation: first method={alt_info_r_sigma_minus}, second method={info_r_sigma_minus}, difference={alt_info_r_sigma_minus - info_r_sigma_minus}")
        
        # Distortion = I[σ+;σ-] - I[R;σ-]
        distortion = total_mi - info_r_sigma_minus
        
        # Calculate objective function
        objective = info_r_sigma_minus - (1/beta) * rate
        
        if np.isnan(objective) or np.isnan(distortion):
            print(f"NaN detected in objective or distortion: objective={objective}, distortion={distortion}")
            continue
        
        if objective > best_objective:
            best_h_sigma_plus = h_sigma_plus
            best_h_sigma_plus_given_r = h_sigma_plus_given_r
            best_objective = objective
            best_rate = rate
            best_rate_alt = rate_alt
            best_distortion = distortion
            best_pt_r_given_sigma_plus = pt_r_given_sigma_plus
            best_pt_r = pt_r
            best_pt_sigma_minus_given_r = pt_sigma_minus_given_r
    
    print('best_objective', best_objective, 'best_distortion', best_distortion, 'best_rate', best_rate)
    print('H[σ+]', best_h_sigma_plus, 'H[σ+|R]', best_h_sigma_plus_given_r)
    
    # Final check for NaN in best values
    if best_rate is None or best_distortion is None or np.isnan(best_rate) or np.isnan(best_distortion):
        print("WARNING: Best rate or distortion contains NaN or is None. Algorithm may have failed for all restarts.")
    
    return best_rate, best_distortion, best_pt_r_given_sigma_plus

def conditional_probability_minus_given_plus(joint_prob, p_sigma_plus):
    """
    Calculate p(σ-|σ+) from joint and marginal distributions.
    
    Parameters:
    joint_prob: Joint probability matrix p(σ+,σ-)
    p_sigma_plus: Marginal probability p(σ+)
    
    Returns:
    numpy.ndarray: Conditional probability matrix p(σ-|σ+)
    """
    M = len(p_sigma_plus) - 1
    p_sigma_minus_given_plus = np.zeros((M+1, M+1))
    
    for sigma_plus in range(M+1):
        if p_sigma_plus[sigma_plus] > 0:
            p_sigma_minus_given_plus[sigma_plus, :] = joint_prob[sigma_plus, :] / p_sigma_plus[sigma_plus]
    
    return p_sigma_minus_given_plus


def safe_kl_divergence(p, q):
    """
    Safely compute KL divergence between distributions p and q.
    Handles zeros in p and q appropriately.
    """
    # Where p is 0, the KL contribution is 0
    safe_p = np.copy(p)
    safe_q = np.copy(q)
    
    # Set very small values to avoid division by zero
    safe_q[safe_q < 1e-10] = 1e-10
    
    # Calculate KL only where p > 0
    kl = np.zeros_like(p)
    positive_p = safe_p > 0
    if np.any(positive_p):
        kl[positive_p] = safe_p[positive_p] * np.log2(safe_p[positive_p] / safe_q[positive_p])
    
    return np.sum(kl)

def improved_kl_divergence(p, q):
    """More robust KL divergence calculation for extreme values"""
    # Ensure no zeros
    p_safe = np.maximum(p, 1e-10)
    q_safe = np.maximum(q, 1e-10)
    
    # Normalize
    p_safe = p_safe / np.sum(p_safe)
    q_safe = q_safe / np.sum(q_safe)
    
    # Calculate KL directly in log domain
    kl = 0
    for i in range(len(p)):
        if p_safe[i] > 1e-10 and q_safe[i] <= 1e-10:  # Only consider non-negligible probabilities
            kl += np.inf
        elif p_safe[i] > 1e-10 and q_safe[i] > 1e-10:
            kl += p_safe[i] * (np.log2(p_safe[i]) - np.log2(q_safe[i]))
    
    return kl