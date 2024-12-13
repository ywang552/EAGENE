
function evaluate_fitness_exponential!(chromosome::Chromosome; X_obs = re, nnz_value = 16,
                                        time_steps_per_trial = 11, trials = 5, k_exp = 2)
    # Extract chromosome parameters
    α = chromosome.α
    X_steady = chromosome.X_steady
    W = chromosome.W
    N = length(α)  # Number of genes

    # Initialize fitness
    total_fitness = 0.0

    # Loop over trials
    for trial in 1:trials
        # Extract trial-specific data
        trial_start = (trial - 1) * time_steps_per_trial + 1
        trial_end = trial * time_steps_per_trial
        X_trial = X_obs[trial_start:trial_end, :]

        # Loop over time steps and genes within this trial
        for t in 1:(time_steps_per_trial - 1)
            for j in 1:N
                # Predicted next state for gene j
                regulation_effect = sum(W[:, j] .* X_trial[t, :])
                X_pred = X_trial[t, j] + α[j] * (X_steady[j] - X_trial[t, j]) + regulation_effect

                # Exponential weighting loss
                loss = exp(k_exp * abs(X_trial[t+1, j] - X_pred)) - 1
                total_fitness -= loss
            end
        end
    end

    # Calculate number of non-zero elements in W
    nnz_W = count(!iszero, W)

    # Apply penalty based on nnz(W)
    if nnz_W > nnz_value
        total_fitness -= 1000 * (nnz_W - nnz_value)  # Severe penalty
    end

    # Normalize by the number of genes and time steps
    total_fitness /= (trials * time_steps_per_trial * N)

    # Update fitness in the chromosome struct
    chromosome.fitness = total_fitness
end

function find_kth_largest_sparse(matrix::SparseMatrixCSC{T}, k::Int) where T
    # Extract nonzero elements from the sparse matrix
    nonzeros = collect(matrix.nzval)  # `nzval` contains all nonzero values
    
    # Sort the nonzero elements in descending order
    sorted_nonzeros = sort(nonzeros, rev=true)
    
    # Ensure k is valid
    @assert k <= length(sorted_nonzeros) "k is larger than the number of nonzero elements"
    
    # Return the k-th largest element
    return sorted_nonzeros[k]
end
