function calculate_gene_specific_epsilon(X_obs, k_epsilon)
    # X_obs: 2D matrix (time steps x genes)
    time_steps, genes = size(X_obs)

    # Calculate variance and epsilon for each gene
    var_gene = var(X_obs, dims=1) |> vec  # Variance for each gene
    epsilon = k_epsilon .* sqrt.(var_gene)  # Epsilon for each gene
    return epsilon
end


function calculate_epsilon_per_trial(X_obs, trials, time_steps_per_trial, k_epsilon)
    # Reshape X_obs into a 3D array (trials, time steps, genes)
    genes = size(X_obs, 2)
    X_obs_3d = reshape(X_obs, trials, time_steps_per_trial, genes)

    # Initialize epsilon for each trial and gene
    epsilon = zeros(trials, genes)

    # Loop over trials to calculate variance and epsilon
    for trial in 1:trials
        # Calculate variance and epsilon for each gene in this trial
        var_gene = var(X_obs_3d[trial, :, :], dims=1) |> vec
        epsilon[trial, :] = k_epsilon .* sqrt.(var_gene)
    end
    return epsilon
end

function separate_phases_gene_specific(X_obs, trials, time_steps_per_trial, epsilon)
    # Reshape X_obs into a 3D array (trials, time steps, genes)
    genes = size(X_obs, 2)
    X_obs_3d = reshape(X_obs, trials, time_steps_per_trial, genes)

    # Initialize recovery end markers
    T_recover = fill(time_steps_per_trial, trials, genes)

    # Loop over trials and genes
    for trial in 1:trials
        for gene in 1:genes
            for t in 1:(time_steps_per_trial - 1)
                # Compute rate of change for the current gene
                delta_X = abs(X_obs_3d[trial, t+1, gene] - X_obs_3d[trial, t, gene])

                # Check if it falls below the gene-specific epsilon
                if delta_X <= epsilon[gene]
                    T_recover[trial, gene] = t
                    break
                end
            end
        end
    end
    return T_recover
end

function determine_steady_state(X_trial, X_steady, delta)
    # X_trial: 2D matrix (time steps x genes)
    # X_steady: Vector of steady-state values for each gene
    # delta: Vector of tolerances for each gene

    time_steps, genes = size(X_trial)
    T_recover = fill(time_steps, genes)  # Default to last time step

    for gene in 1:genes
        for t in 1:time_steps
            # Check if the value is within the steady-state range
            if X_trial[t, gene] >= X_steady[gene] - delta[gene] &&
               X_trial[t, gene] <= X_steady[gene] + delta[gene]
                T_recover[gene] = t
                break
            end
        end
    end
    return T_recover
end
