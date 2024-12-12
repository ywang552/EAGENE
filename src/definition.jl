
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


function crowding_distance_assignment(front::Vector{Chromosome})
    num_individuals = length(front)
    distances = zeros(Float64, num_individuals)

    # Sort by each objective
    objectives = [:fitness, :sparsity]
    for obj in objectives
        # Sort individuals by the current objective
        sorted_indices = sortperm([getfield(front[i], obj) for i in 1:num_individuals])

        # Boundary points get infinite distance
        distances[sorted_indices[1]] = Inf
        distances[sorted_indices[end]] = Inf

        # Compute normalized distances for interior points
        obj_min = getfield(front[sorted_indices[1]], obj)
        obj_max = getfield(front[sorted_indices[end]], obj)

        if obj_max != obj_min
            for k in 2:(num_individuals - 1)
                numerator = getfield(front[sorted_indices[k + 1]], obj) -
                            getfield(front[sorted_indices[k - 1]], obj)
                denominator = obj_max - obj_min

                # Handle potential Inf/Inf or NaN
                if isinf(numerator) && isinf(denominator)
                    distances[sorted_indices[k]] += 0.0  # Assign default contribution
                elseif isnan(numerator / denominator)
                    distances[sorted_indices[k]] += 0.0
                else
                    distances[sorted_indices[k]] += numerator / denominator
                end
            end
        else
            # No variation in this objective, assign 0 contribution for all interior points
            for k in 2:(num_individuals - 1)
                distances[sorted_indices[k]] += 0.0
            end
        end
    end

    # Add distances to the individuals
    for i in 1:num_individuals
        front[i].distance = distances[i]
    end
end