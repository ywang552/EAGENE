import Base: show
using SparseArrays, Plots, StatsBase
using GeometryBasics: Point2f
using Colors
using LinearAlgebra
using MultivariateStats



const CHROMOSOME_ID_COUNTER = Ref(0)  # A mutable counter

function get_next_id()
    CHROMOSOME_ID_COUNTER[] += 1
    return CHROMOSOME_ID_COUNTER[]
end

function track_fitness(population, fitness_history)
    best_fitness = maximum([c.fitness for c in population])
    avg_fitness = mean([c.fitness for c in population])
    push!(fitness_history[:best], best_fitness)
    push!(fitness_history[:average], avg_fitness)
end

mutable struct Chromosome
    α::Vector{Float64}        # Self-regulation values
    X_steady::Vector{Float64} # Steady-state values
    W::SparseMatrixCSC{Float64, Int} # Gene regulation matrix
    fitness::Float64          # Primary fitness score (e.g., prediction error)
    parents::Tuple{Int, Int}  # Indices of parents (-1, -1 for initial population)
    generation::Int           # The generation this chromosome belongs to
    id::Int                   # Unique ID for the chromosome
    sparsity::Int             # Number of non-zero elements in W (secondary objective)
end



function show(io::IO, chromosome::Chromosome)
    println(io, "  Chromosome ID: ", chromosome.id)
    println(io, "  Generation: ", chromosome.generation)
    println(io, "  Parents: ", chromosome.parents)
    println(io, "  Fitness: ", round(chromosome.fitness, digits=3))  # Limit to 4 decimal places
    println(io, "  α (Self-Regulation): ", round.(chromosome.α, digits=3))  # Limit α precision
    println(io, "  X_steady (Steady-State Values): ", round.(chromosome.X_steady, digits=3))  # Limit X_steady precision
    println(io, "  W (Non-zero elements in Sparse Matrix): ", nnz(chromosome.W))

end






function decode_chromosome(chromosome, N)
    # Extract parameters
    α = chromosome[1:N]
    X_steady = chromosome[N+1:2N]
    W_flat = chromosome[2N+1:end]
    W = reshape(W_flat, N, N)  # Reshape W into a matrix

    return α, X_steady, sparse(W)
end

function validate_chromosome(chromosome::Chromosome)
    # Check constraints
    is_valid_α = all(chromosome.α .>= 0)
    is_valid_X_steady = all((0 .< chromosome.X_steady) .& (chromosome.X_steady .< 1))

    return is_valid_α && is_valid_X_steady
end


function repair_chromosome!(chromosome::Chromosome)
    # Fix α to satisfy α >= 0
    chromosome.α .= max.(chromosome.α, 0.0)

    # Fix X_steady to satisfy 0 < X_steady < 1
    chromosome.X_steady .= clamp.(chromosome.X_steady, 0.01, 0.99)
end


function check_and_fix_chromosome!(chromosome::Chromosome)
    if !validate_chromosome(chromosome)
        repair_chromosome!(chromosome)
    end
end





function initialize_population(pop_size, N, nnz_value)
    population = Chromosome[]
    for _ in 1:pop_size
        α = rand(N)
        X_steady = rand(N)
        W = sprand(Float64, N, N, nnz_value / (N * N))
        fitness = -Inf
        parents = (-1, -1)
        generation = 0
        id = get_next_id()
        sparsity = nnz(W)
        push!(population, Chromosome(α, X_steady, W, fitness, parents, generation, id, sparsity))
    end
    return population
end

function mutate_chromosome!(chromosome::Chromosome, mutation_rate::Float64)
    N = length(chromosome.α)

    # Mutate α
    for i in 1:N
        if rand() < mutation_rate
            chromosome.α[i] += 0.1 * randn()
        end
    end

    # Mutate X_steady
    for i in 1:N
        if rand() < mutation_rate
            chromosome.X_steady[i] += 0.1 * randn()
        end
    end

    # Mutate W (sparse matrix)
    for i in 1:N
        for j in 1:N
            if rand() < mutation_rate
                chromosome.W[i, j] += 0.1 * randn()
            end
        end
    end
end

function evaluate_fitness_exponential!(chromosome::Chromosome; re = re, nnz_value = 16, 
                                        time_steps_per_trial = 11, trials = 5, k_exp = 2)
    # Extract chromosome parameters
    α = chromosome.α
    X_steady = chromosome.X_steady
    W = chromosome.W
    N = length(α)  # Number of genes

    # Precompute (I - W)
    I_minus_W = Matrix{Float64}(I, size(W)) - W
    if rank(I_minus_W) < N
        error("Matrix (I - W) is singular and cannot be inverted.")
    end

    # Precomputed reshaped matrix (re = X_obs_reshaped)
    X_obs_reshaped = re  # Already reshaped during preprocessing
    X_current = X_obs_reshaped[1:end-1, :, :]  # Current state X_t

    # Compute predictions for all time steps and trials
    regulation_term = (I_minus_W) \ reshape(X_current, (N, :))  # Compute in batch
    regulation_term = reshape(regulation_term, size(X_current))  # Reshape back to 3D

    # Add self-regulation and steady-state correction
    α_correction = α .* (X_steady .- X_current)  # Broadcasting α term
    X_pred = X_current + α_correction + regulation_term  # Predicted next state

    # Compute fitness loss for all genes, time steps, and trials
    X_next = X_obs_reshaped[2:end, :, :]  # Observed next state X_t+1
    loss_matrix = exp.(k_exp .* abs.(X_next .- X_pred)) .- 1  # Element-wise loss
    total_fitness = -sum(loss_matrix)  # Accumulate total loss

    # Penalize non-zero elements in W
    nnz_W = count(!iszero, W)
    if nnz_W > nnz_value
        total_fitness -= 1000 * (nnz_W - nnz_value)
    end

    # Normalize fitness
    total_fitness /= (trials * time_steps_per_trial * N)

    # Update fitness in the chromosome struct
    chromosome.fitness = total_fitness
end

function evaluate_fitness_multiobjective!(chromosome::Chromosome; X_obs = re, nnz_value = 16,
                                          time_steps_per_trial = 11, trials = 5, k_exp = 2)
    # Use existing function to calculate prediction error
    evaluate_fitness_exponential!(chromosome;)

    # Prediction error is already stored in chromosome.fitness
    prediction_error = chromosome.fitness

    # Calculate sparsity as the number of non-zero elements in W
    sparsity = count(!iszero, chromosome.W)

    # Store sparsity as an additional objective
    chromosome.sparsity = sparsity
end



function evaluate_population_exponential!(population::Vector{Chromosome}; X_obs = re, nnz_value = nnz_value,
                                           time_steps_per_trial = 11, trials = 5, k_exp = 2)
    for chromosome in population
        evaluate_fitness_multiobjective!(chromosome)
    end
end


function visualize_fitness_history(fitness_history)
    generations = 1:length(fitness_history[:best])
    p = plot(
        generations,
        fitness_history[:best],
        xlabel="Generation",
        ylabel="Fitness",
        label = false,
        title="Fitness Evolution",
        linewidth=2
    )
    plot!(
        generations,
        label = false,
        xscale = :log10, 
        fitness_history[:average],
        linewidth=2,
        linestyle=:dash,
        ylim = [-10, 0]
    )
    p
end




function recombine_chromosomes(parent1::Chromosome, parent2::Chromosome, generation::Int, crossover_rate::Float64 = 0.3)
    @assert 0.0 <= crossover_rate <= 1.0 "Crossover rate must be between 0 and 1"

    N = length(parent1.α)  # Number of genes

    # Blend α (self-regulation values)
    α_offspring = (rand(N) .< crossover_rate) .* parent1.α .+ 
                  (rand(N) .>= crossover_rate) .* parent2.α

    # Blend X_steady (steady-state values)
    X_steady_offspring = (rand(N) .< crossover_rate) .* parent1.X_steady .+ 
                         (rand(N) .>= crossover_rate) .* parent2.X_steady

    # Sparse matrix crossover for W (gene regulation matrix)
    W_offspring = spzeros(N, N)
    for i in 1:N
        if rand() < crossover_rate
            W_offspring[i, :] = parent1.W[i, :]
        else
            W_offspring[i, :] = parent2.W[i, :]
        end
    end

    # Create the offspring chromosome
    offspring = Chromosome(
        α_offspring,
        X_steady_offspring,
        W_offspring,
        -Inf,  # Fitness not evaluated yet
        (parent1.id, parent2.id),  # Track parent IDs
        generation,  # Current generation
        get_next_id(),  # Assign unique ID to the offspring
        nnz(W_offspring)
    )

    return offspring
end

function pareto_front_selection(population::Vector{Chromosome})
    pareto_front = Chromosome[]
    for candidate in population
        dominated = false
        for competitor in population
            if competitor.fitness <= candidate.fitness && competitor.sparsity <= candidate.sparsity &&
               (competitor.fitness < candidate.fitness || competitor.sparsity < candidate.sparsity)
                dominated = true
                break
            end
        end
        if !dominated
            push!(pareto_front, candidate)
        end
    end
    return pareto_front
end


function sample_distinct_parents(parents::Vector{Chromosome})
    # Sample two distinct parents
    parent1 = rand(parents)
    parent2 = rand(filter(p -> p.id != parent1.id, parents))
    return parent1, parent2
end




function calculate_pca_diversity(population::Vector{Chromosome})
    # Flatten the W matrices for each chromosome in the population
    flattened_W = [reshape(Matrix(chromosome.W .!= 0), :) for chromosome in population]  # Binarize and flatten
    P = hcat(flattened_W...)'  # Create the population matrix (k × N^2)

    # Check if P has meaningful variance
    if std(P) < 1e-8
        println("Warning: Population has no meaningful diversity (all chromosomes nearly identical).")
        # All variance explained by the first PC
        explained_variance = [1.0]
        cumulative_variance = [1.0]
        num_components_95 = 1
        return nothing, explained_variance, cumulative_variance, num_components_95
    end

    # Perform PCA
    pca_model = fit(PCA, P; maxoutdim=min(size(P)...))  # Fit PCA to population matrix

    # Calculate explained and cumulative variance
    explained_variance = pca_model.prinvars / pca_model.tvar
    cumulative_variance = cumsum(explained_variance)
    num_components_95 = findfirst(cumulative_variance .>= 0.95)

    # Return PCA results
    return pca_model, explained_variance, cumulative_variance, num_components_95
end




function plot_pca_results(explained_variance, cumulative_variance)
    # Scree plot: Explained variance per component
    plt1 = plot(explained_variance, marker=:o, label="Explained Variance",
                xlabel="Principal Component", ylabel="Variance Explained",
                title="Scree Plot")

    # Cumulative explained variance
    plt2 = plot(cumulative_variance, marker=:o, label="Cumulative Variance",
                xlabel="Principal Component", ylabel="Cumulative Variance",
                title="Cumulative Variance Explained", color=:blue)

    # Combine plots
    # display(plt1)
    display(plot(plt1, plt2, layout=(1, 2), legend=:bottomright))
end


function plot_population_in_pca_space(pca_model)
    # Access the transformed data
    P_pca = pca_model.proj

    # Scatter plot in PCA space
    scatter(P_pca[:, 1], P_pca[:, 2], xlabel="PC1", ylabel="PC2",
            title="Population Diversity in PCA Space",
            legend=false, color=:blue, markersize=5)
end

function compute_top_fraction_variance(explained_variance, m)
    # Sum variance of top m components
    fraction_top = sum(explained_variance[1:min(m, length(explained_variance))])
    return fraction_top
end


# function evolutionary_algorithm(
#     population_size, generations;
#     X_obs = re, N = N, nnz_value = nnz_value, crossover_rate=0.7, mutation_rate=0.1,
#     time_steps_per_trial=11, trials=5, k_exp=2.0, pc = 50
# )
#     println("Starting...")
#     # Initialize the population
#     population = initialize_population(population_size, N, nnz_value)

#     entropy_history = []
#     fitness_history = Dict(:best => Float64[], :average => Float64[])

#     # try
#         for gen in 1:generations
#             pca_model, explained_variance, cumulative_variance, num_components_95 = calculate_pca_diversity(population)
#             push!(entropy_history, compute_top_fraction_variance(explained_variance, maximum([3, div(length(explained_variance), 10)])))

#             # Step 1: Evaluate fitness for the population
#             evaluate_population_exponential!(population; )

#             # Track fitness
#             track_fitness(population, fitness_history)

#             # Sort population by fitness (descending order)
#             population = sort(population, by=c -> c.fitness, rev=true)

#             # Print the best fitness
#             if gen % pc == 0
#                 println("Generation $gen")
#                 println("Best fitness: ", population[1].fitness)
#             end 

#             # Step 2: Selection - Retain the top half of the population
#             num_parents = div(population_size, 2)
#             parents = population[1:num_parents]

#             # Step 3: Recombination - Generate offspring
#             offspring = Chromosome[]
#             for _ in 1:(population_size - num_parents)
#                 parent1, parent2 = sample_distinct_parents(parents)
#                 child = recombine_chromosomes(parent1, parent2, gen)
#                 push!(offspring, child)
#             end

#             # Step 4: Mutation - Apply mutation to offspring
#             for child in offspring
#                 mutate_chromosome!(child, mutation_rate)
#                 check_and_fix_chromosome!(child)
#             end

#             # Replace the population with parents and offspring
#             population = vcat(parents, offspring)

#             # Visualization
#             if gen % pc == 0
#                 p1 = plot(1:length(entropy_history), xscale=:log10, label=false, entropy_history,
#                           xlabel="Generation", ylabel="Similarity", title="Population Similarity Over Time")
#                 p2 = visualize_fitness_history(fitness_history)
#                 display(plot(p1, p2, layout=(2, 1)))
#             end
#         end
#     # catch e 
#     #     # println("Error occurred in Generation $gen:")
#     #     println("Error Details: ", e)
#     #     # println(stacktrace(e))
#     #     throw(e)  # Rethrow the error after logging
#     # finally
#     #     evaluate_population_exponential!(population; )

#         # Always return the current population and all_chromosomes
#         return population[1], population
#     # end
# end

function plot_pareto_front(pareto_front::Vector{Chromosome})
    errors = [chromosome.fitness for chromosome in pareto_front]
    sparsities = [chromosome.sparsity for chromosome in pareto_front]

    scatter(sparsities, errors, xlabel="Sparsity (nnz)", ylabel="Prediction Error",
            title="Pareto Front: Sparsity vs. Prediction Error")
end


function evolutionary_algorithm(
    population_size, generations;
    X_obs = re, N = N, nnz_value = nnz_value, crossover_rate=0.7, mutation_rate=0.1,
    time_steps_per_trial=11, trials=5, k_exp=2.0, pc = 50
)
    println("Starting...")

    # Initialize the population
    population = initialize_population(population_size, N, nnz_value)

    entropy_history = []
    fitness_history = Dict(:best => Float64[], :average => Float64[])
    pareto_front_history = []

    for gen in 1:generations
        # Calculate PCA diversity
        pca_model, explained_variance, cumulative_variance, num_components_95 = calculate_pca_diversity(population)
        push!(entropy_history, compute_top_fraction_variance(explained_variance, maximum([3, div(length(explained_variance), 10)])))

        # Step 1: Evaluate multi-objective fitness for the population
        for chromosome in population
            evaluate_fitness_multiobjective!(chromosome, 
                                             X_obs=X_obs, 
                                             nnz_value=nnz_value, 
                                             time_steps_per_trial=time_steps_per_trial, 
                                             trials=trials, 
                                             k_exp=k_exp)
        end

        # Track fitness
        track_fitness(population, fitness_history)

        # Pareto front selection
        pareto_front = pareto_front_selection(population)
        push!(pareto_front_history, pareto_front)

        # Sort population by fitness (descending order for visualization purposes)
        population = sort(population, by=c -> c.fitness, rev=true)

        # Print the best fitness
        if gen % pc == 0
            println("Generation $gen")
            println("Best fitness: ", population[1].fitness)
        end 

        # Step 2: Selection - Retain the top half of the population
        num_parents = div(population_size, 2)
        parents = population[1:num_parents]

        # Step 3: Recombination - Generate offspring
        offspring = Chromosome[]
        for _ in 1:(population_size - num_parents)
            parent1, parent2 = sample_distinct_parents(parents)
            child = recombine_chromosomes(parent1, parent2, gen)
            push!(offspring, child)
        end

        # Step 4: Mutation - Apply mutation to offspring
        for child in offspring
            mutate_chromosome!(child, mutation_rate)
            check_and_fix_chromosome!(child)
        end

        # Replace the population with parents and offspring
        population = vcat(parents, offspring)

        # Visualization
        if gen % pc == 0
            # p1 = plot(1:length(entropy_history), xscale=:log10, label=false, entropy_history,
            #           xlabel="Generation", ylabel="Similarity", title="Population Similarity Over Time")
            # p2 = visualize_fitness_history(fitness_history)
            p3 = plot_pareto_front(pareto_front)
            display(p3)
            # display(plot(p1, p2, p3, layout=(3, 1), dpi = 6000))
        end
    end

    # Return the best chromosome, the entire population, and Pareto fronts
    return population[1], population, pareto_front_history
end






function predict_time_series(chromosome::Chromosome, re::Array{Float64, 3})
    """
    Predict the next states for a time series using a chromosome.
    
    Args:
        chromosome::Chromosome: The chromosome containing the state transition function.
        re::Array{Float64, 3}: Input data (time steps × trials × genes).
    
    Returns:
        Array{Float64, 3}: Predicted states (time steps - 1 × trials × genes).
    """
    T, trials, N = size(re)  # Dimensions: time steps × trials × genes

    # Precompute (I - W)
    I_minus_W = I - chromosome.W
    if rank(I_minus_W) < N
        error("Matrix (I - W) is singular and cannot be inverted.")
    end

    # Extract current states (all except the last time step)
    X_current = re[1:end-1, :, :]  # Shape: (T-1) × trials × genes

    # Reshape for batch matrix operations
    X_current_flat = reshape(X_current, (T-1) * trials, N)'  # Shape: genes × ((T-1) * trials)

    # Compute regulation term
    regulation_term_flat = (I_minus_W) \ X_current_flat  # Shape: genes × ((T-1) * trials)

    # Reshape back to 3D
    regulation_term = reshape(regulation_term_flat', (T-1, trials, N))

    # Add self-regulation and steady-state correction
    # Reshape α and X_steady to broadcast correctly across trials and time steps
    α_expanded = reshape(chromosome.α, (1, 1, N))  # Shape: 1 × 1 × genes
    X_steady_expanded = reshape(chromosome.X_steady, (1, 1, N))  # Shape: 1 × 1 × genes
    α_correction = α_expanded .* (X_steady_expanded .- X_current)  # Correction term

    # Compute final predicted states
    X_pred = X_current + α_correction + regulation_term  # Predicted states

    return X_pred
end



function evaluate_prediction_error(X_obs::Matrix{Float64}, X_pred::Matrix{Float64})
    """
    Evaluate error metrics between observed and predicted states.
    
    Args:
        X_obs::Matrix{Float64}: Observed data (time steps × genes).
        X_pred::Matrix{Float64}: Predicted data (time steps × genes).
    
    Returns:
        Dict: Error metrics (MSE, MAE, RMSE).
    """
    T, N = size(X_obs)
    mse = sum((X_obs .- X_pred).^2) / (T * N)
    mae = sum(abs.(X_obs .- X_pred)) / (T * N)
    rmse = sqrt(mse)

    return Dict("MSE" => mse, "MAE" => mae, "RMSE" => rmse)
end


best_solution, final_population, ph = evolutionary_algorithm(400, 1000, pc = 20)



plot(predict_time_series(best_solution, re)[:,1,1])
plot!(re[2:end,1,1])