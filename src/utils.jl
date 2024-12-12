import Base: show
using SparseArrays, Plots, StatsBase
using GeometryBasics: Point2f
using Colors
using LinearAlgebra
using MultivariateStats


mutable struct Chromosome
    α::Vector{Float64}        # Self-regulation values
    X_steady::Vector{Float64} # Steady-state values
    W::SparseMatrixCSC{Float64, Int} # Gene regulation matrix
    fitness::Float64          # Primary fitness score (e.g., prediction error)
    parents::Tuple{Int, Int}  # Indices of parents (-1, -1 for initial population)
    generation::Int           # The generation this chromosome belongs to
    id::Int                   # Unique ID for the chromosome
    sparsity::Float16             # Number of non-zero elements in W (secondary objective)
    distance::Float64
end

import Base: isless
function isless(c1::Chromosome, c2::Chromosome)
    # Compare based on crowding distance (higher is better)
    # if c1.crowding_distance != c2.crowding_distance
    #     return c1.crowding_distance < c2.crowding_distance
    # end

    # Fallback comparison (e.g., based on fitness if distances are equal)
    # Assuming fitness is maximized (higher is better)
    return c1.fitness < c2.fitness
end



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
        push!(population, Chromosome(α, X_steady, W, fitness, parents, generation, id, sparsity, 0))
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


function evaluate_fitness_exponential!(chromosome::Chromosome; re = re, time_steps_per_trial = 11, trials = 5, k_exp = 0.5)
    # Extract chromosome parameters
    α = chromosome.α
    X_steady = chromosome.X_steady
    W = chromosome.W
    N = length(α)  # Number of genes

    # Precompute (I - W)
    I_minus_W = I - W

    # Precomputed reshaped matrix
    X_obs_reshaped = re  # Already reshaped during preprocessing
    X_current = X_obs_reshaped[1:end-1, :, :]  # Current state X_t
    X_next = X_obs_reshaped[2:end, :, :]      # Observed next state X_t+1

    # Compute predictions for all time steps and trials
    # regulation_term = (W) * reshape(X_current, (N, :))  # Compute in batch
    # regulation_term = reshape(regulation_term, size(X_current))  # Reshape back to 3D
    regulation_term =  I_minus_W \ reshape(X_current, (N, :))
    regulation_term = reshape(regulation_term, size(X_current))  # Reshape back to 3D

    # Add self-regulation and steady-state correction
    α_correction = α .* (X_steady .- X_current)  # Broadcasting α term
    X_pred = X_current + α_correction + regulation_term  # Predicted next state

    # Compute RMSD loss
    rmsd_normalizer = maximum(abs.(X_next)) - minimum(abs.(X_next))
    if rmsd_normalizer > 0
        loss_matrix = rmsd(X_pred, X_next) / rmsd_normalizer
    else
        loss_matrix = rmsd(X_pred, X_next)  # Fallback if range is 0
    end
    # Total fitness is negative RMSD (to maximize accuracy)
    total_fitness = loss_matrix

    steady_state_penalty = 0.1 * sum((chromosome.X_steady .- avgwt).^2)

    # Add penalty for extreme alpha values
    alpha_penalty = 0.1 * sum((chromosome.α .- 0.1).^2)

    # Total fitness
    chromosome.fitness = -(total_fitness + steady_state_penalty + alpha_penalty)
end

avgwt
println("α: $(p.α), X_steady: $(p.X_steady), ground truth X_steady: $(avgwt)")
p.X_steady


function evaluate_fitness_multiobjective!(chromosome::Chromosome; 
                                          re = re, time_steps_per_trial = 11, trials = 5, k_exp = 2,
                                          sparsity_epsilon = 1e-6)
    # Compute prediction error
    evaluate_fitness_exponential!(chromosome; re = re, time_steps_per_trial = time_steps_per_trial, trials = trials, k_exp = k_exp)

    # Compute and normalize sparsity
    sparsity = compute_sparsity(chromosome.W, sparsity_epsilon)
    sparsity /= prod(size(chromosome.W))  # Normalize by total elements in W

    # Update objectives
    chromosome.fitness = chromosome.fitness  # Prediction accuracy
    chromosome.sparsity = 1 * sparsity  # Smooth sparsity
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
    )
    p
end

function recombine_chromosomes_with_exploration(parent1::Chromosome, parent2::Chromosome, generation::Int, crossover_rate::Float64 = 0.5, mutation_rate::Float64 = 0.1)
    N = length(parent1.α)  # Number of genes

    # Weighted average for α with noise
    # blend_ratio = rand()
    # α_offspring = blend_ratio .* parent1.α .+ (1.0 - blend_ratio) .* parent2.α
    # α_offspring += 0.05 .* randn(N)  # Add Gaussian noise
    # α_offspring = clamp.(α_offspring, 0.01, 1.0)
    
    # # Weighted average for X_steady with noise
    # X_steady_offspring = blend_ratio .* parent1.X_steady .+ (1.0 - blend_ratio) .* parent2.X_steady
    # X_steady_offspring += 0.05 .* randn(N)  # Add Gaussian noise
    # X_steady_offspring = clamp.(X_steady_offspring, 0.0, 1.0)

    α_offspring = 0.5 .* (parent1.α .+ parent2.α)
    X_steady_offspring = 0.5 .* (parent1.X_steady .+ parent2.X_steady)


    W_offspring = spzeros(N, N)
    for i in 1:N
        if rand() < crossover_rate
            W_offspring[i, :] = parent1.W[i, :]
        else
            W_offspring[i, :] = parent2.W[i, :]
        end
    end

    # Create offspring
    offspring = Chromosome(
        α_offspring,
        X_steady_offspring,
        W_offspring,
        -1,  # Fitness not evaluated yet
        (parent1.id, parent2.id),
        generation,
        get_next_id(),
        nnz(W_offspring),
        0
    )

    return offspring
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

function non_dominated_sorting(population::Vector{Chromosome})
    # Initialize domination data
    domination_count = Dict{Chromosome, Int}()         # Number of solutions dominating each chromosome
    dominated_solutions = Dict{Chromosome, Set{Chromosome}}()  # Solutions dominated by each chromosome
    fronts = Vector{Vector{Chromosome}}()

    for chrom in population
        domination_count[chrom] = 0
        dominated_solutions[chrom] = Set{Chromosome}()
    end

    # Calculate domination relationships
    tolerance = 1e-6  # To handle precision issues
    for chrom1 in population
        for chrom2 in population
            if chrom1 != chrom2
                # Compare objectives with tolerance
                better_in_all = (chrom1.fitness + tolerance >= chrom2.fitness) &&
                                (chrom1.sparsity + tolerance <= chrom2.sparsity)
                strictly_better_in_at_least_one = (chrom1.fitness + tolerance > chrom2.fitness) ||
                                                  (chrom1.sparsity + tolerance < chrom2.sparsity)

                # Debugging output
                # println("Comparing Chromosome 1 (Fitness: $(chrom1.fitness), Sparsity: $(chrom1.sparsity))")
                # println("         with Chromosome 2 (Fitness: $(chrom2.fitness), Sparsity: $(chrom2.sparsity))")
                # println("Better in all: $better_in_all, Strictly better in at least one: $strictly_better_in_at_least_one")

                if better_in_all && strictly_better_in_at_least_one
                    push!(dominated_solutions[chrom1], chrom2)
                elseif (chrom2.fitness + tolerance >= chrom1.fitness) &&
                       (chrom2.sparsity + tolerance <= chrom1.sparsity) &&
                       ((chrom2.fitness + tolerance > chrom1.fitness) || 
                        (chrom2.sparsity + tolerance < chrom1.sparsity))
                    domination_count[chrom1] += 1
                end
            end
        end
    end

    # Assign to Pareto fronts
    current_front = Vector{Chromosome}()
    for chrom in population
        if domination_count[chrom] == 0
            push!(current_front, chrom)
        end
    end

    while !isempty(current_front)
        push!(fronts, current_front)
        next_front = Vector{Chromosome}()
        for chrom in current_front
            for dominated_chrom in dominated_solutions[chrom]
                domination_count[dominated_chrom] -= 1
                if domination_count[dominated_chrom] == 0
                    push!(next_front, dominated_chrom)
                end
            end
        end
        current_front = next_front
    end

    return fronts
end



function crowding_distance_assignment(front::Vector{Chromosome})
    num_individuals = length(front)
    
    # Handle edge case: small fronts
    if num_individuals <= 2
        # Assign infinite distance to all individuals
        for chrom in front
            chrom.distance = Inf
        end
        return
    end

    distances = zeros(Float64, num_individuals)
    objectives = [:fitness, :sparsity]

    # Sort by each objective
    for obj in objectives
        # Sort individuals by the current objective
        sorted_indices = sortperm([getfield(front[i], obj) for i in 1:num_individuals])

        # Boundary points get infinite distance
        distances[sorted_indices[1]] = Inf
        distances[sorted_indices[end]] = Inf

        # Compute distances for interior points
        obj_min = getfield(front[sorted_indices[1]], obj)
        obj_max = getfield(front[sorted_indices[end]], obj)

        if obj_max != obj_min
            for k in 2:(num_individuals - 1)
                numerator = getfield(front[sorted_indices[k + 1]], obj) - getfield(front[sorted_indices[k - 1]], obj)
                denominator = obj_max - obj_min
                distances[sorted_indices[k]] += numerator / denominator
            end 
        end
    end

    # Assign distances to the individuals
    for i in 1:num_individuals
        front[i].distance = distances[i]
    end
end


function compute_sparsity(W::SparseMatrixCSC{Float64, Int64}, ε::Float64 = 1e-6)
    return sum(log.(1 .+ abs.(W) ./ ε))
end


using Plots

function plot_pareto_fronts(fronts::Vector{Vector{Chromosome}})
    # Prepare the plot
    p = plot(title="Pareto Fronts", xlabel="Fitness (Objective 1)", ylabel="Sparsity (Objective 2)", legend=false)

    # Assign unique colors to each Pareto front
    for (i, front) in enumerate(fronts)
        # Extract and sort points in the current Pareto front by fitness (Objective 1)
        sorted_front = sort(front, by = c -> c.fitness)

        # Get x and y values for the sorted front
        x = [chrom.fitness for chrom in sorted_front]  # Objective 1: Fitness
        y = [chrom.sparsity for chrom in sorted_front]  # Objective 2: Sparsity

        # Plot the Pareto front as a line with markers
        plot!(x, y, label="Front $i", marker=:circle, lw=2)
    end

    p
end

function normalize_objectives!(fronts::Vector{Vector{Chromosome}})
    # Extract global ranges for fitness and sparsity
    min_fitness = minimum([chrom.fitness for front in fronts for chrom in front])
    max_fitness = maximum([chrom.fitness for front in fronts for chrom in front])
    min_sparsity = minimum([chrom.sparsity for front in fronts for chrom in front])
    max_sparsity = maximum([chrom.sparsity for front in fronts for chrom in front])

    # Normalize each chromosome's objectives
    for front in fronts
        for chrom in front
            chrom.fitness = (chrom.fitness - min_fitness) / (max_fitness - min_fitness + 1e-6)
            chrom.sparsity = (chrom.sparsity - min_sparsity) / (max_sparsity - min_sparsity + 1e-6)
        end
    end
end


function nsga2(
    population_size::Int, generations::Int;
    X_obs=re, N=N, nnz_value=nnz_value, crossover_rate=0.7, mutation_rate=0.1,
    time_steps_per_trial=11, trials=5, k_exp=2.0, pc=50
)
    println("Starting NSGA-II...")

    # Step 1: Initialize Population
    population = initialize_population(population_size, N, nnz_value)

    # History Tracking
    pareto_front_history = Vector{Vector{Chromosome}}()
    fitness_history = Dict(:best => Float64[], :average => Float64[])

    for gen in 1:generations
        println("Generation $gen...")

        # Step 2: Evaluate Fitness for All Individuals
        for chromosome in population
            evaluate_fitness_multiobjective!(chromosome)
        end

        # Step 3: Generate Offspring
        offspring = Chromosome[]
        for _ in 1:population_size
            parent1, parent2 = sample_distinct_parents(population)
            child = recombine_chromosomes_with_exploration(parent1, parent2, gen)
            mutate_chromosome!(child, mutation_rate)
            check_and_fix_chromosome!(child)
            evaluate_fitness_multiobjective!(child)
            push!(offspring, child)
        end

        # Step 4: Combine Parent and Offspring Populations
        combined_population = vcat(population, offspring)

        # Step 5: Perform Non-Dominated Sorting
        fronts = non_dominated_sorting(combined_population)
        # Step 6: Assign Crowding Distances
        for front in fronts
            crowding_distance_assignment(front)
        end
        p = fronts[1]
        f = [x.fitness for x in p]
        # Step 7: Environmental Selection
        population = environmental_selection_simple(population_size, fronts)

        # Step 8: Track and Store Pareto Front
        @assert length(population) == population_size 

        push!(pareto_front_history, fronts[1])  # Store the best front
        # normalize_objectives!(fronts)
        track_fitness(population, fitness_history)

        # for (i, front) in enumerate(fronts)
        #     println("Front $i:")
        #     for chrom in front
        #         println("  Fitness: $(chrom.fitness), Sparsity: $(chrom.sparsity)")
        #     end
        # end
        

        # Visualization Every `pc` Generations
        if gen % pc == 0
            println("Visualizing progress at generation $gen...")
            p1 = (visualize_fitness_history(fitness_history))
            p2 = plot_pareto_fronts(fronts)
            display(plot(p1, p2))
        end
    end

    println("NSGA-II completed.")
    # Return final population and best Pareto front
    return population, pareto_front_history
end


function select_from_last_front_simple(last_front::Vector{Chromosome}, remaining_slots::Int)
    # Sort the front by crowding distance (descending), including Inf values
    sorted_front = sort(last_front, by=c -> c.distance, rev=true)

    # Select the top individuals up to the remaining slots
    return sorted_front[1:remaining_slots]
end

function environmental_selection_simple(population_size::Int, fronts::Vector{Vector{Chromosome}})
    new_population = Vector{Chromosome}()

    for front in fronts
        if length(new_population) + length(front) <= population_size
            # Add the entire front if it fits
            append!(new_population, front)
        else
            # If the front doesn’t fully fit, prioritize using crowding distance
            remaining_slots = population_size - length(new_population)

            # Sort the current front by crowding distance (descending)
            sorted_front = sort(front, by = c -> c.distance, rev = true)

            # Select the top individuals based on remaining slots
            append!(new_population, sorted_front[1:remaining_slots])
            break
        end
    end

    return new_population
end


final_population, ph = nsga2(200, 500, pc = 1)


Ws = [x.W for x in final_population]

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
sum(Ws[163].!=0)
findmin([sum(x.!=0) for x in Ws])
Ws[163].!=0 .&& gs.==0
# prd = predict_time_series(final_population[k], re)
# plot(prd[:,1,2])
# plot!(re[2:end,1,2])