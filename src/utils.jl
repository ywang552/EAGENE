import Base: show
using SparseArrays, Plots, StatsBase



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

    # # Print sparse matrix non-zero values with limited precision
    # for (i, j, val) in zip(chromosome.W.rowval, chromosome.W.colval, chromosome.W.nzval)
    #     println(io, "    [$i, $j] = ", round(val, digits=4))
    # end
end

mutable struct Chromosome
    α::Vector{Float64}        # Self-regulation values
    X_steady::Vector{Float64} # Steady-state values
    W::SparseMatrixCSC{Float64, Int} # Gene regulation matrix
    fitness::Float64          # Fitness score
    parents::Tuple{Int, Int}  # Indices of parents (-1, -1 for initial population)
    generation::Int           # The generation this chromosome belongs to
    id::Int                   # Unique ID for the chromosome
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

        push!(population, Chromosome(α, X_steady, W, fitness, parents, generation, id))
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
            for i in 1:N
                # Predicted next state for gene i
                X_pred = X_trial[t, i] + α[i] * (X_steady[i] - X_trial[t, i]) + sum(W[i, :] .* X_trial[t, :])

                # Exponential weighting loss
                loss = exp(k_exp * abs(X_trial[t+1, i] - X_pred)) - 1
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

function visualize_fitness_history(fitness_history)
    generations = 1:length(fitness_history[:best])
    p = plot(
        generations,
        fitness_history[:best],
        label="Best Fitness",
        xlabel="Generation",
        ylabel="Fitness",
        title="Fitness Evolution",
        linewidth=2
    )
    plot!(
        generations,
        fitness_history[:average],
        label="Average Fitness",
        linewidth=2,
        linestyle=:dash
    )
    p
end


function evaluate_population_exponential!(population::Vector{Chromosome}; X_obs = re, nnz_value = nnz_value,
                                           time_steps_per_trial = 11, trials = 5, k_exp = 2)
    for chromosome in population
        evaluate_fitness_exponential!(chromosome, X_obs = X_obs,  nnz_value = nnz_value,
                                       time_steps_per_trial = time_steps_per_trial, 
                                       trials = trials, k_exp = k_exp)
    end
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
        get_next_id()  # Assign unique ID to the offspring
    )

    return offspring
end

function sample_distinct_parents(parents::Vector{Chromosome})
    # Sample two distinct parents
    parent1 = rand(parents)
    parent2 = rand(filter(p -> p.id != parent1.id, parents))
    return parent1, parent2
end


function evolutionary_algorithm(
    population_size, generations;
    X_obs = re, N = N, nnz_value = nnz_value, crossover_rate=0.7, mutation_rate=0.1,
    time_steps_per_trial=11, trials=5, k_exp=2.0
    )


    population = initialize_population(population_size, N, nnz_value)

    # Fitness history for visualization
    fitness_history = Dict(:best => Float64[], :average => Float64[])

    for gen in 1:generations
        println("Generation $gen")

        # Step 2: Evaluate fitness for the population
        evaluate_population_exponential!(population; 
                                         X_obs=X_obs, nnz_value=nnz_value,
                                         time_steps_per_trial=time_steps_per_trial, 
                                         trials=trials, k_exp=k_exp)

        # Track fitness
        track_fitness(population, fitness_history)

        # Sort population by fitness (descending order)
        population = sort(population, by=c -> c.fitness, rev=true)

        # Print the best fitness
        println("Best fitness: ", population[1].fitness)

        # Step 3: Selection - Retain the top half of the population
        num_parents = div(population_size, 2)
        parents = population[1:num_parents]

        # Step 4: Recombination - Generate offspring
        offspring = Chromosome[]
        for _ in 1:(population_size - num_parents)
            # Randomly select two parents
            parent1, parent2 = sample_distinct_parents(parents)            
            push!(offspring, recombine_chromosomes(parent1, parent2, gen))
        end

        # Step 5: Mutation - Apply mutation to offspring
        for child in offspring
            mutate_chromosome!(child, mutation_rate)
        end

        # Step 6: Repair - Ensure offspring satisfy constraints
        for child in offspring
            check_and_fix_chromosome!(child)
        end

        # Replace the old population with parents and offspring
        population = vcat(parents, offspring)
    end
    # Step 7: Visualize fitness history
    display(visualize_fitness_history(fitness_history))

    # Return the best chromosome after all generations
    return population[1]
end

function predict_next_states(chromosome::Chromosome, X_input::Matrix{Float64})
    """
    Predicts the next states for the input time series using the chromosome's parameters.
    
    Args:
        chromosome::Chromosome: The chromosome containing α, X_steady, and W.
        X_input::Matrix{Float64}: A matrix of current states (time steps × genes).
    
    Returns:
        Matrix{Float64}: Predicted states for the next time step (time steps × genes).
    """
    N = size(X_input, 2)  # Number of genes
    T = size(X_input, 1)  # Number of time steps

    # Extract parameters
    α = chromosome.α
    X_steady = chromosome.X_steady
    W = chromosome.W

    # Initialize predictions
    X_pred = similar(X_input)

    # Predict each time step
    for t in 1:T
        for i in 1:N
            # Compute next state for gene i
            regulation_effect = sum(W[i, :] .* X_input[t, :])
            X_pred[t, i] = X_input[t, i] + α[i] * (X_steady[i] - X_input[t, i]) + regulation_effect
        end
    end

    return X_pred
end


pop = initialize_population(100, N, nnz_value)
s = evolutionary_algorithm(5, 100)

pred = predict_next_states(s, re[1:10,:])
cur = re[1:10,:]
w = s.W


g = 7 
w[:,g]
plot(pred[:,g])
plot!(cur[:,g])