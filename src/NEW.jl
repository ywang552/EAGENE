using CSV
using DataFrames
using SparseArrays, StatsBase
using Plots


df = CSV.read(joinpath("data", "gene_raw.csv"), DataFrame; delim='\t')
col_names = split(names(df)[1], ",")
# Assuming `col_names` is already defined as a vector of column names
trial_1_indices = []
trial_2_indices = []
trial_3_indices = []

trial_1_names = []
trial_2_names = []
trial_3_names = []

# Iterate through column names with their indices to group trials
for (index, col_name) in enumerate(col_names)
    if endswith(col_name, "_1")
        push!(trial_1_indices, index)
        push!(trial_1_names, col_name)
    elseif endswith(col_name, "_2")
        push!(trial_2_indices, index)
        push!(trial_2_names, col_name)
    elseif endswith(col_name, "_3")
        push!(trial_3_indices, index)
        push!(trial_3_names, col_name)
    end
end

# Assuming `df` is a DataFrame with 15744 rows and each row contains gene name + trial data
data_column = df[:, 1]  # Extract the single column (assumes it's a Vector of Strings)

# Initialize containers for the separated trials
num_genes = size(df, 1)
num_time_steps = length(trial_1_indices)
gene_names = Vector{String}(undef, num_genes)
trial_1_matrix = Array{Float64}(undef, num_genes, num_time_steps)
trial_2_matrix = Array{Float64}(undef, num_genes, num_time_steps + 1)
trial_3_matrix = Array{Float64}(undef, num_genes, num_time_steps + 1)

# Process each row to separate gene names and trial data
for i in 1:num_genes
    # Split the comma-separated data
    row_data = split(data_column[i], ",")
    
    # First value is the gene name
    gene_names[i] = row_data[1]
    
    # Extract trial data using dynamically computed indices
    trial_1_matrix[i, :] = parse.(Float64, row_data[trial_1_indices])
    trial_2_matrix[i, :] = parse.(Float64, row_data[trial_2_indices])
    trial_3_matrix[i, :] = parse.(Float64, row_data[trial_3_indices])
end

# Now you have:
# - `gene_names` as a vector of gene names
# - `trial_1_matrix`, `trial_2_matrix`, and `trial_3_matrix` containing the time series data
println("Gene Names: ", gene_names[1:5])  # Example to check the first few names
println("Trial 1 Matrix (first row): ", trial_1_matrix[1, :])
println("Trial 2 Matrix (first row): ", trial_2_matrix[1, :])
println("Trial 3 Matrix (first row): ", trial_3_matrix[1, :])

using Statistics, Plots
all_values = vcat(trial_1_matrix[:], trial_2_matrix[:], trial_3_matrix[:])

# Calculate key statistics
mean_val = mean(all_values)
median_val = median(all_values)
min_val = minimum(all_values)
max_val = maximum(all_values)
std_val = std(all_values)
println("Mean: ", mean_val)
println("Median: ", median_val)
println("Min: ", min_val)
println("Max: ", max_val)
println("Standard Deviation: ", std_val)
using Statistics, Plots

using Statistics, Plots

# Flatten all values into a single vector
all_values = vcat(trial_1_matrix[:], trial_2_matrix[:], trial_3_matrix[:])

# Count zeros
num_zeros = count(x -> x == 0, all_values)
num_nonzeros = count(x -> x > 0, all_values)
zero_percentage = num_zeros / length(all_values) * 100

println("Total Values: ", length(all_values))
println("Number of Zeros: ", num_zeros)
println("Number of Nonzeros: ", num_nonzeros)
println("Percentage of Zeros: ", zero_percentage, "%")

threshold = 58

# Find indices of genes that meet the threshold in at least one time step
valid_gene_indices = []
for i in 1:num_genes
    if maximum(vcat(trial_1_matrix[i, :], trial_2_matrix[i, :], trial_3_matrix[i, :])) >= threshold
        push!(valid_gene_indices, i)
    end
end

# Apply the filter
filtered_gene_names = gene_names[valid_gene_indices]
filtered_trial_1_matrix = trial_1_matrix[valid_gene_indices, :]
filtered_trial_2_matrix = trial_2_matrix[valid_gene_indices, :]
filtered_trial_3_matrix = trial_3_matrix[valid_gene_indices, :]

# Print results
println("Threshold Used: ", threshold)
println("Number of Genes Before Filtering: ", num_genes)
println("Number of Genes After Filtering: ", length(valid_gene_indices))
println("Example Filtered Gene Names: ", filtered_gene_names[1:5])  # Preview first 5

plot(filtered_trial_1_matrix[1,:])
num_genes = size(filtered_trial_1_matrix, 1)

using Statistics

function classify_inhibition(X_row)
    ΔX = diff(X_row)  # Compute expression changes
    inhibitory_changes = filter(x -> x < 0, ΔX)  # Keep only inhibitory drops

    if isempty(inhibitory_changes)
        return (0, 0, 0)  # No inhibition events
    end

    # Define classification thresholds based on percentiles
    p25 = quantile(inhibitory_changes, 0.25)
    p75 = quantile(inhibitory_changes, 0.75)

    # Count occurrences in each category
    num_small = count(x -> x < p25, inhibitory_changes)
    num_medium = count(x -> p25 ≤ x ≤ p75, inhibitory_changes)
    num_large = count(x -> x > p75, inhibitory_changes)

    return (num_small, num_medium, num_large)
end

# Apply classification to all genes
inhibition_classifications = [classify_inhibition(filtered_trial_1_matrix[i, :]) for i in 1:num_genes]


# Convert to DataFrame for analysis
df_inhibition = DataFrame(
    Gene = 1:num_genes,
    Small_Inhibition = [x[1] for x in inhibition_classifications],
    Medium_Inhibition = [x[2] for x in inhibition_classifications],
    Large_Inhibition = [x[3] for x in inhibition_classifications]
)


all_inhibitory_changes = vcat([diff(filtered_trial_1_matrix[i, :]) for i in 1:num_genes]...)  # Flatten inhibition changes
all_inhibitory_changes = filter(x -> x < 0, all_inhibitory_changes)  # Keep only inhibitory drops

# Compute variance
inhibition_variance = var(all_inhibitory_changes)

println("Variance of Inhibitory Changes Across Genes: ", inhibition_variance)
