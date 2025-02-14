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

using Statistics, DataFrames

# Function to normalize a gene's time series using Z-score
function normalize_time_series(X_row)
    min_val = minimum(X_row)
    max_val = maximum(X_row)
    
    if max_val == min_val  # Avoid division by zero
        return fill(0.5, length(X_row))  # Set all values to 0.5 if constant
    end

    return (X_row .- min_val) ./ (max_val - min_val)  # Normalize between 0 and 1
end

# Normalize all genes' time series
normalized_trial_1_matrix = [normalize_time_series(filtered_trial_1_matrix[i, :]) for i in 1:num_genes]
normalized_trial_2_matrix = [normalize_time_series(filtered_trial_2_matrix[i, :]) for i in 1:num_genes]
normalized_trial_3_matrix = [normalize_time_series(filtered_trial_3_matrix[i, :]) for i in 1:num_genes]

# Function to compute inhibition variance within a gene
function compute_gene_inhibition_variance(X_row)
    ΔX = diff(X_row)  # Compute expression changes
    inhibitory_changes = filter(x -> x < 0, ΔX)  # Keep only inhibitory drops

    return isempty(inhibitory_changes) ? NaN : var(inhibitory_changes)  # Compute variance, return NaN if no inhibition
end

gene_inhibition_variance = [compute_gene_inhibition_variance(normalized_trial_1_matrix[i]) for i in 1:num_genes]

# Store results in a DataFrame
df_variance = DataFrame(Gene=1:num_genes, Inhibition_Variance=gene_inhibition_variance)
# Remove NaN values from inhibition variance calculations
valid_variance_values = filter(!isnan, df_variance.Inhibition_Variance)

# Compute statistics without NaN values
println("Summary of Normalized Inhibition Variance Across Genes (Filtered):")
println("Mean Variance: ", mean(valid_variance_values))
println("Median Variance: ", median(valid_variance_values))
println("Variance of Variance: ", var(valid_variance_values))

using KernelDensity, Plots

# Compute density estimate only on valid values
density_estimate = kde(valid_variance_values)

# Plot density
plot(density_estimate.x, density_estimate.density, linewidth=2, xlabel="Normalized Inhibition Variance",
     ylabel="Density", title="Density Plot of Normalized Inhibition Variance Across Genes")
println("Min Variance: ", minimum(valid_variance_values))
println("Max Variance: ", maximum(valid_variance_values))


low_variance_genes = count(x -> x < 0.05, valid_variance_values)
println("Number of genes with extremely low inhibition variance: ", low_variance_genes)
println("Percentage of genes with low inhibition variance: ", low_variance_genes / length(valid_variance_values) * 100, "%")
using Plots

histogram(valid_variance_values, bins=30, xlabel="Variance of Inhibition", ylabel="Frequency",
          title="Distribution of Inhibition Variance Across Genes")

          using StatsPlots

# Compute mean expression level for each gene
gene_mean_expression = [mean(filtered_trial_1_matrix[i, :]) for i in 1:num_genes]

# Scatter plot inhibition variance vs. gene expression level
scatter(gene_mean_expression, valid_variance_values, xlabel="Mean Gene Expression Level",
        ylabel="Inhibition Variance", title="Self-Inhibition Variance vs. Gene Expression",
        alpha=0.6, marker=:o)



        using Statistics

# Function to extract inhibition magnitudes and corresponding expression levels
function get_inhibition_data(X_row)
    ΔX = diff(X_row)  # Compute expression changes
    inhibition_times = findall(x -> x < 0, ΔX)  # Identify inhibition time steps
    inhibition_magnitudes = ΔX[inhibition_times]  # Get inhibition values
    expression_levels = X_row[inhibition_times]  # Get gene expression at inhibition times

    return (inhibition_times, inhibition_magnitudes, expression_levels)
end

# Apply function to all genes
gene_inhibition_data = [get_inhibition_data(normalized_trial_1_matrix[i]) for i in 1:num_genes]

# Flatten data across all genes
all_inhibition_magnitudes = vcat([gene_inhibition_data[i][2] for i in 1:num_genes]...)
all_expression_levels = vcat([gene_inhibition_data[i][3] for i in 1:num_genes]...)

# Print example results
println("Total inhibition events: ", length(all_inhibition_magnitudes))
println("Example inhibition magnitudes: ", all_inhibition_magnitudes[1:5])
println("Example expression levels at inhibition: ", all_expression_levels[1:5])

# Compute Pearson correlation coefficient
correlation_value = cor(all_expression_levels, all_inhibition_magnitudes)

println("Correlation between gene expression level and inhibition magnitude: ", correlation_value)


using StatsPlots, Random

# Sample 500 random inhibition events for visualization
num_samples = min(500, length(all_expression_levels))  # Ensure we don’t sample more than available data
sample_indices = randperm(length(all_expression_levels))[1:num_samples]

sampled_expression_levels = all_expression_levels[sample_indices]
sampled_inhibition_magnitudes = all_inhibition_magnitudes[sample_indices]

# Scatter plot for sampled points
scatter(sampled_expression_levels, sampled_inhibition_magnitudes, xlabel="Gene Expression Level at Inhibition",
        ylabel="Inhibition Magnitude", title="Correlation Between Gene Expression and Inhibition Strength (Sampled)",
        alpha=0.6, marker=:o)



using Statistics

# Function to extract inhibition magnitudes and corresponding expression levels from raw data
function get_inhibition_data_raw(X_row)
    ΔX = diff(X_row)  # Compute expression changes
    inhibition_times = findall(x -> x < 0, ΔX)  # Identify inhibition time steps
    inhibition_magnitudes = ΔX[inhibition_times]  # Get inhibition values
    expression_levels = X_row[inhibition_times]  # Get gene expression at inhibition times

    return (inhibition_times, inhibition_magnitudes, expression_levels)
end
# Apply function to all genes using **unnormalized data**
gene_inhibition_data_raw = [get_inhibition_data_raw(filtered_trial_1_matrix[i,:]) for i in 1:num_genes]

# Flatten data across all genes
all_inhibition_magnitudes_raw = vcat([gene_inhibition_data_raw[i][2] for i in 1:num_genes]...)
all_expression_levels_raw = vcat([gene_inhibition_data_raw[i][3] for i in 1:num_genes]...)

# Compute correlation using **unnormalized** data
correlation_raw = cor(all_expression_levels_raw, all_inhibition_magnitudes_raw)

println("Correlation between gene expression level and inhibition magnitude (Unnormalized): ", correlation_raw)
using StatsPlots, Random

# Sample 500 random inhibition events for visualization
num_samples = min(500, length(all_expression_levels_raw))
sample_indices = randperm(length(all_expression_levels_raw))[1:num_samples]

sampled_expression_levels_raw = all_expression_levels_raw[sample_indices]
sampled_inhibition_magnitudes_raw = all_inhibition_magnitudes_raw[sample_indices]

# Scatter plot for sampled points using raw data
scatter(sampled_expression_levels_raw, sampled_inhibition_magnitudes_raw, xlabel="Gene Expression Level (Unnormalized)",
        ylabel="Inhibition Magnitude", title="Correlation Between Gene Expression and Inhibition Strength (Unnormalized, Sampled)",
        alpha=0.6, marker=:o)


# Function to extract inhibition events from a gene's time series
function extract_inhibition_events(X_row)
    ΔX = diff(X_row)  # Compute expression changes
    inhibition_times = findall(x -> x < 0, ΔX)  # Identify inhibition time steps
    inhibition_values = ΔX[inhibition_times]  # Extract inhibition magnitudes
    expression_levels = X_row[inhibition_times]  # Expression levels at inhibition points

    return (inhibition_times, inhibition_values, expression_levels)
end




# Apply to all genes
gene_inhibition_data = [extract_inhibition_events(filtered_trial_1_matrix[i,:]) for i in 1:num_genes]

