using Colors  # For color utilities

function visualize_family_tree(tree::Dict{Int, Tuple{Int, Int}}, chromosome_data::Dict{Int, Chromosome})
    g = SimpleDiGraph()
    id_to_node = Dict{Int, Int}()

    # Add nodes and edges
    for (id, parents) in tree
        if !haskey(id_to_node, id)
            id_to_node[id] = add_vertex!(g)
        end
        for parent in parents
            if parent != -1  # Ignore invalid parents
                if !haskey(id_to_node, parent)
                    id_to_node[parent] = add_vertex!(g)
                end
                add_edge!(g, id_to_node[parent], id_to_node[id])
            end
        end
    end

    # Generate node labels
    node_labels = [string(id, "\nGen:", chromosome_data[id].generation, "\nFit:", round(chromosome_data[id].fitness, digits=2)) for id in keys(tree)]

    # Precompute positions: separate x and y coordinates
    loc_x = [chromosome_data[id].generation for id in keys(tree)]  # X-coordinates based on generation
    loc_y = [id for id in keys(tree)]  # Y-coordinates based on ID or other attribute

    # Define node colors (e.g., generation-based coloring)
    node_colors = [RGB(0.3, 0.6, 0.9) for _ in keys(tree)]  # Light blue for all nodes

    # Define edge colors
    edge_colors = [RGB(0.2, 0.2, 0.2) for _ in edges(g)]  # Dark gray for edges

    # Plot the graph
    z = gplot(
        g,
        loc_x, loc_y,
        nodelabel=node_labels,
        title="Family Tree of Chromosomes",
        nodesize=0.5,
        nodefillc=node_colors,
        edgestrokec=edge_colors
    )
    z
end

# function build_family_tree(all_chromosomes::Dict{Int, Chromosome})
#     tree = Dict{Int, Tuple{Int, Int}}()
#     for (id, chrom) in all_chromosomes
#         tree[id] = chrom.parents
#     end
#     return tree
# end

# tree = Dict(
#     1 => (-1, -1),
#     2 => (1, -1),
#     3 => (1, 2),
#     4 => (2, 3)
# )

# # Example chromosome data
# chromosome_data = Dict(
#     1 => Chromosome(rand(10), rand(10), sprand(Float64, 10, 10, 0.1), -Inf, (-1, -1), 0, 1),
#     2 => Chromosome(rand(10), rand(10), sprand(Float64, 10, 10, 0.1), -Inf, (1, -1), 1, 2),
#     3 => Chromosome(rand(10), rand(10), sprand(Float64, 10, 10, 0.1), -Inf, (1, 2), 2, 3),
#     4 => Chromosome(rand(10), rand(10), sprand(Float64, 10, 10, 0.1), -Inf, (2, 3), 3, 4)
# )


