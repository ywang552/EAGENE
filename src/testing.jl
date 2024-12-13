using SparseArrays, Optim

function objective(params::Vector{Float64}, gs::SparseMatrixCSC{Int}, X_obs::Matrix{Float64})
    N = size(gs, 1)
    T = size(X_obs, 2)

    # Extract parameters
    α = params[1:N]
    X_steady = params[N+1:2N]
    W_dense = reshape(params[2N+1:end], N, N)
    W = W_dense .* Matrix(gs)  # Reapply sparsity mask

    # Compute total loss
    total_loss = 0.0
    for t in 1:(T-1)
        X_current = X_obs[:, t]
        X_next = X_obs[:, t+1]

        # Predict next state
        X_pred = X_current .+ α .* (X_steady .- X_current) .+ W * X_current

        # Add MSE loss
        total_loss += sum((X_pred .- X_next).^2)
    end

    return total_loss
end


function initialize_weights_sparse(gs::SparseMatrixCSC{Int}, weight_range = (0.01:0.001:0.1))
    similar(gs)
end

W = initialize_weights_sparse(gs)

function initialize_params(gs::SparseMatrixCSC{Int})
    N = size(gs, 1)

    # Initialize α, X_steady, and W
    α = rand(0.01:0.01:0.1, N)
    X_steady = rand(0.1:0.1:1.0, N)
    W = initialize_weights_sparse(gs)

    # Flatten parameters into a dense vector
    params = vcat(α, X_steady, vec(Matrix(W)))
    return params
end

function fit_with_optim(gs::SparseMatrixCSC{Int}, X_obs::Matrix{Float64}, 
                        max_iters::Int = 1000)
    # Initialize parameters
    params = initialize_params(gs)

    # Define objective function
    obj_func = params -> objective(params, gs, X_obs)

    # Optimize using LBFGS
    result = optimize(obj_func, params, LBFGS())

    # Extract optimized parameters
    optimized_params = Optim.minimizer(result)
    N = size(gs, 1)
    α = optimized_params[1:N]
    X_steady = optimized_params[N+1:2N]
    W_dense = reshape(optimized_params[2N+1:end], N, N)
    W = W_dense .* Matrix(gs)  # Reapply sparsity mask

    return α, X_steady, W
end


function predict_one_step(X_obs::Matrix{Float64}, W::SparseMatrixCSC{Float64}, 
                          α::Vector{Float64}, X_steady::Vector{Float64})
    N, T = size(X_obs)
    X_pred = zeros(Float64, N, T)  # Store predictions
    X_pred[:, 1] = X_obs[:, 1]  # Use first time step as initial condition

    for t in 1:(T-1)
        X_pred[:, t+1] = X_pred[:, t] .+ α .* (X_steady .- X_pred[:, t]) .+ W * X_pred[:, t]
    end

    return X_pred
end


ts = reshape(ts, 21, 5, 10)
tsode = reshape(tsode, 21, 5, 10)
tssde = reshape(tssde, 21, 5, 10)
tssden = reshape(tssden, 21, 5, 10)
pt
println(gs[:,4])
g = 1
p1 = plot(ts[11:end,1,g], label = "sdenoisy")
plot!(tsode[11:end,1,g], label = "ode")
plot!(tssde[11:end,1,g], label = "sde")

p2 = plot(ts[1:end,1,g], label = "sdenoisy")
plot!(tsode[1:end,1,g], label = "ode")
plot!(tssde[1:end,1,g], label = "sde")
