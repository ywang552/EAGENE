using DifferentialEquations, Optim, Plots
ts = reshape(ts, 21, 5, 10)
pt[:,10]

# Observed data
t_obs = 1:11  # Time steps 11 to 21
X_obs = ts[11:21,5,10]

# Spring model (2nd-order differential equation)
function spring!(du, u, p, t)
    X, dX = u
    β, ω0, X_steady = p
    du[1] = dX
    du[2] = -β * dX - ω0^2 * (X - X_steady)
end

# Loss function to optimize
function loss(params)
    β, ω0, X_steady = params
    u0 = [X_obs[1], 0.0]  # Initial conditions: [X(0), dX(0)]
    tspan = (0.0, 10.0)
    prob = ODEProblem(spring!, u0, tspan, params)
    sol = solve(prob, saveat=1.0)
    X_pred = [sol[i][1] for i in 1:length(t_obs)]
    return sqrt(mean((X_obs .- X_pred).^2))
end

# Initial guesses for parameters
initial_params = [0.5, 1.0, 0.7]  # β, ω0, X_steady

# Optimize the parameters
result = optimize(loss, initial_params, NelderMead())

# Extract optimized parameters
β_opt, ω0_opt, X_steady_opt = Optim.minimizer(result)

println("Optimized β: $β_opt, Optimized ω0: $ω0_opt, Optimized X_steady: $X_steady_opt")

# Solve the spring equation with optimized parameters
u0 = [X_obs[1], 0.0]
tspan = (0.0, 10.0)
prob = ODEProblem(spring!, u0, tspan, [β_opt, ω0_opt, X_steady_opt])
sol = solve(prob, saveat=1.0)
X_pred = [sol[i][1] for i in 1:length(t_obs)]

# Plot observed vs predicted
plot(t_obs .+ 10, X_obs, label="Observed", marker=:circle, lw = 4)
plot!(t_obs .+ 10, X_pred, label="solver (RK4)", linestyle=:dash, lw = 4)
xlabel!("Time")
ylabel!("Gene 1 Expression")


# Optimized parameters (replace with actual values)
β_opt = β_opt             # Replace with the optimized β
ω0_opt = ω0_opt            # Replace with the optimized ω0
X_steady_opt = X_steady_opt      # Replace with the optimized X_steady

# Initial conditions
x_0 = X_obs[1]          # Initial position (from observed data)
v_0 = 0.0               # Initial velocity
Δt = 0.1                # Time step size
num_steps = 100         # Number of steps to predict

# Function to predict motion
function predict_motion(x_0, v_0, β, ω0, X_steady, Δt, num_steps)
    # Initialize arrays to store results
    x_values = [x_0]  # Store positions
    v_values = [v_0]  # Store velocities

    # Initialize current position and velocity
    x_t = x_0
    v_t = v_0

    # Predict step-by-step
    for step in 1:num_steps
        # Compute the next step
        x_t, v_t = predict_next_step(x_t, v_t, β, ω0, X_steady, Δt)
        # Store results
        push!(x_values, x_t)
        push!(v_values, v_t)
    end

    # Generate time points
    t_values = (0:num_steps) .* Δt

    return t_values, x_values, v_values
end

function predict_motion_rk4(x_0, v_0, β, ω0, X_steady, Δt, num_steps)
    # Initialize arrays to store results
    x_values = [x_0]
    v_values = [v_0]

    # Initialize current position and velocity
    x_t = x_0
    v_t = v_0

    # Predict step-by-step
    for step in 1:num_steps
        x_t, v_t = predict_next_step_rk4(x_t, v_t, β, ω0, X_steady, Δt)
        push!(x_values, x_t)
        push!(v_values, v_t)
    end

    # Generate time points
    t_values = (0:num_steps) .* Δt

    return t_values, x_values, v_values
end



# Initial conditions
x_0 = X_obs[1]          # Initial position (from observed data)
v_0 = 0.0               # Initial velocity
Δt = 0.1                # Time step size
num_steps = 100         # Number of steps to predict


function predict_motion_rk4(x_0, v_0, β, ω0, X_steady, Δt, num_steps)
    # Initialize arrays to store results
    x_values = [x_0]
    v_values = [v_0]

    # Initialize current position and velocity
    x_t = x_0
    v_t = v_0

    # Predict step-by-step
    for step in 1:num_steps
        x_t, v_t = predict_next_step_rk4(x_t, v_t, β, ω0, X_steady, Δt)
        push!(x_values, x_t)
        push!(v_values, v_t)
    end

    # Generate time points
    t_values = (0:num_steps) .* Δt

    return t_values, x_values, v_values
end

# Call the function to predict motion
t_values, x_values, v_values = predict_motion_rk4(x_0, v_0, β_opt, ω0_opt, X_steady_opt, Δt, num_steps)

# Plot the predicted motion
using Plots
plot!(t_values.+10, x_values, label="RK4", xlabel="Time", ylabel="gene value", lw =4)
# savefig("figs\\g1(unregulated)_SHM.png")



function regulated_spring!(du, u, p, t)
    X, dX = u                          # u[1] = X_i(t), u[2] = dX_i(t)/dt
    γ, ω0, X_steady, W, n, K = p       # Parameters for gene i
    regulation = 0.0

    for j in 1:length(W)  # Iterate over regulators
        if W[j] > 0       # Activation
            regulation += W[j] * (X^n / (K^n + X^n))
        elseif W[j] < 0   # Repression
            regulation += W[j] * (K^n / (K^n + X^n))
        end
    end

    du[1] = dX                        # First derivative: velocity
    du[2] = -2γ * dX - ω0^2 * (X - X_steady) + regulation  # Second derivative
end
function flatten_params(params, weights)
    flat_params = vcat([p[:] for p in params]...)
    flat_weights = vcat([w[:] for w in weights]...)
    return vcat(flat_params, flat_weights)
end

function reconstruct_params(flat_params, num_genes, weight_size)
    idx = 1
    params = []
    weights = []
    for _ in 1:num_genes
        push!(params, flat_params[idx:idx+2])  # Parameters: γ, ω0, X_steady
        idx += 3
    end
    for _ in 1:num_genes
        push!(weights, flat_params[idx:idx+weight_size-1])  # Weight vector for each gene
        idx += weight_size
    end
    return params, weights
end


function regulated_loss(params::Vector{Float64}, gs::SparseMatrixCSC{Int, Int}, X_obs::Matrix{Float64}, n::Int, K::Float64)

    num_genes = size(gs, 1)
    params = reconstruct_params(flat_params, num_genes)
    total_loss = 0.0

    for i in 1:num_genes
        γ, ω0, X_steady = params[i]
        W = gs[:, i] .* params[:, i]  # Extract weights for gene i
        u0 = [X_obs[1, i], 0.0]      # Initial conditions
        tspan = (0.0, 10.0)
        prob = ODEProblem(regulated_spring!, u0, tspan, [γ, ω0, X_steady, W, n, K])
        sol = solve(prob, saveat=1.0)
        X_pred = [sol[j][1] for j in 1:length(X_obs[:, i])]
        total_loss += sqrt(mean((X_obs[:, i] .- X_pred).^2))  # RMSE for gene i
    end

    return total_loss
end

# Flatten initial parameters
initial_params = [[0.5, 1.0, 0.7] for _ in 1:size(gs, 1)]  # γ, ω0, X_steady for each gene
flat_initial_params = flatten_params(initial_params)

result = optimize(
    params -> regulated_loss(params, gs, X_obs, n=2, K=0.5),
    flat_initial_params,
    NelderMead()
)

# Reconstruct optimized parameters
optimized_params = reconstruct_params(result.minimizer, size(gs, 1))


for i in 1:size(gs, 1)
    γ, ω0, X_steady = optimized_params[i]
    W = gs[:, i] .* params[:, i]
    u0 = [X_obs[1, i], 0.0]
    prob = ODEProblem(regulated_spring!, u0, (0.0, 10.0), [γ, ω0, X_steady, W, n=2, K=0.5])
    sol = solve(prob, saveat=1.0)
    X_pred = [sol[j][1] for j in 1:length(X_obs[:, i])]
    plot!(1:length(X_obs[:, i]), X_obs[:, i], label="Observed Gene $i")
    plot!(1:length(X_pred), X_pred, label="Predicted Gene $i", linestyle=:dash)
end

