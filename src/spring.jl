using DifferentialEquations, Optim, Plots
ts = reshape(ts, 21, 5, 10)
pt[:,10]

# Observed data
t_obs = 1:11  # Time steps 11 to 21
X_obs = tsode[11:21,1,1]

# Spring model (2nd-order differential equation)
function spring!(du, u, p, t)
    X, dX = u
    β, ω0, X_steady = p
    du[1] = dX
    du[2] = -β * dX - ω0^2 * (X - X_steady)
end

# Loss function to optimize
function loss(params)
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
p = plot(t_obs .+ 10, X_obs, label="Observed", marker=:circle, lw = 4)
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





# # Initial conditions
# x_0 = X_obs[1]          # Initial position (from observed data)
# v_0 = 0.0               # Initial velocity
# Δt = 0.1                # Time step size
# num_steps = 100         # Number of steps to predict


# function predict_motion_rk4(x_0, v_0, β, ω0, X_steady, Δt, num_steps)
#     # Initialize arrays to store results
#     x_values = [x_0]
#     v_values = [v_0]

#     # Initialize current position and velocity
#     x_t = x_0
#     v_t = v_0

#     # Predict step-by-step
#     for step in 1:num_steps
#         x_t, v_t = predict_next_step_rk4(x_t, v_t, β, ω0, X_steady, Δt)
#         push!(x_values, x_t)
#         push!(v_values, v_t)
#     end

#     # Generate time points
#     t_values = (0:num_steps) .* Δt

#     return t_values, x_values, v_values
# end

# # Call the function to predict motion
# t_values, x_values, v_values = predict_motion_rk4(x_0, v_0, β_opt, ω0_opt, X_steady_opt, Δt, num_steps)

# plot!(t_values.+10, x_values, label="RK4", xlabel="Time", ylabel="gene value", lw =4)
# savefig("figs\\g1(unregulated)_SHM.png")

display(p)



