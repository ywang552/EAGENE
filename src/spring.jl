using DifferentialEquations, Optim, Plots
ts = reshape(ts, 21, 5, 10)
tsode = reshape(tsode, 21, 5, 10)
# Observed data
t_obs = 1:11  # Time steps 11 to 21
X_obs = ts[1:11,5,1]

pt

# Spring model (2nd-order differential equation)
function spring!(du, u, p, t)
    X, dX = u
    β, ω0, X_steady = p
    du[1] = dX
    du[2] = - β * dX - ω0^2 * (X - X_steady)
end

# Loss function to optimize
function loss(params)
    u0 = [X_obs[1], 0.0]  # Initial conditions: [X(0), dX(0)]
    tspan = (0.0, 20.0)
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
p = plot(t_obs .+ 10, X_obs, label="Observed", marker=:circle, lw = 4, ylims = [0, 1])
plot!(t_obs .+ 10, X_pred, label="solver (RK4)", linestyle=:dash, lw = 4)
xlabel!("Time")
ylabel!("Gene 1 Expression")










display(p)



