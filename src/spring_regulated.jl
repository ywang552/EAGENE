using DifferentialEquations, Optim, Plots
# Observed time-series data
t_obs = 1:11  # Time steps
ts = reshape(ts, 21, 5, 10)

tsode = reshape(tsode, 21, 5, 10)
tssde = reshape(tssde, 21, 5, 10)
pt
g = 1
tri = 1
plot(ts[1:end,tri,g])
plot!(tsode[1:end,tri,g])
plot!(tssde[1:end,tri,g])
X_obs_2 = tsode[11:21, tri, g]  # Observed data for g2
X_inputs = tsode[11:21, tri, findall(x->x!=0,gs[:,g])]  # Observed data for g1, g6, g8

# Define ODE for g2 dynamics

function regulated_spring!(du, u, p, t)
    X2, dX2 = u  # Current state of g2
    β, ω0, X_steady_2, W... = p  # Parameters
    du[1] = dX2
    du[2] = -β * dX2 - ω0^2 * (X2 - X_steady_2) + X_inputs[Int(t),:]' * W
end

function regulated_repression!(du, u, p, t)
    X, dX = u  # Current state and derivative
    γ, ω0, X_steady, W, K, n = p  # Parameters
    
    # Regulation term
    R = 0.0
    for j in eachindex(W)
        R *= W[j] * (abs(K)^n / ((K)^n + (X_inputs[t, j])^n))
    end
    R = -abs(R)
    du[1] = dX
    du[2] = -2γ * dX - ω0^2 * (X - X_steady) + R
end

function loss_h(params)
    # β, ω0, X_steady_2, W1, W6, W8 = params
    γ, ω0, X_steady = params[1:3]
    W = params[4:6]
    K, n = params[7:8]

    u0 = [X_obs_2[1], 0.0]
    tspan = (1, 11)
    prob = DiscreteProblem(regulated_repression!, u0, tspan, [γ, ω0, X_steady, W, K, n])
    sol = solve(prob, saveat=1)
    X_pred = [sol[i][1] for i in 1:length(X_obs_2)]
    return sqrt(mean((X_obs_2 .- X_pred).^2))
end

function loss_spring(params)
    β, ω0, X_steady_2, W... = params
    u0 = [X_obs_2[1], 0.0]
    tspan = (1, 11)
    prob = DiscreteProblem(regulated_spring!, u0, tspan, [β, ω0, X_steady_2, W...])
    sol = solve(prob, saveat=1)
    X_pred = [sol[i][1] for i in 1:length(X_obs_2)]
    return sqrt(mean((X_obs_2 .- X_pred).^2))
end

initial_params_hill = [0.5, 1.0, 0.7,fill(-.1, (nnz(gs[:,g])))..., 0.5, 100.0]
initial_params_spring = [0.5, 1.0, 0.7,fill(-.1, (nnz(gs[:,g])))...]
inital_params = initial_params_spring

result = optimize(loss_spring, inital_params, NelderMead())
optimized_params = Optim.minimizer(result)
println("Optimized parameters: ", optimized_params)

optimized_params
# Solve the spring equation with optimized parameters
u0 = [X_obs_2[1], 0.0]
tspan = (0, 10)
prob = DiscreteProblem(regulated_spring!, u0, tspan, optimized_params)
sol = solve(prob, saveat=1.0)



X_pred = [sol[i][1] for i in 1:length(t_obs)]
X_obs_2


p = plot(t_obs .+ 10, X_obs_2, label="Observed", marker=:circle, lw = 4)
plot!(t_obs .+ 10, X_pred, label="solver (RK4)", linestyle=:dash, lw = 4)
xlabel!("Time")
ylabel!("Gene 1 Expression")
