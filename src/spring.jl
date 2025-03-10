using DifferentialEquations, Optim, Plots


using FFTW
function estimate_omega0_fft(time_series, time_step)
    n = length(time_series)
    
    # Apply FFT and get power spectrum
    freq = fftfreq(n, 1/time_step)  # Frequency bins
    power_spectrum = abs.(fft(time_series)).^2  # Power
    
    # Identify dominant frequency (ignoring zero frequency)
    dominant_idx = argmax(power_spectrum[2:end]) + 1  # Ignore DC component
    f0 = abs(freq[dominant_idx])
    
    return 2π * f0  # Convert frequency to ω0
end

function estimate_omega0_autocorr(time_series, time_step)
    autocorr_values = autocor(time_series, 1:round(Int, length(time_series)/2))
    
    # Find the first peak in the autocorrelation function
    peak_idx = findmax(autocorr_values[2:end])[2] + 1
    T_est = peak_idx * time_step  # Estimate period
    
    return 2π / T_est  # Convert period to ω₀
end
estimate_omega0_fft(X_obs, 36)
estimate_omega0_autocorr(X_obs, 36)



# ts = reshape(ts, 21, 5, 10)
# tsode = reshape(tsode, 21, 5, 10)
# Observed data
t_obs = 1:z  # Time steps 11 to 21
X_obs = filtered_trial_1_matrix[g,:]

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
    tspan = (0.0, z-1)
    prob = ODEProblem(spring!, u0, tspan, params)
    sol = solve(prob, saveat=1.0)
    X_pred = [sol[i][1] for i in 1:length(t_obs)]
    return sqrt(mean((X_obs .- X_pred).^2))
end

# Initial guesses for parameters
initial_params = [0.5, estimate_omega0_autocorr(X_obs, 36), median(X_obs)]  # β, ω0, X_steady
initial_params = [0.5, estimate_omega0_autocorr(X_obs, 36), median(X_obs)]  # β, ω0, X_steady

# Optimize the parameters
result = optimize(loss, initial_params, NelderMead())

# Extract optimized parameters
β_opt, ω0_opt, X_steady_opt = Optim.minimizer(result)

println("Optimized β: $β_opt, Optimized ω0: $ω0_opt, Optimized X_steady: $X_steady_opt")

# Solve the spring equation with optimized parameters
u0 = [X_obs[1], 0.0]
tspan = (0.0, z-1)
prob = ODEProblem(spring!, u0, tspan, [β_opt, ω0_opt, X_steady_opt])
sol = solve(prob, saveat=1.0)
X_pred = [sol[i][1] for i in 1:length(t_obs)]

# Plot observed vs predicted
p = plot(t_obs .+ 10, X_obs, label="Observed", marker=:circle, lw = 4)
plot!(t_obs .+ 10, X_pred, label="solver (RK4)", linestyle=:dash, lw = 4)
xlabel!("Time")
ylabel!("Gene 1 Expression")

X_obs
X_pred

# println("X_obs:", X_obs)
# println("X_pred:", X_pred)


# plot(X_obs .- X_pred)
# println(round.(X_obs .- X_pred,digits = 3))
# display(p)

# savefig("figs\\g$(g)_SHM.png")



