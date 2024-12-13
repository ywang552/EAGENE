ts = reshape(ts, 21, 5, 10)

X_0 = ts[12,1,1]

function predict_next_step(x_t, v_t, X_steady, beta, ω0, Δt)
    # Compute acceleration term
    acceleration = -2 * beta * v_t - ω0^2 * (x_t - X_steady)
    
    # Euler's method for position and velocity
    x_next = x_t + v_t * Δt
    v_next = v_t + acceleration * Δt

    return x_next, v_next
end

# Parameters
x_0 = 0.1170103                 # Initial position
v_0 = 0.0                       # Initial velocity
X_steady = -0.19236767843809982 # Steady-state position
beta = -2.6251781633999887      # Damping coefficient
ω0 = -8.58072962595166e-6       # Natural angular frequency
Δt = 1.0                        # Time step

# Parameters remain the same
X_steady = -0.19236767843809982 # Steady-state position
beta = -2.6251781633999887      # Damping coefficient
ω0 = -8.58072962595166e-6       # Natural angular frequency
Δt = 1.0                        # Time step

# New initial conditions
x_1 = 0.4708819                 # Updated position at step 1
v_1 = v_next                    # Velocity from the previous step

# Predict the next step
x_next2, v_next2 = predict_next_step(x_1, v_1, X_steady, beta, ω0, Δt)

println("Next position (step 2): $x_next2")
println("Next velocity (step 2): $v_next2")
