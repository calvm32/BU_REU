# Ramped Cahn-Hilliard Simulation in Julia using ETDRK4
using MultiFloats
using GenericFFT
using AbstractFFTs
using LinearAlgebra

MultiFloats.use_bigfloat_transcendentals()

"""
Computes the critical bifurcation parameters for a given mode j.
"""
function critical_bifurcation(j, L, mass, mu0)
    KS = 2 * pi * j / L
    MUS = -mu0 + 2 * (3 * mass^2 + KS^2)
    return MUS, KS
end

# ─────────────────────────────────────────────────────────────────────────────
# Allocation-free helpers
# ─────────────────────────────────────────────────────────────────────────────

"""
Zero-allocation argmax of |x|² over a complex vector, skipping the first
element (DC / k=0 component).
"""
function argmax_abs2_skip1(arr::AbstractVector{<:Complex})
    max_val = abs2(arr[2])
    max_idx = 2
    @inbounds for i in 3:length(arr)
        v = abs2(arr[i])
        if v > max_val
            max_val = v
            max_idx = i
        end
    end
    return max_idx
end

"""
In-place FFT of a real vector: copies real data into the pre-allocated complex
buffer `dst` and applies fft! in-place.  Result is left in `dst`.
"""
function fft_real!(dst::Vector{Complex{T}}, src::Vector{T}) where {T}
    @inbounds for i in eachindex(dst, src)
        dst[i] = Complex{T}(src[i], zero(T))
    end
    fft!(dst)
end

"""
In-place IFFT: copies `src` into `scratch`, applies ifft! in-place, then
extracts real parts into `dst`.  `src` is preserved (not aliased with scratch).
"""
function ifft_to_real!(dst::Vector{T}, src::Vector{Complex{T}}, scratch::Vector{Complex{T}}) where {T}
    copyto!(scratch, src)
    ifft!(scratch)
    @inbounds for i in eachindex(dst)
        dst[i] = real(scratch[i])
    end
end

# ─────────────────────────────────────────────────────────────────────────────

"""
Computes the ETDRK4 integration coefficients using a complex contour integral.
"""
function compute_etdrk4_coefficients(L_operator, dt::T, ::Type{T}) where {T}
    M = 64  # Number of points on the contour circle
    Nx = length(L_operator)
    
    # Define roots of unity for the circle contour integration
    r = [exp(im * T(pi) * (T(k) - T(0.5)) / T(M)) for k in 1:M]
    
    # Pre-allocate coefficient arrays
    E  = zeros(T, Nx)
    E2 = zeros(T, Nx)
    Q  = zeros(T, Nx)
    f1 = zeros(T, Nx)
    f2 = zeros(T, Nx)
    f3 = zeros(T, Nx)
    
    # In-place evaluation for linear filters
    @. E  = exp(dt * L_operator)
    @. E2 = exp(dt * L_operator / 2)
    
    for i in 1:Nx
        z = dt * L_operator[i]
        
        sum_Q  = zero(Complex{T})
        sum_f1 = zero(Complex{T})
        sum_f2 = zero(Complex{T})
        sum_f3 = zero(Complex{T})
        
        for k in 1:M            # scalar iteration avoids allocating z .+ r
            lr = z + r[k]
            exp_lr_2 = exp(lr / 2)
            exp_lr   = exp(lr)
            
            sum_Q  += (exp_lr_2 - 1) / lr
            sum_f1 += (-4 - lr + exp_lr * (4 - 3*lr + lr^2)) / (lr^3)
            sum_f2 += (2 + lr + exp_lr * (-2 + lr)) / (lr^3)
            sum_f3 += (-4 - 3*lr - lr^2 + exp_lr * (4 - lr)) / (lr^3)
        end
        
        Q[i]  = dt * real(sum_Q / T(M))
        f1[i] = dt * real(sum_f1 / T(M))
        f2[i] = dt * real(sum_f2 / T(M))
        f3[i] = dt * real(sum_f3 / T(M))
    end
    
    return E, E2, Q, f1, f2, f3
end

"""
Allocation-free nonlinear forcing: N̂ = dealias .* Δ .* FFT(u³ − μu).
Uses `fft_scratch` as a pre-allocated complex buffer for in-place FFT.
"""
function compute_nonlinear_forcing_hat!(N_hat, u, mu, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
    @. u3_scratch = u * (u * u - mu)
    fft_real!(fft_scratch, u3_scratch)
    @. N_hat = dealias_mask * Laplacian_k * fft_scratch
end

"""
Optimized ETDRK4 Time Stepping Engine utilizing full in-place loop fusion with ramped mu.
"""
function solve_etdrk4!(
    u::Vector{T}, 
    num_time_steps::Int, 
    dt::T,
    mu0::T,
    ep::T,
    mass::T,
    Lx::T,
    kx::Vector{T},
    Laplacian_k::Vector{T}, 
    dealias_mask::Vector{T}, 
    E::Vector{T}, E2::Vector{T}, Q::Vector{T}, f1::Vector{T}, f2::Vector{T}, f3::Vector{T},
    l2_thr::T
) where {T}
    
    Nx = length(u)
    
    # Pre-allocated workspace (real)
    u3_scratch = zeros(T, Nx)
    a          = zeros(T, Nx)
    b          = zeros(T, Nx)
    c          = zeros(T, Nx)
    
    # Pre-allocated workspace (complex) for in-place FFT/IFFT
    fft_scratch  = zeros(Complex{T}, Nx)
    ifft_scratch = zeros(Complex{T}, Nx)
    
    Nu_hat = zeros(Complex{T}, Nx)
    Na_hat = zeros(Complex{T}, Nx)
    Nb_hat = zeros(Complex{T}, Nx)
    Nc_hat = zeros(Complex{T}, Nx)
    
    a_hat  = zeros(Complex{T}, Nx)
    b_hat  = zeros(Complex{T}, Nx)
    c_hat  = zeros(Complex{T}, Nx)
    
    # Initial FFT of u (one-time allocation for persistent state)
    fft_real!(fft_scratch, u)
    u_hat = copy(fft_scratch)
    
    # Seed dominant mode tracking using u_hat directly, skipping DC.
    max_index = argmax_abs2_skip1(u_hat)
    
    # dominate_modes stores pairs of (time, wavenumber)
    dominate_modes = Tuple{T, T}[(T(0.0), abs(kx[max_index]))]
    
    t_thr = T(-1.0)
    mu_thr = T(NaN)
    
    l2_norm = sqrt(Lx) * norm(u_hat) / Nx
    if l2_norm > l2_thr
        t_thr = T(0.0)
        mu_thr = mu0
    end

    for n in 1:num_time_steps
        t = T(n - 1) * dt
        tmid = t + dt / 2
        tend = t + dt
        
        mu_t = mu0 + t * ep
        mu_tmid = mu0 + tmid * ep
        mu_tend = mu0 + tend * ep
        
        # ETDRK4 In-place Step Formulation
        
        # Stage 1
        compute_nonlinear_forcing_hat!(Nu_hat, u, mu_t, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
        
        # Stage 2 (Predictor step 'a')
        @. a_hat = E2 * u_hat + Q * Nu_hat
        ifft_to_real!(a, a_hat, ifft_scratch)
        compute_nonlinear_forcing_hat!(Na_hat, a, mu_tmid, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
        
        # Stage 3 (Predictor step 'b')
        @. b_hat = E2 * u_hat + Q * Na_hat
        ifft_to_real!(b, b_hat, ifft_scratch)
        compute_nonlinear_forcing_hat!(Nb_hat, b, mu_tmid, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
        
        # Stage 4 (Predictor step 'c')
        @. c_hat = E2 * a_hat + Q * (2 * Nb_hat - Nu_hat)
        ifft_to_real!(c, c_hat, ifft_scratch)
        compute_nonlinear_forcing_hat!(Nc_hat, c, mu_tend, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
        
        # Final Correction Step
        @. u_hat = E * u_hat + f1 * Nu_hat + 2 * f2 * (Na_hat + Nb_hat) + f3 * Nc_hat
        ifft_to_real!(u, u_hat, ifft_scratch)
        
        # Physics Tracking
        max_index = argmax_abs2_skip1(u_hat)
        dom_mode  = abs(kx[max_index])

        if dom_mode != dominate_modes[end][2]
            push!(dominate_modes, (tend, dom_mode))
        end

        # Check L2 threshold
        if t_thr < 0
            l2_norm = sqrt(Lx) * norm(u_hat) / Nx
            if l2_norm > l2_thr
                t_thr = tend
                mu_thr = mu_tend
            end
        end
    end
    
    dom_mode_at_thr = T(NaN)
    if t_thr >= 0
        idx = findlast(x -> x[1] < t_thr, dominate_modes)
        if idx !== nothing
            dom_mode_at_thr = dominate_modes[idx][2]
        else
            dom_mode_at_thr = dominate_modes[1][2]
        end
    end
    
    dom_mode_final = dominate_modes[end][2]
    
    return mu_thr, dom_mode_at_thr, dom_mode_final
end

function run_simulation(T_type::Type{T}) where {T}
    println("Initializing execution parameters with type: ", T_type)
    
    EP = [T(10)^(-5.5)]
    
    dt = T(20.0)
    mass = T(0.0)
    mu0 = T(-0.2)
    muf = T(0.21)
    muf0 = muf
    xscale = T(10.0)
    Lx = xscale * T(pi)
    Nx = 1024
    l2_thr = T(1.0)
    
    for ep in EP
        println("Running simulation for epsilon = ", Float64(ep))
        
        muf_temp = muf0

        T_end = (muf_temp - mu0) / ep
        num_time_steps = ceil(Int, Float64(T_end / dt))
        
        # Fourier wavenumbers + linear operator
        half_Nx = Nx ÷ 2
        kx = (T(pi) / Lx) .* vcat(0:(half_Nx - 1), (-half_Nx):-1) 
        Laplacian_k = -kx.^2
        L_operator = -(Laplacian_k.^2)
        
        kx_max = maximum(abs.(kx))
        dealias_mask = [abs(k) <= (T(2)/3) * kx_max ? T(1) : T(0) for k in kx]
        
        # Coefficients
        E, E2, Q, f1, f2, f3 = compute_etdrk4_coefficients(L_operator, dt, T)
        
        # Initial Conditions: All modes on, except mean
        u_hat0 = ones(Complex{T}, Nx)
        u_hat0[1] = 0
        u0 = real(ifft(u_hat0))
        u = copy(u0)
        
        @time mu_thr, dom_mode, dom_mode_final = solve_etdrk4!(
            u, num_time_steps, dt, mu0, ep, mass, Lx, kx, Laplacian_k, dealias_mask,
            E, E2, Q, f1, f2, f3, l2_thr
        )
        
        println("--- Results for epsilon = ", Float64(ep), "")
        println("MU THRESHOLD:  ", Float64(mu_thr))
        println("DOMINANT MODE: ", Float64(dom_mode))
        println("FINAL MODE:    ", Float64(dom_mode_final))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    # Run with Float64x2 for high numerical precision
    run_simulation(Float64x4)
end
