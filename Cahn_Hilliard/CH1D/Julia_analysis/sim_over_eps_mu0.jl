# Ramped Cahn-Hilliard Simulation in Julia using ETDRK4
using MultiFloats
using GenericFFT
using AbstractFFTs
using FFTW
using LinearAlgebra
using MAT
using Printf

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

function fft_real!(dst::Vector{Complex{T}}, src::Vector{T}, P_fft) where {T}
    @inbounds for i in eachindex(dst, src)
        dst[i] = Complex{T}(src[i], zero(T))
    end
    P_fft * dst # Allocation-free in-place execution of the pre-computed plan
end

function ifft_to_real!(dst::Vector{T}, src::Vector{Complex{T}}, scratch::Vector{Complex{T}}, P_ifft) where {T}
    copyto!(scratch, src)
    P_ifft * scratch # Allocation-free in-place execution of the pre-computed plan
    @inbounds for i in eachindex(dst)
        dst[i] = real(scratch[i])
    end
end

# ─────────────────────────────────────────────────────────────────────────────
function compute_etdrk4_coefficients(L_operator, dt::T, ::Type{T}) where {T}
    M = 64
    Nx = length(L_operator)
    r = [exp(im * T(pi) * (T(k) - T(0.5)) / T(M)) for k in 1:M]
    
    E  = zeros(T, Nx)
    E2 = zeros(T, Nx)
    Q  = zeros(T, Nx)
    f1 = zeros(T, Nx)
    f2 = zeros(T, Nx)
    f3 = zeros(T, Nx)
    
    @. E  = exp(dt * L_operator)
    @. E2 = exp(dt * L_operator / 2)
    
    for i in 1:Nx
        z = dt * L_operator[i]
        sum_Q  = zero(Complex{T})
        sum_f1 = zero(Complex{T})
        sum_f2 = zero(Complex{T})
        sum_f3 = zero(Complex{T})
        
        for k in 1:M
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

function compute_nonlinear_forcing_hat!(N_hat, u, mu, Laplacian_k, dealias_mask, u3_scratch, fft_scratch, P_fft)
    @. u3_scratch = u * (u * u - mu)
    fft_real!(fft_scratch, u3_scratch, P_fft)
    @. N_hat = dealias_mask * Laplacian_k * fft_scratch
end

"""
Optimized ETDRK4 Time Stepping Engine utilizing full in-place loop fusion.
Now updated to collect histories and terminate early based on L2 threshold.
"""
function solve_etdrk4!(
    u::Vector{T}, 
    num_time_steps::Int, 
    dt::T,
    mu0::T,
    ep::T,
    Lx::T,
    kx::Vector{T},
    Laplacian_k::Vector{T}, 
    dealias_mask::Vector{T}, 
    E::Vector{T}, E2::Vector{T}, Q::Vector{T}, f1::Vector{T}, f2::Vector{T}, f3::Vector{T},
    terminate_thr::T
) where {T}
    
    Nx = length(u)
    
    # Pre-allocated workspaces
    u3_scratch = zeros(T, Nx)
    a          = zeros(T, Nx)
    b          = zeros(T, Nx)
    c          = zeros(T, Nx)
    
    fft_scratch  = zeros(Complex{T}, Nx)
    ifft_scratch = zeros(Complex{T}, Nx)
    Nu_hat = zeros(Complex{T}, Nx)
    Na_hat = zeros(Complex{T}, Nx)
    Nb_hat = zeros(Complex{T}, Nx)
    Nc_hat = zeros(Complex{T}, Nx)
    a_hat  = zeros(Complex{T}, Nx)
    b_hat  = zeros(Complex{T}, Nx)
    c_hat  = zeros(Complex{T}, Nx)
    
    P_fft  = plan_fft!(fft_scratch)
    P_ifft = plan_ifft!(ifft_scratch)
    
    fft_real!(fft_scratch, u, P_fft)
    u_hat = copy(fft_scratch)
    
    # Track the specific indices for modes 1 through 5 
    # kx[1] is the DC component (k=0), so modes start at kx[2]
    mode_indices = 2:6
    
    # History Allocations
    t_hist = Vector{Float64}(undef, num_time_steps + 1)
    l2_hist = Vector{Float64}(undef, num_time_steps + 1)
    amp_hist = zeros(Float64, 5, num_time_steps + 1)
    dominate_modes = Tuple{Float64, Float64}[]
    
    # Initial State logging
    l2_norm = sqrt(Lx) * norm(u_hat) / Nx
    t_hist[1] = 0.0
    l2_hist[1] = Float64(l2_norm)
    for j in 1:5
        amp_hist[j, 1] = Float64(abs(u_hat[mode_indices[j]]))
    end
    
    max_index = argmax_abs2_skip1(u_hat)
    push!(dominate_modes, (0.0, Float64(abs(kx[max_index]))))
    
    step_idx = 1

    for n in 1:num_time_steps
        t = T(n - 1) * dt
        tmid = t + dt / 2
        tend = t + dt
        
        mu_t = mu0 + t * ep
        mu_tmid = mu0 + tmid * ep
        mu_tend = mu0 + tend * ep
        
        # ETDRK4 Stages
        compute_nonlinear_forcing_hat!(Nu_hat, u, mu_t, Laplacian_k, dealias_mask, u3_scratch, fft_scratch, P_fft)
        
        @. a_hat = E2 * u_hat + Q * Nu_hat
        ifft_to_real!(a, a_hat, ifft_scratch, P_ifft)
        compute_nonlinear_forcing_hat!(Na_hat, a, mu_tmid, Laplacian_k, dealias_mask, u3_scratch, fft_scratch, P_fft)
        
        @. b_hat = E2 * u_hat + Q * Na_hat
        ifft_to_real!(b, b_hat, ifft_scratch, P_ifft)
        compute_nonlinear_forcing_hat!(Nb_hat, b, mu_tmid, Laplacian_k, dealias_mask, u3_scratch, fft_scratch, P_fft)
        
        @. c_hat = E2 * a_hat + Q * (2 * Nb_hat - Nu_hat)
        ifft_to_real!(c, c_hat, ifft_scratch, P_ifft)
        compute_nonlinear_forcing_hat!(Nc_hat, c, mu_tend, Laplacian_k, dealias_mask, u3_scratch, fft_scratch, P_fft)
        
        @. u_hat = E * u_hat + f1 * Nu_hat + 2 * f2 * (Na_hat + Nb_hat) + f3 * Nc_hat
        ifft_to_real!(u, u_hat, ifft_scratch, P_ifft)
        
        # Tracking Physics
        max_index = argmax_abs2_skip1(u_hat)
        dom_mode  = Float64(abs(kx[max_index]))

        if dom_mode != dominate_modes[end][2]
            push!(dominate_modes, (Float64(tend), dom_mode))
        end

        l2_norm = sqrt(Lx) * norm(u_hat) / Nx
        
        # Update histories
        step_idx += 1
        t_hist[step_idx] = Float64(tend)
        l2_hist[step_idx] = Float64(l2_norm)
        for j in 1:5
            amp_hist[j, step_idx] = Float64(abs(u_hat[mode_indices[j]]))
        end
        
        # Termination check
        if l2_norm >= terminate_thr
            break
        end
    end
    
    # Trim the preallocated arrays down to their actual used lengths and return
    return t_hist[1:step_idx], l2_hist[1:step_idx], amp_hist[:, 1:step_idx], dominate_modes
end

function run_grid_simulation(T_type::Type{T}) where {T}
    if string(nameof(T_type)) == "MultiFloat"
        # Extracts the precision integer (e.g., 4 from MultiFloat{Float64, 4})
        datatype = "Float64x$(T_type.parameters[2])" 
    else
        # Fallback for standard types like Float64
        datatype = string(nameof(T_type))
    end

    println("Initializing execution parameters with type: ", datatype)
    
    # Set up 100x100 Grid equivalent to MATLAB parameters
    EP_vec  = 10 .^ range(-5.6, stop=-5.4, length=3)
    MU0_vec = range(-0.5, stop=0, length=50)
    
    Nx = 512
    dt0 = T(0.02)
    muf = T(2.0)
    L = T(10.0)
    Lx = L * T(pi)
    terminate_thr = T(5.0)
    
    # Fourier wavenumbers + linear operator
    half_Nx = Nx ÷ 2
    kx = (T(pi) / Lx) .* vcat(0:(half_Nx - 1), (-half_Nx):-1) 
    Laplacian_k = -kx.^2
    L_operator = -(Laplacian_k.^2)
    kx_max = maximum(abs.(kx))
    dealias_mask = [abs(k) <= (T(2)/3) * kx_max ? T(1) : T(0) for k in kx]
    
    # Base Initial Conditions
    u_hat0 = ones(Complex{T}, Nx)
    u_hat0[1] = 0
    u0 = real(ifft(u_hat0))
    
    # Pre-allocate Matrix outputs for flattening equivalent
    num_ep = length(EP_vec)
    num_mu = length(MU0_vec)
    
    # Flatten parameters to use a single 1D threaded loop (like MATLAB's parfor)
    num_total_sims = num_ep * num_mu
    flat_EP = repeat(EP_vec, outer=num_mu)
    flat_MU0 = repeat(MU0_vec, inner=num_ep)
    
    # Thread-safe counter
    counter = Threads.Atomic{Int}(0)
    
    println("Starting parallel parameter sweep over $num_total_sims simulations...")
    
    Threads.@threads :dynamic for idx in 1:num_total_sims
        ep  = T(flat_EP[idx])
        mu0 = T(flat_MU0[idx])
        
        # Calculate array coordinates for storage
        i = (idx - 1) % num_ep + 1
        j = (idx - 1) ÷ num_ep + 1
        
        dt = dt0 * T(1e-3) / ep
        T_end = (muf - mu0) / ep
        num_time_steps = ceil(Int, Float64(T_end / dt))
        
        # Compute coefficients specific to this simulation's dt
        E, E2, Q, f1, f2, f3 = compute_etdrk4_coefficients(L_operator, dt, T)
        
        u = copy(u0)
        
        # Run local ETDRK4
        t_hist, l2_hist, amp_hist, dom_modes = solve_etdrk4!(
            u, num_time_steps, dt, mu0, ep, Lx, kx, Laplacian_k, dealias_mask,
            E, E2, Q, f1, f2, f3, terminate_thr
        )
        
        # Convert Dom Modes from Array of Tuples to [N x 2] Float64 Matrix for MATLAB struct compat
        dm_mat = zeros(Float64, length(dom_modes), 2)
        for k in 1:length(dom_modes)
            dm_mat[k, 1] = dom_modes[k][1]
            dm_mat[k, 2] = dom_modes[k][2]
        end
        
        export_dict = Dict(
            "DMODE_HIST"    => dm_mat,
            "L2_HIST"       => l2_hist,
            "AMP_HISTS"     => amp_hist,
            "ep"            => Float64(ep),
            "mu0"           => Float64(mu0),
            "Lx"            => Float64(Lx),
            "Nx"            => Float64(Nx),
            "dt"            => Float64(dt),
            "terminate_thr" => Float64(terminate_thr),
            "datatype"      => datatype
        )

        filename = @sprintf("data/sample_ep=1e%.5f_mu0=%.7f_%s.mat", log10(ep), mu0, datatype)
        matwrite(filename, export_dict)
        
        # Progress Update 
        Threads.atomic_add!(counter, 1)
        if counter[] % 10 == 0
            @printf("Completed %d / %d simulations...\n", counter[], num_total_sims)
        end
    end
    println("Finished completely!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_grid_simulation(Float64x4)
end
