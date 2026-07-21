import Pkg
# Pkg.add(["MultiFloats", "GenericFFT", "Plots", "FFMPEG", "AbstractFFTs"])

using MultiFloats
using GenericFFT
using AbstractFFTs
using LinearAlgebra
using Plots

MultiFloats.use_bigfloat_transcendentals()

# PRECISION CONTROL
# Float64, Float64x2, Float64x4, etc.
const T_type = Float64x2

"""
Computes the critical bifurcation parameters for a given mode j.
"""
function critical_bifurcation(j, L, mass, ::Type{T}) where {T}
    k_j = 2 * T(pi) * T(j) / T(L)
    mu_j = 3 * T(mass)^2 + k_j^2
    return mu_j, k_j
end

# -----------------------
# Allocation-free helpers
# -----------------------

"""
Zero-allocation argmax of |x|^2 over a complex vector, skipping k=0
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


"""
Computes the ETDRK4 integration coefficients using a complex contour integral.
"""
function compute_etdrk4_coefficients(L_operator, dt_float::Float64, ::Type{T}) where {T}
    dt = T(dt_float)
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
Optimized ETDRK4 Time Stepping Engine with allocation-free time-stepping loop.
"""
function solve_etdrk4!(
    u::Vector{T}, 
    num_time_steps::Int, 
    t_vec::StepRangeLen,
    mu::T, 
    mass::T,
    kx::Vector{T},
    Laplacian_k::Vector{T}, 
    dealias_mask::Vector{T}, 
    E::Vector{T}, E2::Vector{T}, Q::Vector{T}, f1::Vector{T}, f2::Vector{T}, f3::Vector{T},
    plot_every::Int,
    num_frames::Int
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
    
    # Tracking histories
    u_hist = zeros(Float64, Nx, num_frames)
    u_hat_hist = zeros(Float64, Nx, num_frames)
    
    fft_real!(fft_scratch, u)
    u_hat = copy(fft_scratch)
    
    max_index = argmax_abs2_skip1(u_hat)
    
    dominate_mode_t = Float64[max(1e-2, t_vec[1])]
    dominate_mode_k = Float64[abs(Float64(kx[max_index]))]

    @. u_hist[:, 1] = Float64(u)
    @. u_hat_hist[:, 1] = Float64(abs(u_hat))

    for n in 2:num_time_steps
        # Stage 1
        compute_nonlinear_forcing_hat!(Nu_hat, u, mu, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
        
        # Stage 2
        @. a_hat = E2 * u_hat + Q * Nu_hat
        ifft_to_real!(a, a_hat, ifft_scratch)
        compute_nonlinear_forcing_hat!(Na_hat, a, mu, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
        
        # Stage 3 
        @. b_hat = E2 * u_hat + Q * Na_hat
        ifft_to_real!(b, b_hat, ifft_scratch)
        compute_nonlinear_forcing_hat!(Nb_hat, b, mu, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
        
        # Stage 4
        @. c_hat = E2 * a_hat + Q * (2 * Nb_hat - Nu_hat)
        ifft_to_real!(c, c_hat, ifft_scratch)
        compute_nonlinear_forcing_hat!(Nc_hat, c, mu, Laplacian_k, dealias_mask, u3_scratch, fft_scratch)
        
        # Final Correction
        @. u_hat = E * u_hat + f1 * Nu_hat + 2 * f2 * (Na_hat + Nb_hat) + f3 * Nc_hat
        ifft_to_real!(u, u_hat, ifft_scratch)
        
        max_index = argmax_abs2_skip1(u_hat)
        dom_mode  = abs(Float64(kx[max_index]))

        if dom_mode != dominate_mode_k[end]
            push!(dominate_mode_t, t_vec[n])
            push!(dominate_mode_k, dom_mode)
        end

        if (n - 1) % plot_every == 0
            frame_idx = (n - 1) ÷ plot_every + 1
            @. u_hist[:, frame_idx] = Float64(u)
            @. u_hat_hist[:, frame_idx] = Float64(abs(u_hat))
        end

        if (n - 1) % 100 == 0
            println("Done w/ $(n-1)/$num_time_steps time steps")
        end
    end
    
    return u_hist, u_hat_hist, dominate_mode_t, dominate_mode_k
end

function run_simulation()
    println("Initializing execution parameters with type: ", T_type)
    
    t0_float = 0.0
    dt_float = 0.1
    L = 10
    Lx = T_type(L) * T_type(pi)
    Nx = 4096
    max_mode_num = 12

    n_subc = 8 
    n_mu = 2 

    mass = (2*n_subc+1)*sqrt(2)/20
    
    mu_n = 3*mass^2 + (n_mu/L)^2
    mu_next = 3*mass^2 + ((n_mu + 1)/L)^2
    mu = (mu_n + mu_next)/2

    mu_float = Float64(mu)
    mass_float = Float64(mass)

    plot_dt = 30.0
    plot_every = round(Int, plot_dt / dt_float)
    
    # Generating full range of wavenumbers up to max_mode_num+1 to match MATLAB
    k_j_exact = T_type(pi) .* (0:(max_mode_num+1)) ./ Lx
    k_i = k_j_exact[2:end] # Slice correctly instead of returning a scalar
    k_j = [Float64(val) for val in k_j_exact] 

    dx = T_type(2) * Lx / T_type(Nx)
    x_grid = collect((-Nx÷2):(Nx÷2 - 1)) .* dx 
        
    half_Nx = Nx ÷ 2
    kx = (T_type(pi) / Lx) .* vcat(0:(half_Nx - 1), (-half_Nx):-1) 
    kx_float = [Float64(val) for val in kx]
    Laplacian_k = @. -kx^2
    L_operator = @. -(Laplacian_k^2)
    
    kx_max = maximum(abs.(kx))
    dealias_mask = [abs(k) <= (T_type(2)/3) * kx_max ? T_type(1) : T_type(0) for k in kx]
    
    println("Computing matrix exponential filters...")
    E, E2, Q, f1, f2, f3 = compute_etdrk4_coefficients(L_operator, dt_float, T_type)
    
    l = @layout [a; b c]

    # -----------------------------------   
    # run simulation for each wave number
    # -----------------------------------   

    for m in 1:length(k_i)
        k_init = k_i[m]

        lambda_k = k_init^2 * (mu - 3 * mass^2 - k_init^2)

        # final T defined so that freeze-out happens
        T_end_float = lambda_k > 0 ? Float64(2.5/lambda_k) : 10000.0

        t_vec = t0_float:dt_float:T_end_float
        num_time_steps = length(t_vec)
        plot_indices = 1:plot_every:num_time_steps
        num_frames = length(plot_indices)

        println("\n⚡ Launching Simulation: Initial Wavenumber k = $(round(k_init, digits=3))")
        
        # IC
        u0 = T_type(0.001) .* cos.(k_init .* x_grid) .+ T_type(mass)
        u  = copy(u0)
        
        @time u_hist, u_hat_hist, dom_t, dom_k = solve_etdrk4!(
            u, num_time_steps, t_vec, T_type(mu), T_type(mass), kx, Laplacian_k, dealias_mask,
            E, E2, Q, f1, f2, f3, plot_every, num_frames
        )

        println("Generating Video...")
        video_filename = "$(T_type)_k=$(round(k_init, digits=3))_T=$(round(T_end_float, digits=0)).mp4"
        shifted_kx = fftshift(kx_float)
        
        anim = @animate for i in 1:num_frames
            current_t = t_vec[plot_indices[i]]
            
            # subtract mass
            p1 = plot(x_grid, u_hist[:, i] .- mass_float, 
                      title="t = $(round(current_t, digits=3))", 
                      xlabel="x", ylabel="v = u - m", 
                      linewidth=1.5, color=:blue, legend=false, 
                      ylims=(-1.2 * sqrt(mu_float), 1.2 * sqrt(mu_float)), grid=true)
            
            plot_t = max(1e-2, current_t) 
            valid_idx = dom_t .<= plot_t
            cur_dom_t = dom_t[valid_idx]
            cur_dom_k = dom_k[valid_idx]
            
            if isempty(cur_dom_k)
                push!(cur_dom_t, plot_t)
                push!(cur_dom_k, dom_k[1])
            else
                push!(cur_dom_t, plot_t)
                push!(cur_dom_k, cur_dom_k[end])
            end
            
            p2 = plot(cur_dom_t, cur_dom_k, 
                      linetype=:steppost, linewidth=2, 
                      xlabel="t", ylabel="Wavenumber (k)", 
                      xlims=(1e-2, T_end_float), ylims=(0, k_j[end]), 
                      color=:blue, legend=false, grid=true)
            
            hline!(p2, k_j, linestyle=:dash, color=:red, alpha=0.6)
                      
            u_hat_shifted_raw = fftshift(u_hat_hist[:, i])
            shifted_u_hat = @. max(u_hat_shifted_raw, 1e-70)
            
            p3 = plot(shifted_kx, shifted_u_hat, 
                      linewidth=2, color=:blue, 
                      xlabel="k", ylabel="|u_hat|", 
                      yscale=:log10, xlims=(-10, 10), ylims=(1e-70, max(1.0, maximum(shifted_u_hat))), 
                      legend=false, grid=true)
                      
            plot(p1, p2, p3, layout=l, size=(1200, 700), margin=5Plots.mm, dpi=300)
        end
        
        mp4(anim, video_filename, fps=30)
        println("Saved: $video_filename")
    end
end

run_simulation()