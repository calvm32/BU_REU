import Pkg
Pkg.add([
    "MultiFloats", "GenericFFT", "AbstractFFTs",
    "Plots", "FFMPEG", "Printf"
])

using MultiFloats
using GenericFFT
using AbstractFFTs
using LinearAlgebra
using Statistics
using Plots
using Printf
using Colors

MultiFloats.use_bigfloat_transcendentals()


# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# Description and configuration
# Find how different ICs and epsilon values affect equilibrium mode

# 0: single run ( plot solution, L2 norm, equilibrium mode plot, fourier spectrum )
# 1: single mode IC ( plot x = IC mode VS. y = Eq. mode VS. color = probability )
# 2: single mode IC + vary epsilon ( plot 1 but varying in time = epsilon )
# 3: random IC ( plot x = epsilon VS. y = Eq. mode VS. color = probability )

type = 2;

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# 
const IC_modes      = 0:10
const num_runs      = 20
const num_runs        = 10

const epsilon       = 0.025                        # used for type == 1
const epsilon_list  = range(0.0, 1.5; length=100)  # used for type == 2 or 3

const t0            = -10.0
const mu_final      = 2.0
const dt_float      = 0.1
const mass          = 0.0

const lx_float      = num_runs * pi # half mode size
const NX            = 2^10

# ------------------------------------------------------------------------------------------
# Helper: build ETDRK4 coefficients for the *linear* bi-Laplacian operator
# L = -(del_xx)^2  (constant-coefficient, autonomous linear part only).
# The non-linear forcing carries the time-dependent mu(t).
# ------------------------------------------------------------------------------------------
function compute_etdrk4_coefficients(L_operator::Vector{Tf},
                                     dt_float::Float64,
                                     ::Type{T}) where {Tf, T}
    dt = T(dt_float)
    M  = 16          # points on contour circle (matches MATLAB)
    Nx = length(L_operator)

    imT = complex(zero(T), one(T))

    r = [exp(imT*T(pi)*(T(k)-T(1//2))/T(M))
        for k=1:M]

    E  = @. exp(dt * L_operator)
    E2 = @. exp(dt * L_operator / 2)

    Q  = zeros(T, Nx)
    f1 = zeros(T, Nx)
    f2 = zeros(T, Nx)
    f3 = zeros(T, Nx)

    for i in 1:Nx
        z   = dt * L_operator[i]
        LR  = [z + ri for ri in r]   # length-M vector

        Q[i]  = dt * real(mean((exp.(LR ./ 2) .- 1) ./ LR))
        f1[i] = dt * real(mean((-4 .- LR .+ exp.(LR) .* (4 .- 3 .* LR .+ LR .^ 2)) ./ LR .^ 3))
        f2[i] = dt * real(mean((2 .+ LR .+ exp.(LR) .* (-2 .+ LR)) ./ LR .^ 3))
        f3[i] = dt * real(mean((-4 .- 3 .* LR .- LR .^ 2 .+ exp.(LR) .* (4 .- LR)) ./ LR .^ 3))
    end

    return E, E2, Q, f1, f2, f3
end

# ------------------------------------------------------------------------------------------
# Core time-stepper  (non-autonomous: mu = mu(t))
# Returns the dominant wavenumber index at the final time step.
# ------------------------------------------------------------------------------------------
"""
    final_dominant_mode(u0, t_vec, mu_fn, kx, Laplacian_k, dealias_mask,
                        E, E2, Q, f1, f2, f3, mass, T) -> kx[argmax]

Run one full ETDRK4 trajectory and return the rounded dominant wavenumber at t_end.
"""
function final_dominant_mode(
        u0          :: Vector{Tf},
        t_vec       :: AbstractVector{Float64},
        mu_fn,
        kx          :: Vector{Tf},
        Laplacian_k :: Vector{Tf},
        dealias_mask:: Vector{Tf},
        E  :: Vector{Tf},  E2 :: Vector{Tf},
        Q  :: Vector{Tf},
        f1 :: Vector{Tf},  f2 :: Vector{Tf},  f3 :: Vector{Tf},
        mass :: Tf,
        ::Type{T}
    ) where {Tf, T}

    Nx   = length(u0)
    u    = copy(u0)
    u_hat = fft(u)

    u3   = zeros(Tf, Nx)
    a    = zeros(Tf, Nx);  b = zeros(Tf, Nx);  c = zeros(Tf, Nx)

    Nu_hat = zeros(Complex{Tf}, Nx)
    Na_hat = zeros(Complex{Tf}, Nx)
    Nb_hat = zeros(Complex{Tf}, Nx)
    Nc_hat = zeros(Complex{Tf}, Nx)
    a_hat  = zeros(Complex{Tf}, Nx)
    b_hat  = zeros(Complex{Tf}, Nx)
    c_hat  = zeros(Complex{Tf}, Nx)

    num_steps = length(t_vec)

    for n in 2:num_steps
        t_prev = Tf(t_vec[n-1])
        t_half = Tf(t_vec[n-1] + dt_float / 2)
        t_curr = Tf(t_vec[n])

        mu_prev = mu_fn(t_prev)
        mu_half = mu_fn(t_half)
        mu_curr = mu_fn(t_curr)

        # Stage 1
        @. u3      = u^3 - mu_prev * u
        Nu_hat .= dealias_mask .* Laplacian_k .* fft(u3)

        # Stage 2
        @. a_hat = E2 * u_hat + Q * Nu_hat
        a .= real.(ifft(a_hat))
        @. u3     = a^3 - mu_half * a
        Na_hat .= dealias_mask .* Laplacian_k .* fft(u3)

        # Stage 3
        @. b_hat = E2 * u_hat + Q * Na_hat
        b .= real.(ifft(b_hat))
        @. u3     = b^3 - mu_half * b
        Nb_hat .= dealias_mask .* Laplacian_k .* fft(u3)

        # Stage 4
        @. c_hat = E2 * a_hat + Q * (2 * Nb_hat - Nu_hat)
        c .= real.(ifft(c_hat))
        @. u3     = c^3 - mu_curr * c
        Nc_hat .= dealias_mask .* Laplacian_k .* fft(u3)

        # Combine
        @. u_hat = E * u_hat + f1 * Nu_hat + 2 * f2 * (Na_hat + Nb_hat) + f3 * Nc_hat
        u .= real.(ifft(u_hat))
    end

    # Dominant mode: argmax of |FFT(u - mass)|, then round the wavenumber
    u_hat_final = fft(u .- mass)
    max_idx = argmax(abs.(u_hat_final))
    return Float64(kx[max_idx])
end

# ------------------------------------------------------------------------------------------
# Build shared spatial / spectral arrays (type-parameterised)
# ------------------------------------------------------------------------------------------
function build_grid(::Type{T}) where T
    Lx  = T(lx_float)
    dx  = T(2) * Lx / T(NX)
    x   = [T(i) * dx for i in (-NX÷2):(NX÷2 - 1)] # may need collect(-Lx:dx:Lx-dx)

    half_NX = NX ÷ 2
    kx  = (T(π) / Lx) .* T.(vcat(0:(half_NX - 1), (-half_NX):-1))

    Laplacian_k  = @. -kx^2
    L_operator   = @. -(Laplacian_k^2)

    kx_max       = maximum(abs.(kx))
    dealias_mask = T[abs(k) <= (T(2)/T(3)) * kx_max ? T(1) : T(0) for k in kx]

    return x, kx, Laplacian_k, L_operator, dealias_mask
end

# ------------------------------------------------------------------------------------------
# Custom dark-blue -> white colormap (mirrors MATLAB cmap)
# ------------------------------------------------------------------------------------------
function dark_blue_white_cmap(levels::Int)
    dark_blue = (0.02, 0.12, 0.55)
    white     = (1.0,  1.0,  1.0)
    r = range(dark_blue[1], white[1]; length=levels)
    g = range(dark_blue[2], white[2]; length=levels)
    b = range(dark_blue[3], white[3]; length=levels)
    # reverse so that high probability = dark blue
    return reverse([RGB(r[i], g[i], b[i]) for i in 1:levels])
end

# ------------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------------
function run_simulation()

    T_type = Float64          # ← swap to Float64x2 / Float64x4 for higher precision

    # Shared grid 
    x, kx, Laplacian_k, L_operator, dealias_mask = build_grid(T_type)
    Lx   = T_type(lx_float)
    mass = T_type(mass)

    # Time vector (shared across all runs / epsilon values) 
    # T_end = mu_final / epsilon  →  recomputed per epsilon in type 2.
    # For type 1 we use the fixed epsilon.
    t_vec_type1 = collect(t0 : dt_float : (mu_final / epsilon))

    # ETDRK4 coefficients (depend only on L, dt — NOT on mu) 
    E, E2, Q, f1, f2, f3 = compute_etdrk4_coefficients(
        L_operator, dt_float, T_type
    )

    IC_modes_vec = collect(IC_modes)
    num_IC       = length(IC_modes_vec)

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

    if type == 0

        # Initialize Cache Object
        cache = CHCache(mu, dealias_mask, Laplacian_k, zeros(FloatQD, Nx))

        # 5. OrdinaryDiffEq Setup
        # A SplitODEProblem takes the linear matrix first, then the nonlinear function.
        # We cast to ComplexQD to prevent SciML from failing type-promotions natively.
        L_matrix = Diagonal(ComplexQD.(L_operator))
        L_op = MatrixOperator(L_matrix)
        prob = SplitODEProblem(L_op, ch_nonlin!, u_hat0, (t0, T), cache)    
        
        # Solve using ETDRK4. SciML caches the contour integration of the operators automatically.
        plot_dt = FloatQD(1.0)
        println("Starting ODE Solve...")
        
        # adaptive=false enforces fixed pseudo-spectral stepping; saveat dictates our video frames
        sol = solve(prob, ETDRK4(), dt=dt, adaptive=false, saveat=plot_dt)
        
        println("ODE Solve Complete! Rendering MP4...")

        # --- 6. POST PROCESSING & PLOTTING ---
        # StaticArrays track dynamic dominant modes with zero heap allocations
        dominate_mode = Vector{SVector{2, Float64}}()
        
        anim = Animation()
        k_j = [π * i / Lx for i in 0:6]
        x_f64 = Float64.([(-Nx/2 + i) * dx for i in 0:Nx-1])
        mu_f64 = Float64(mu)
        
        # Iterate exclusively through the saved solution snapshots
        for i in 1:length(sol.t)
            t_curr = Float64(sol.t[i])
            u_hat_curr = @view sol.u[i][:] # Slice solution array using @view
            u_curr = real(ifft(u_hat_curr))
            
            # Mode Tracking
            max_index = findmax(abs.(fft(u_curr .- mass)))[2]
            dom_mode = Float64(kx[max_index])
            
            # Push to SVector if state changes
            if isempty(dominate_mode) || dominate_mode[end][2] != dom_mode
                push!(dominate_mode, SVector(t_curr, dom_mode))
            end
            
            u_f64 = Float64.(u_curr)
            
            # Subplot 1
            p1 = plot(x_f64, u_f64, linewidth=1.5, title="t = $(round(t_curr, digits=4))", 
                    xlabel="x", ylabel="u", ylims=1.2 * [-sqrt(mu_f64), sqrt(mu_f64)], legend=false, grid=true)
            
            # Subplot 2 
            valid_times = [m[1] for m in dominate_mode]
            valid_modes = [m[2] for m in dominate_mode]
            t_stairs = max.(vcat(valid_times, t_curr), 1e-4)
            k_stairs = vcat(valid_modes, valid_modes[end])
            
            p2 = plot(t_stairs, k_stairs, linetype=:steppost, linewidth=2, num_runs=:log10,
                    xlabel="t", ylabel="Wavenumber (k)", ylims=(0, Float64(k_j[end])), legend=false, grid=true)
            for kj_val in k_j
                hline!(p2, [Float64(kj_val)], linestyle=:dash, linecolor=:red, alpha=0.5)
            end
            
            # Subplot 3
            kx_shifted = Float64.(fftshift(kx))
            u_hat_shifted = max.(Float64.(fftshift(abs.(u_hat_curr))), 1e-65)
            
            p3 = plot(kx_shifted, u_hat_shifted, linewidth=2, yscale=:log10, xlims=(-10.0, 10.0),
                    ylims=(1e-65, maximum(u_hat_shifted)*10), 
                    xlabel="k", ylabel="|û|", legend=false, grid=true)
            
            layout_schema = @layout [grid(1, 1); grid(1, 2)]
            plt = plot(p1, p2, p3, layout=layout_schema, size=(1200, 700), left_margin=15mm, bottom_margin=5mm)
            frame(anim, plt)
        end
        
        video_filename = "long_alpha=$(round(alpha_val, digits=3))_T=$(round(Int, Float64(T))).mp4"
        mp4(anim, video_filename, fps=30)
        println("Simulation finished and video saved to: ", video_filename)

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

    if type == 1
        # Fixed epsilon, sweep IC modes 
        epsilon = T_type(epsilon)
        mu_fn   = t -> epsilon * t

        Heat = zeros(num_IC, num_IC)   # Heat[eq_mode_idx, ic_idx]

        for (ic_idx, k0) in enumerate(IC_modes_vec)
            u0 = @. mass + cos(2 * T_type(π) * T_type(k0) * x / (2 * Lx))

            for run in 1:num_runs
                kdom = final_dominant_mode(
                    u0, t_vec_type1, mu_fn,
                    kx, Laplacian_k, dealias_mask,
                    E, E2, Q, f1, f2, f3, mass, T_type
                )
                final_mode = round(Int, abs(kdom)*Lx/pi)
                eq_idx = findfirst(==(final_mode), IC_modes_vec)
                if !isnothing(eq_idx)
                    Heat[eq_idx, ic_idx] += 1
                end

                @printf("IC=%2d/%2d  run=%2d/%2d  dom_k=%.3f -> mode %d\n",
                    ic_idx, num_IC, run, num_runs, kdom, final_mode)
            end
        end

        # Plot 
        cmap   = dark_blue_white_cmap(num_runs + 1)
        T_end  = mu_final / epsilon

        p = heatmap(
            IC_modes_vec, IC_modes_vec, Heat ./ num_runs;
            color        = cmap,
            clims        = (0.0, 1.0),
            xlabel       = "IC mode number",
            ylabel       = "Eq. mode number",
            title        = "Likelihood of equilibrium modes reached\nfor various initial conditions",
            colorbar     = true,
            aspect_ratio = :equal,
            xticks       = (IC_modes_vec[1] - 0.5 : 1.0 : IC_modes_vec[end] + 0.5,
                             string.(IC_modes_vec[1]-1 : IC_modes_vec[end]+1)),
            yticks       = (IC_modes_vec[1] - 0.5 : 1.0 : IC_modes_vec[end] + 0.5,
                             string.(IC_modes_vec[1]-1 : IC_modes_vec[end]+1)),
            grid         = true,
            size         = (700, 600)
        )

        annotate!(p, 0.5, -0.12,
            text(@sprintf("Parameters: T = %.0f,  t_a = %.0f,  eps = %.3f\nEach IC ran for %d trials",
                          T_end, t0, epsilon, num_runs),
                 :center, 10))

        savefig(p, "bifurcation_heatmap_type1.png")

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------

    elseif type == 2
        # Vary epsilon, produce animated heatmap video 
        num_eps   = length(epsilon_list)
        Heat_3d   = zeros(num_IC, num_IC, num_eps)   # [eq, ic, eps_idx]

        for (eidx, eps_val) in enumerate(epsilon_list)
            eps   = T_type(eps_val)
            mu_fn = t -> eps * t
            T_end = eps_val > 0 ? mu_final / eps_val : abs(t0)
            t_vec = collect(t0 : dt_float : T_end)

            # Recompute ETDRK4 coefficients only when dt changes (dt is fixed here)
            # so we reuse E, E2, Q, f1, f2, f3 from above.

            for (ic_idx, k0) in enumerate(IC_modes_vec)
                u0 = @. mass + cos(2 * T_type(π) * T_type(k0) * x / (2 * Lx))

                for run in 1:num_runs
                    kdom = final_dominant_mode(
                        u0, t_vec, mu_fn,
                        kx, Laplacian_k, dealias_mask,
                        E, E2, Q, f1, f2, f3, mass, T_type
                    )
                    final_mode = round(Int, abs(kdom)*Lx/pi)
                    eq_idx = findfirst(==(final_mode), IC_modes_vec)
                    if !isnothing(eq_idx)
                        Heat_3d[eq_idx, ic_idx, eidx] += 1
                    end
                end

                @printf("epsilon idx %2d/%2d | IC %2d/%2d done\n",
                    eidx, num_eps, ic_idx, num_IC)
            end
        end

        # Animate 
        cmap = dark_blue_white_cmap(num_runs + 1)

        anim = @animate for eidx in 1:num_eps
            eps_val = epsilon_list[eidx]
            T_end   = eps_val > 0 ? mu_final / eps_val : abs(t0)

            heatmap(
                IC_modes_vec, IC_modes_vec, Heat_3d[:, :, eidx] ./ num_runs;
                color        = cmap,
                clims        = (0.0, 1.0),
                xlabel       = "IC mode number",
                ylabel       = "Eq. mode number",
                title        = "Likelihood of equilibrium modes\n" *
                               @sprintf("epsilon = %.4f  T = %.0f  t₀ = %.0f  runs = %d",
                                        eps_val, T_end, t0, num_runs),
                colorbar     = true,
                aspect_ratio = :equal,
                grid         = true,
                size         = (700, 600)
            )
        end

        mp4(anim, "bifurcation_heatmap_type2.mp4"; fps=10)


    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    
    elseif type == 3
        # Fixed epsilon, sweep IC modes 
        epsilon = T_type(epsilon)
        mu_fn   = t -> epsilon * t

        Heat = zeros(num_IC, num_IC)   # Heat[eq_mode_idx, ic_idx]

        for (ic_idx, k0) in enumerate(IC_modes_vec)
            sigma = 0.01
            u0 = @. mass + sigma*rand(Normal(0, 1))
            u0 = @. mass + cos(2 * T_type(π) * T_type(k0) * x / (2 * Lx))

            for run in 1:num_runs
                kdom = final_dominant_mode(
                    u0, t_vec_type1, mu_fn,
                    kx, Laplacian_k, dealias_mask,
                    E, E2, Q, f1, f2, f3, mass, T_type
                )
                final_mode = round(Int, abs(kdom)*Lx/pi)
                eq_idx = findfirst(==(final_mode), IC_modes_vec)
                if !isnothing(eq_idx)
                    Heat[eq_idx, ic_idx] += 1
                end

                @printf("IC=%2d/%2d  run=%2d/%2d  dom_k=%.3f -> mode %d\n",
                    ic_idx, num_IC, run, num_runs, kdom, final_mode)
            end
        end

        # Plot 
        cmap   = dark_blue_white_cmap(num_runs + 1)
        T_end  = mu_final / epsilon

        p = heatmap(
            IC_modes_vec, IC_modes_vec, Heat ./ num_runs;
            color        = cmap,
            clims        = (0.0, 1.0),
            xlabel       = "IC mode number",
            ylabel       = "Eq. mode number",
            title        = "Likelihood of equilibrium modes reached\nfor various initial conditions",
            colorbar     = true,
            aspect_ratio = :equal,
            xticks       = (IC_modes_vec[1] - 0.5 : 1.0 : IC_modes_vec[end] + 0.5,
                             string.(IC_modes_vec[1]-1 : IC_modes_vec[end]+1)),
            yticks       = (IC_modes_vec[1] - 0.5 : 1.0 : IC_modes_vec[end] + 0.5,
                             string.(IC_modes_vec[1]-1 : IC_modes_vec[end]+1)),
            grid         = true,
            size         = (700, 600)
        )

        annotate!(p, 0.5, -0.12,
            text(@sprintf("Parameters: T = %.0f,  t_a = %.0f,  eps = %.3f\nEach IC ran for %d trials",
                          T_end, t0, epsilon, num_runs),
                 :center, 10))

        savefig(p, "bifurcation_heatmap_type1.png")

    else
        error("Unknown type = $type")
    end
end

run_simulation()