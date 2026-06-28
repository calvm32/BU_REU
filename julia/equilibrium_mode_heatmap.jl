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

# 1: single mode IC ( plot x = IC mode VS. y = Eq. mode VS. color = probability )
# 2: single mode IC + vary epsilon ( plot 1 but varying in time = epsilon )
# 3: random IC ( plot x = epsilon VS. y = Eq. mode VS. color = probability )

type = 1;

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# ──────────────────────────────────────────────────────────────────────────────
#  Parameters
# ──────────────────────────────────────────────────────────────────────────────
const IC_MODES      = 0:10
const NUM_RUNS      = 20
const XSCALE        = 8

const EPSILON       = 0.025                        # used for type == 1
const EPSILON_LIST  = range(0.0, 0.1; length=50)  # used for type == 2 or 3

const T_START       = -10.0
const MU_FINAL      = 2.0
const DT_FLOAT      = 0.1
const MASS          = 0.0

const LX_FLOAT      = XSCALE * pi
const NX            = 2^10

# ==============================================================================
#  Helper: build ETDRK4 coefficients for the *linear* bi-Laplacian operator
#  L = -(∂_xx)^2  (constant-coefficient, autonomous linear part only).
#  The non-linear forcing carries the time-dependent mu(t).
# ==============================================================================
"""
    compute_etdrk4_coefficients(L_operator, dt_float, T) -> E, E2, Q, f1, f2, f3

Contour-integral ETDRK4 coefficient arrays (Cox–Matthews / Kassam–Trefethen).
"""
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

# ==============================================================================
#  Core time-stepper  (non-autonomous: mu = mu(t))
#  Returns the dominant wavenumber index at the final time step.
# ==============================================================================
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
        t_half = Tf(t_vec[n-1] + DT_FLOAT / 2)
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

# ==============================================================================
#  Build shared spatial / spectral arrays (type-parameterised)
# ==============================================================================
function build_grid(::Type{T}) where T
    Lx  = T(LX_FLOAT)
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

# ==============================================================================
#  Custom dark-blue → white colormap (mirrors MATLAB cmap)
# ==============================================================================
function dark_blue_white_cmap(levels::Int)
    dark_blue = (0.02, 0.12, 0.55)
    white     = (1.0,  1.0,  1.0)
    r = range(dark_blue[1], white[1]; length=levels)
    g = range(dark_blue[2], white[2]; length=levels)
    b = range(dark_blue[3], white[3]; length=levels)
    # reverse so that high probability = dark blue
    return reverse([RGB(r[i], g[i], b[i]) for i in 1:levels])
end

# ==============================================================================
#  MAIN
# ==============================================================================
function run_simulation()

    T_type = Float64          # ← swap to Float64x2 / Float64x4 for higher precision
    println("Precision type: ", T_type)

    # ── Shared grid ──────────────────────────────────────────────────────────
    x, kx, Laplacian_k, L_operator, dealias_mask = build_grid(T_type)
    Lx   = T_type(LX_FLOAT)
    mass = T_type(MASS)

    # ── Time vector (shared across all runs / epsilon values) ─────────────────
    # T_end = mu_final / epsilon  →  recomputed per epsilon in type 2.
    # For type 1 we use the fixed EPSILON.
    t_vec_type1 = collect(T_START : DT_FLOAT : (MU_FINAL / EPSILON))

    # ── ETDRK4 coefficients (depend only on L, dt — NOT on mu) ───────────────
    println("Computing ETDRK4 coefficients...")
    E, E2, Q, f1, f2, f3 = compute_etdrk4_coefficients(
        L_operator, DT_FLOAT, T_type
    )

    IC_modes_vec = collect(IC_MODES)
    num_IC       = length(IC_modes_vec)

    # ──────────────────────────────────────────────────────────────────────────
    if type == 1
        # ── Fixed epsilon, sweep IC modes ─────────────────────────────────────
        println("\n=== Type 1: single-mode IC heatmap (ε = $EPSILON) ===")

        epsilon = T_type(EPSILON)
        mu_fn   = t -> epsilon * t

        Heat = zeros(num_IC, num_IC)   # Heat[eq_mode_idx, ic_idx]

        for (ic_idx, k0) in enumerate(IC_modes_vec)
            u0 = @. mass + cos(2 * T_type(π) * T_type(k0) * x / (2 * Lx))

            for run in 1:NUM_RUNS
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

                @printf("IC=%2d/%2d  run=%2d/%2d  dom_k=%.3f → mode %d\n",
                    ic_idx, num_IC, run, NUM_RUNS, kdom, final_mode)
            end
        end

        # ── Plot ──────────────────────────────────────────────────────────────
        cmap   = dark_blue_white_cmap(NUM_RUNS + 1)
        T_end  = MU_FINAL / EPSILON

        p = heatmap(
            IC_modes_vec, IC_modes_vec, Heat ./ NUM_RUNS;
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
            text(@sprintf("Parameters: T = %.0f,  t₀ = %.0f,  ε = %.3f\nEach IC ran for %d trials",
                          T_end, T_START, EPSILON, NUM_RUNS),
                 :center, 10))

        savefig(p, "bifurcation_heatmap_type1.png")
        println("\nSaved: bifurcation_heatmap_type1.png")

    # ──────────────────────────────────────────────────────────────────────────
    elseif type == 2
        # ── Vary epsilon, produce animated heatmap video ──────────────────────
        println("\n=== Type 2: sweep ε, animated heatmap ===")

        num_eps   = length(EPSILON_LIST)
        Heat_3d   = zeros(num_IC, num_IC, num_eps)   # [eq, ic, eps_idx]

        for (eidx, eps_val) in enumerate(EPSILON_LIST)
            eps   = T_type(eps_val)
            mu_fn = t -> eps * t
            T_end = eps_val > 0 ? MU_FINAL / eps_val : abs(T_START)
            t_vec = collect(T_START : DT_FLOAT : T_end)

            # Recompute ETDRK4 coefficients only when dt changes (dt is fixed here)
            # so we reuse E, E2, Q, f1, f2, f3 from above.

            for (ic_idx, k0) in enumerate(IC_modes_vec)
                u0 = @. mass + cos(2 * T_type(π) * T_type(k0) * x / (2 * Lx))

                for run in 1:NUM_RUNS
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

                @printf("  ε idx %2d/%2d | IC %2d/%2d done\n",
                    eidx, num_eps, ic_idx, num_IC)
            end
        end

        # ── Animate ──────────────────────────────────────────────────────────
        println("Generating animation...")
        cmap = dark_blue_white_cmap(NUM_RUNS + 1)

        anim = @animate for eidx in 1:num_eps
            eps_val = EPSILON_LIST[eidx]
            T_end   = eps_val > 0 ? MU_FINAL / eps_val : abs(T_START)

            heatmap(
                IC_modes_vec, IC_modes_vec, Heat_3d[:, :, eidx] ./ NUM_RUNS;
                color        = cmap,
                clims        = (0.0, 1.0),
                xlabel       = "IC mode number",
                ylabel       = "Eq. mode number",
                title        = "Likelihood of equilibrium modes\n" *
                               @sprintf("ε = %.4f  T = %.0f  t₀ = %.0f  runs = %d",
                                        eps_val, T_end, T_START, NUM_RUNS),
                colorbar     = true,
                aspect_ratio = :equal,
                grid         = true,
                size         = (700, 600)
            )
        end

        mp4(anim, "bifurcation_heatmap_type2.mp4"; fps=10)
        println("Saved: bifurcation_heatmap_type2.mp4")

    # ──────────────────────────────────────────────────────────────────────────
    elseif type == 3
        println("Type 3 (random IC) not yet implemented — mirrors MATLAB stub.")

    else
        error("Unknown type = $type")
    end
end

run_simulation()