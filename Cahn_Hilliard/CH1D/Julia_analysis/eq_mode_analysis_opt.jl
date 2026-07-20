import Pkg
Pkg.add([
    "MultiFloats", "GenericFFT", "AbstractFFTs",
    "Plots", "FFMPEG", "Printf", "Distributions"
])

using MultiFloats
using GenericFFT
using AbstractFFTs
using LinearAlgebra
using Statistics
using Plots
using Printf
using Colors
using Distributions
using Measures
using ColorSchemes
using LaTeXStrings

MultiFloats.use_bigfloat_transcendentals()
gr()

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# Description and configuration
# Find how different ICs and epsilon values affect equilibrium mode

# 0: single run  ( 2x2 plot: solution | fourier spectrum / dominant mode | L2 norm )
# 1: single mode IC ( heatmap: x = IC mode, y = Eq. mode, color = probability )
# 2: single mode IC + vary epsilon ( animated heatmap, epsilon sweeps over time )
# 3: random IC ( heatmap: x = epsilon, y = Eq. mode, color = probability )

const type = 3

# ------------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------------

# Shared parameters
const IC_modes      = 0:10
const num_runs      = 10        # num runs taken for averaging

const epsilon       = 0.025                        # used for type == 0 and type == 1
const epsilon_list  = range(0.00001, 0.2; length=100)  # used for type == 2 or 3

const t0            = -10.0
const mu_final      = 2.0
const dt_float      = 0.1
const MASS          = 0.0

const L        = 10
const lx_float      = L * pi    # half-domain size
const NX            = 2^10

# Threshold for equilibrium detection: relative change in dominant mode is below this
# for at least `EQ_WINDOW` consecutive steps
const EQ_REL_TOL    = 1e-4
const EQ_WINDOW     = 50
const ETDRK4_QUAD   = 32

const PRECISION = Float64x4

# ------------------------------------------------------------------------------------------
# Helper: globally-defined plotting measures
# ------------------------------------------------------------------------------------------
default(
    size=(1200,800),
    dpi=200,
    linewidth=2,
    guidefontsize=14,
    tickfontsize=11,
    titlefontsize=16,
    legendfontsize=11,
    margin=8mm,
    left_margin=10mm,
    right_margin=8mm,
    top_margin=12mm,
    bottom_margin=10mm,
    gridalpha=0.25,
    framestyle=:box,
    fontfamily="Helvetica",
    legend=:topright
)

palette(:batlow)

# ------------------------------------------------------------------------------------------
# Helper: build ETDRK4 coefficients for the linear bi-Laplacian operator
# L = -(k^2)^2  (constant-coefficient, autonomous linear part only).
# ------------------------------------------------------------------------------------------
function compute_etdrk4_coefficients(
    L_operator::Vector{T},
    dt::T,
    ::Type{T};
    M::Int = 32
) where {T<:Number}

    Nx = length(L_operator)

    imT = complex(one(T) * 0, one(T))
    r = [exp(imT * T(pi) * (T(k) - T(1//2)) / T(M)) for k in 1:M]

    E  = exp.(dt .* L_operator)
    E2 = exp.(dt .* L_operator ./ 2)

    Q  = zeros(T, Nx)
    f1 = zeros(T, Nx)
    f2 = zeros(T, Nx)
    f3 = zeros(T, Nx)

    for i in 1:Nx
        z = dt * L_operator[i]
        LR = z .+ r

        Q[i]  = dt * real(mean((exp.(LR ./ 2) .- 1) ./ LR))
        f1[i] = dt * real(mean((-4 .- LR .+ exp.(LR) .* (4 .- 3 .* LR .+ LR .^ 2)) ./ LR .^ 3))
        f2[i] = dt * real(mean((2 .+ LR .+ exp.(LR) .* (-2 .+ LR)) ./ LR .^ 3))
        f3[i] = dt * real(mean((-4 .- 3 .* LR .- LR .^ 2 .+ exp.(LR) .* (4 .- LR)) ./ LR .^ 3))
    end

    return E, E2, Q, f1, f2, f3
end


# ------------------------------------------------------------------------------------------
# Core time-stepper (non-autonomous: mu = mu(t))
# ------------------------------------------------------------------------------------------
function run_etdrk4(
        u0          :: Vector{Tf},
        t0_val      :: Tf,
        dt_val      :: Tf,
        num_steps   :: Int,
        mu_fn,
        kx          :: Vector{Tf},
        Laplacian_k :: Vector{Tf},
        dealias_mask:: Vector{Tf},
        E  :: Vector{Tf},  E2 :: Vector{Tf},
        Q  :: Vector{Tf},
        f1 :: Vector{Tf},  f2 :: Vector{Tf},  f3 :: Vector{Tf},
        mass :: Tf,
        ::Type{T};
        store_history :: Bool = false
    ) where {Tf, T}

    Nx        = length(u0)
    u         = copy(u0)
    u_hat     = fft(u)

    u3        = zeros(Tf, Nx)
    a         = zeros(Tf, Nx)
    b         = zeros(Tf, Nx)
    c         = zeros(Tf, Nx)

    Nu_hat    = zeros(Complex{Tf}, Nx)
    Na_hat    = zeros(Complex{Tf}, Nx)
    Nb_hat    = zeros(Complex{Tf}, Nx)
    Nc_hat    = zeros(Complex{Tf}, Nx)
    a_hat     = zeros(Complex{Tf}, Nx)
    b_hat     = zeros(Complex{Tf}, Nx)
    c_hat     = zeros(Complex{Tf}, Nx)

    Lx        = Tf(lx_float)

    # History arrays (only allocated when needed)
    if store_history
        u_hist    = Vector{Vector{Tf}}(undef, num_steps + 1)
        t_hist    = Vector{PRECISION}(undef, num_steps + 1)
        kdom_hist = Vector{PRECISION}(undef, num_steps + 1)
        l2_hist   = Vector{PRECISION}(undef, num_steps + 1)

        # Store initial state
        u_hist[1]    = copy(u)
        t_hist[1]    = t0_val
        u_hat_init   = fft(u .- mass)
        kdom_hist[1] = kx[argmax(abs.(u_hat_init))] |> PRECISION
        l2_hist[1]   = (sqrt(sum(u .^ 2) * (2 * Lx / Nx))) |> PRECISION
    end

    for n in 1:num_steps
        # Compute time on the fly to avoid allocating massive time vectors
        t_prev = t0_val + Tf(n - 1) * dt_val
        t_half = t0_val + (Tf(n) - Tf(0.5)) * dt_val
        t_curr = t0_val + Tf(n) * dt_val

        mu_prev = mu_fn(t_prev)
        mu_half = mu_fn(t_half)
        mu_curr = mu_fn(t_curr)

        # Stage 1
        @. u3      = u^3 - mu_prev * u
        Nu_hat    .= dealias_mask .* Laplacian_k .* fft(u3)

        # Stage 2
        @. a_hat   = E2 * u_hat + Q * Nu_hat
        a         .= real.(ifft(a_hat))
        @. u3      = a^3 - mu_half * a
        Na_hat    .= dealias_mask .* Laplacian_k .* fft(u3)

        # Stage 3
        @. b_hat   = E2 * u_hat + Q * Na_hat
        b         .= real.(ifft(b_hat))
        @. u3      = b^3 - mu_half * b
        Nb_hat    .= dealias_mask .* Laplacian_k .* fft(u3)

        # Stage 4
        @. c_hat   = E2 * a_hat + Q * (2 * Nb_hat - Nu_hat)
        c         .= real.(ifft(c_hat))
        @. u3      = c^3 - mu_curr * c
        Nc_hat    .= dealias_mask .* Laplacian_k .* fft(u3)

        # Combine
        @. u_hat   = E * u_hat + f1 * Nu_hat + 2 * f2 * (Na_hat + Nb_hat) + f3 * Nc_hat
        u         .= real.(ifft(u_hat))

        if store_history
            u_hist[n+1]    = copy(u)
            t_hist[n+1]    = t_curr
            u_hat_snap   = fft(u .- mass)
            kdom_hist[n+1] = (kx[argmax(abs.(u_hat_snap))]) |> PRECISION
            l2_hist[n+1]   = (sqrt(sum(u .^ 2) * (2 * Lx / Nx))) |> PRECISION
        end
    end

    if store_history
        return u, u_hist, t_hist, kdom_hist, l2_hist
    else
        # Return only dominant mode of final state
        u_hat_final = fft(u .- mass)
        return (kx[argmax(abs.(u_hat_final))]) |> PRECISION
    end
end


# ------------------------------------------------------------------------------------------
# Thin wrapper for heatmap sweeps: returns only the final dominant wavenumber
# ------------------------------------------------------------------------------------------
function final_dominant_mode(
        u0, t0_val, dt_val, num_steps, mu_fn, kx, Laplacian_k, dealias_mask,
        E, E2, Q, f1, f2, f3, mass, PRECISION
    )
    return run_etdrk4(
        u0, t0_val, dt_val, num_steps, mu_fn, kx, Laplacian_k, dealias_mask,
        E, E2, Q, f1, f2, f3, mass, PRECISION;
        store_history=false
    )
end


# ------------------------------------------------------------------------------------------
# Equilibrium detection
# ------------------------------------------------------------------------------------------
function find_equilibrium_time(
        kdom_hist :: Vector{PRECISION},
        t_hist    :: Vector{PRECISION}
    ) :: Union{PRECISION, Nothing}

    N = length(kdom_hist)
    N < EQ_WINDOW && return nothing

    # Take absolute values to avoid +/- frequency flip-flop noise
    abs_kdom = abs.(kdom_hist)

    for i in EQ_WINDOW:N
        window = abs_kdom[(i-EQ_WINDOW+1):i]

        if all(==(window[end]), window)
            eq_mode = window[end]
            abs_kdom_f = Float64.(abs_kdom)
            eq_mode = Float64(eq_mode)
            first_idx = findfirst(==(eq_mode), abs_kdom_f)

            return t_hist[first_idx]
        end
    end

    return nothing
end

# ------------------------------------------------------------------------------------------
# Build shared spatial / spectral arrays
# ------------------------------------------------------------------------------------------
function build_grid(::Type{T}) where T
    Lx       = T(lx_float)
    dx       = T(2) * Lx / T(NX)
    x        = [T(i) * dx for i in (-NX÷2):(NX÷2 - 1)]

    half_NX  = NX ÷ 2
    kx       = (T(pi) / Lx) .* T.(vcat(0:(half_NX - 1), (-half_NX):-1))

    Laplacian_k  = @. -kx^2
    L_operator   = @. -(Laplacian_k^2)

    kx_max       = maximum(abs.(kx))
    dealias_mask = T[abs(k) <= (T(2)/T(3)) * kx_max ? T(1) : T(0) for k in kx]

    return x, kx, Laplacian_k, L_operator, dealias_mask
end


# ------------------------------------------------------------------------------------------
# Custom dark-blue → white colormap  (high probability = dark blue)
# ------------------------------------------------------------------------------------------
function dark_blue_white_cmap(levels::Int)
    dark_blue = (0.02, 0.12, 0.55)
    white     = (1.0,  1.0,  1.0)
    r = range(dark_blue[1], white[1]; length=levels)
    g = range(dark_blue[2], white[2]; length=levels)
    b = range(dark_blue[3], white[3]; length=levels)
    return reverse([RGB(r[i], g[i], b[i]) for i in 1:levels])
end


# ------------------------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------------------------
function run_simulation()

    # Shared grid
    x, kx, Laplacian_k, L_operator, dealias_mask = build_grid(PRECISION)
    Lx   = PRECISION(lx_float)
    mass = PRECISION(MASS)

    # ETDRK4 coefficients (depend only on L and dt, not on mu)
    E, E2, Q, f1, f2, f3 =
    compute_etdrk4_coefficients(
        PRECISION.(L_operator),
        PRECISION(dt_float),
        PRECISION
    )

    IC_modes_vec = collect(IC_modes)
    num_IC       = length(IC_modes_vec)

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    if type == 0

        eps_val = PRECISION(epsilon)
        mu_fn   = t -> eps_val * t
        T_end = max(mu_final / eps_val, PRECISION(150))

        t0_T    = PRECISION(t0)
        dt_T    = PRECISION(dt_float)
        
        num_steps = floor(Int, Float64((T_end - t0) / dt_float))

        # Single-mode IC: mode k0 = 1
        k0 = 1
        u0 = @. mass + cos(2 * PRECISION(pi) * PRECISION(k0) * x / (2 * Lx))

        println("Running type=0 single trajectory …")
        _, u_hist, t_hist, kdom_hist, l2_hist = run_etdrk4(
            u0, t0_T, dt_T, num_steps, mu_fn, kx, Laplacian_k, dealias_mask,
            E, E2, Q, f1, f2, f3, mass, PRECISION;
            store_history=true
        )

        t_eq = find_equilibrium_time(kdom_hist, t_hist)
        if isnothing(t_eq)
            println("Equilibrium not detected within simulation window.")
        else
            @printf("Equilibrium reached at t ≈ %.3f\n", t_eq)
        end

        plot_stride = max(1, length(t_hist) ÷ 500)
        idx_plot    = 1:plot_stride:length(t_hist)

        t_plot    = t_hist[idx_plot]
        l2_plot   = l2_hist[idx_plot]
        kdom_plot = kdom_hist[idx_plot]
        u_final   = u_hist[end]
        x_f64     = PRECISION.(x)

        u_hat_final   = fft(PRECISION.(u_final .- mass))
        kx_shifted    = PRECISION.(fftshift(kx))
        uhat_shifted  = max.(PRECISION.(fftshift(abs.(u_hat_final))), 1e-65)

        k_ref = [PRECISION(pi * j / lx_float) for j in 0:6]

        p1 = plot(x_f64, PRECISION.(u_final);
            linewidth = 1.5,
            xlabel    = "x",
            ylabel    = "u(x, T)",
            title     = @sprintf("Solution at t = %.2f", t_hist[end]),
            legend    = false,
            grid      = true,
            color     = :steelblue)

        p2 = plot(kx_shifted, uhat_shifted;
            linewidth = 1.5,
            yscale    = :log10,
            xlims     = (-8.0, 8.0),
            ylims     = (1e-14, maximum(uhat_shifted) * 10),
            xlabel    = "k",
            ylabel    = "|û(k)|",
            title     = "Fourier spectrum (final state)",
            legend    = false,
            grid      = true,
            color     = :darkorange)
        for kj in k_ref
            vline!(p2, [kj]; linestyle=:dash, color=:red, alpha=0.4)
            kj != 0 && vline!(p2, [-kj]; linestyle=:dash, color=:red, alpha=0.4)
        end

        p3 = plot(t_plot, abs.(kdom_plot);
            linetype  = :steppost,
            linewidth = 2.0,
            xlabel    = "t",
            ylabel    = "dominant |k|",
            title     = "Dominant wavenumber vs time",
            legend    = false,
            grid      = true,
            color     = :purple)
        for kj in k_ref
            hline!(p3, [kj]; linestyle=:dash, color=:red, alpha=0.4)
        end
        if !isnothing(t_eq)
            vline!(p3, [t_eq]; color=:green, linewidth=1.5,
                   label="eq. onset t=$(round(t_eq,digits=2))")
        end

        p4 = plot(t_plot, l2_plot;
            linewidth = 1.5,
            xlabel    = "t",
            ylabel    = "‖u‖_2",
            title     = "L2 Norm vs Time",
            legend    = false,
            grid      = true,
            color     = :teal)
        if !isnothing(t_eq)
            vline!(p4, [t_eq]; color=:green, linewidth=2.0, linestyle=:dash,
                   label=@sprintf("eq. t=%.1f", t_eq))
            annotate!(p4, t_eq, maximum(l2_plot) * 0.9,
                text(@sprintf("t_eq≈%.1f", t_eq), :left, 8, :green))
            eq_idx = findfirst(>=(t_eq), t_hist)
            if !isnothing(eq_idx)
                scatter!(p4, [t_hist[eq_idx]], [l2_hist[eq_idx]];
                    color=:green, markersize=6, markershape=:circle, legend=false)
            end
        end

        bigtitle = "\$\\epsilon=$epsilon,\\;t_0=$t0,\\;mass=$mass,\\;IC\\,mode=$k0\\;\$"

        plt = plot(
            p1,p2,p3,p4,
            layout=grid(2,2),
            size=(1400,900)
        )

        plot!(
            plt,
            plot_title=bigtitle,
            top_margin=18mm,
            left_margin=10mm,
            bottom_margin=10mm,
            right_margin=4mm
        )

        savefig(plt, "type0_single_run.png")
        println("Saved type0_single_run.png")

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    elseif type == 1

        eps_val = PRECISION(epsilon)
        mu_fn   = t -> eps_val * t
        T_end = max(mu_final / eps_val, PRECISION(150))

        t0_T    = PRECISION(t0)
        dt_T    = PRECISION(dt_float)

        num_steps = floor(Int, Float64((T_end - t0) / dt_float))

        Heat = zeros(num_IC, num_IC)

        for (ic_idx, k0) in enumerate(IC_modes_vec)
            u0 = @. mass + cos(2 * PRECISION(pi) * PRECISION(k0) * x / (2 * Lx))

            for run in 1:num_runs
                kdom = final_dominant_mode(
                    u0, t0_T, dt_T, num_steps, mu_fn,
                    kx, Laplacian_k, dealias_mask,
                    E, E2, Q, f1, f2, f3, mass, PRECISION
                )
                final_mode = Int(round(Float64(abs(kdom) * Lx / pi)))
                eq_idx     = findfirst(==(final_mode), IC_modes_vec)
                !isnothing(eq_idx) && (Heat[eq_idx, ic_idx] += 1)

                @printf("IC=%2d/%2d  run=%2d/%2d  dom_k=%.3f -> mode %d\n",
                    ic_idx, num_IC, run, num_runs, kdom, final_mode)
            end
        end

        cmap = dark_blue_white_cmap(num_runs + 1)

        p = heatmap(
            IC_modes_vec, IC_modes_vec, Heat ./ num_runs;
            color        = cmap,
            clims        = (0.0, 1.0),
            xlabel       = "IC mode number",
            ylabel       = "Eq. mode number",
            title = "Equilibrium Mode Likelihood for Single-Mode ICs\n" *
                    "\$\\mu_{\\rm final}=$mu_final,\\;\\epsilon=$epsilon,\\;L=$(L)\\times2\\pi\$",
            colorbar     = true,
            aspect_ratio = :equal,
            grid         = true,
            size         = (700, 600),
            yflip=false,
            colorbar_title="Probability"
        )

        annotate!(p, mean(IC_modes_vec), IC_modes_vec[1] - 1.5,
            text(@sprintf("\epsilon=%.3f   T=%.0f   t_0=%.0f   runs=%d",
                          eps_val, T_end, t0, num_runs),
                 :center, 9))

        savefig(p, "bifurcation_heatmap_type1.png")
        println("Saved bifurcation_heatmap_type1.png")

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    elseif type == 2

        num_eps   = length(epsilon_list)
        Heat_3d   = zeros(num_IC, num_IC, num_eps)

        for (eidx, eps_val) in enumerate(epsilon_list)
            eps   = PRECISION(eps_val)
            mu_fn = t -> eps * t
            T_end = max(mu_final / eps_val, PRECISION(150))

            t0_T    = PRECISION(t0)
            dt_T    = PRECISION(dt_float)

            num_steps = floor(Int, Float64((T_end - t0) / dt_float))

            for (ic_idx, k0) in enumerate(IC_modes_vec)
                u0 = @. mass + cos(2 * PRECISION(pi) * PRECISION(k0) * x / (2 * Lx))

                for run in 1:num_runs
                    kdom = final_dominant_mode(
                        u0, t0_T, dt_T, num_steps, mu_fn,
                        kx, Laplacian_k, dealias_mask,
                        E, E2, Q, f1, f2, f3, mass, PRECISION
                    )
                    final_mode = Int(round(Float64(abs(kdom) * Lx / pi)))
                    eq_idx     = findfirst(==(final_mode), IC_modes_vec)
                    !isnothing(eq_idx) && (Heat_3d[eq_idx, ic_idx, eidx] += 1)
                end

                @printf("\epsilon idx %3d/%3d | IC %2d/%2d done\n",
                    eidx, num_eps, ic_idx, num_IC)
            end
        end

        cmap = dark_blue_white_cmap(num_runs + 1)

        anim = @animate for eidx in 1:num_eps
            eps_val = epsilon_list[eidx]

            heatmap(
                IC_modes_vec, IC_modes_vec, Heat_3d[:, :, eidx] ./ num_runs;
                color        = cmap,
                clims        = (0.0, 1.0),
                xlabel       = "IC mode number",
                ylabel       = "Eq. mode number",
                title = "Equilibrium Mode Likelihood for Single-Mode ICs\n" *
                        "\$\\mu_{\\rm final}=$(round(mu_final,digits=2)),\\;L=$(L)\\times2\\pi\$",
                colorbar     = true,
                aspect_ratio = :equal,
                grid         = true,
                size         = (700, 600),
                yflip=false,
                colorbar_title="Probability"
            )
        end

        mp4(anim, "bifurcation_heatmap_type2.mp4"; fps=10)
        println("Saved bifurcation_heatmap_type2.mp4")

    # ------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------
    elseif type == 3
        sigma     = 0.05
        num_eps   = length(epsilon_list)

        Heat = zeros(num_IC, num_eps)

        for (eidx, eps_val) in enumerate(epsilon_list)
            eps   = PRECISION(eps_val)
            mu_fn = t -> eps * t
            T_end = max(mu_final / eps_val, PRECISION(150))

            t0_T    = PRECISION(t0)
            dt_T    = PRECISION(dt_float)
            
            num_steps = floor(Int, Float64((T_end - t0) / dt_float))

            for run in 1:num_runs
                noise = sigma .* randn(PRECISION, NX)
                u0    = mass .+ noise

                kdom = final_dominant_mode(
                    u0, t0_T, dt_T, num_steps, mu_fn,
                    kx, Laplacian_k, dealias_mask,
                    E, E2, Q, f1, f2, f3, mass, PRECISION
                )
                final_mode = Int(round(Float64(abs(kdom) * Lx / pi)))
                eq_idx     = findfirst(==(final_mode), IC_modes_vec)
                !isnothing(eq_idx) && (Heat[eq_idx, eidx] += 1)
            end

            @printf("\epsilon idx %3d/%3d done\n", eidx, num_eps)
        end

        col_sums = sum(Heat; dims=1)
        Heat_norm = Heat ./ max.(col_sums, eps())

        cmap = dark_blue_white_cmap(num_runs + 1)

        eps_labels = round.(collect(epsilon_list); digits=3)

        p = heatmap(
            1:num_eps, IC_modes_vec, Heat_norm;
            color    = cmap,
            clims    = (0.0, 1.0),
            xlabel = latexstring("\\epsilon\\ \\mathrm{index}\\quad", "(\\epsilon\\in[",
                round(epsilon_list[1], digits=3), ",", round(epsilon_list[end], digits=2),"])"),
            ylabel   = "Eq. mode number",
            title = "Equilibrium Mode Likelihood for Random ICs\n" *
                    "\$\\mu_{\\rm final}=$(round(mu_final,digits=2)),\\;L=$(L)\\times2\\pi\$",
            colorbar = true,
            grid     = true,
            size     = (900, 550),
            aspect_ratio=1,
            yflip=false,
            colorbar_title="Probability"
        )

        tick_step = max(1, num_eps ÷ 10)
        xtick_idx = collect(1:tick_step:num_eps)
        
        plot!(p;
            xticks = (xtick_idx,
                      [@sprintf("%.2f", epsilon_list[i]) for i in xtick_idx]),
                      
            top_margin=18mm,
            left_margin=10mm,
            bottom_margin=10mm,
            right_margin=4mm)

        savefig(p, "bifurcation_heatmap_type3.png")
        println("Saved bifurcation_heatmap_type3.png")

    else
        error("Unknown type = $type  (must be 0, 1, 2, or 3)")
    end
end

run_simulation()
