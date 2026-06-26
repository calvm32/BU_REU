import Pkg
Pkg.add(["MultiFloats", "GenericFFT", "Plots", "FFMPEG", "ExponentialUtilities", "OrdinaryDiffEq", "StaticArrays", "SciMLOperators", "OrdinaryDiffEqExponentialRK"])

# Note this uses a LOT of memory

module CahnHilliardSimulation

using MultiFloats
using GenericFFT
using OrdinaryDiffEq
using OrdinaryDiffEqExponentialRK
using LinearAlgebra
using StaticArrays
using Statistics
using Plots
using Plots.PlotMeasures
using SciMLOperators

# Enable arbitrary precision transcendentals required for SciML's internal contour integrations
MultiFloats.use_bigfloat_transcendentals()

# =====================================================================
# PATCH: Implement missing remainder and division operations for MultiFloats
# =====================================================================
import Base: rem, mod, div, fld, cld

for f in (:rem, :mod, :fld, :cld)
    @eval function Base.$f(x::MultiFloats.MultiFloat{T,N}, y::MultiFloats.MultiFloat{T,N}) where {T,N}
        return MultiFloats.MultiFloat{T,N}($f(BigFloat(x), BigFloat(y)))
    end
    @eval function Base.$f(x::MultiFloats.MultiFloat{T,N}, y::Real) where {T,N}
        return MultiFloats.MultiFloat{T,N}($f(BigFloat(x), BigFloat(y)))
    end
    @eval function Base.$f(x::Real, y::MultiFloats.MultiFloat{T,N}) where {T,N}
        return MultiFloats.MultiFloat{T,N}($f(BigFloat(x), BigFloat(y)))
    end
end

function Base.div(x::MultiFloats.MultiFloat{T,N}, y::MultiFloats.MultiFloat{T,N}, r::RoundingMode) where {T,N}
    return MultiFloats.MultiFloat{T,N}(div(BigFloat(x), BigFloat(y), r))
end
function Base.div(x::MultiFloats.MultiFloat{T,N}, y::Real, r::RoundingMode) where {T,N}
    return MultiFloats.MultiFloat{T,N}(div(BigFloat(x), BigFloat(y), r))
end
function Base.div(x::Real, y::MultiFloats.MultiFloat{T,N}, r::RoundingMode) where {T,N}
    return MultiFloats.MultiFloat{T,N}(div(BigFloat(x), BigFloat(y), r))
end

function (::Type{T})(x::MultiFloats.MultiFloat{U,N}) where {T<:Integer,U,N}
    return T(BigFloat(x))
end

import ExponentialUtilities
import MultiFloats

function ExponentialUtilities.exponential!(
    A::AbstractMatrix{<:Union{MultiFloats.MultiFloat{T,N}, Complex{<:MultiFloats.MultiFloat{T,N}}}}, 
    ::ExponentialUtilities.ExpMethodHigham2005Base, 
    args...
) where {T,N}
    # Direct the solver to use ExponentialUtilities' pure-Julia generic Padé routine
    return ExponentialUtilities.exponential!(A, ExponentialUtilities.ExpMethodGeneric())
end
# =====================================================================

# 1. Strict Constants to prevent global variable dispatch penalties
const FloatQD = Float64x4
const ComplexQD = Complex{FloatQD}
const Nx = 4096
const Lx = FloatQD(6) * π
const dx = 2 * Lx / Nx

# 2. State Cache to prevent allocations in the hot ODE loop
struct CHCache{T1, T2}
    mu::FloatQD
    dealias_mask::T1
    Laplacian_hat::T2
    u3_buf::Vector{FloatQD}
end

# 3. The In-Place Nonlinear ODE Function
function ch_nonlin!(du_hat, u_hat, cache, t)
    # Transform to physical space
    u_real = real(ifft(u_hat))
    
    # Multithreaded nonlinear computation (u^3 - mu*u)
    Threads.@threads for i in 1:Nx
        @inbounds cache.u3_buf[i] = u_real[i]^3 - cache.mu * u_real[i]
    end
    
    # Transform back to spectral space
    u3_hat = fft(cache.u3_buf)
    
    # Multithreaded application of the dealiasing mask and Laplacian
    Threads.@threads for i in 1:Nx
        @inbounds du_hat[i] = cache.dealias_mask[i] * cache.Laplacian_hat[i] * u3_hat[i]
    end
end

# 4. Main Executable Function
function run_simulation(alpha_val::Float64)
    println("Initializing Simulation for alpha = $alpha_val")
    
    t0 = FloatQD(0.0)
    T  = FloatQD(500.0)
    dt = FloatQD(0.1)
    mass = FloatQD(0.0)
    
    # Grid and Operator setup
    kx = vcat(collect(0:Nx÷2-1), collect(-Nx÷2:-1)) .* (π / Lx)
    Laplacian_hat = .- (kx .^ 2)
    L_operator = .- (Laplacian_hat .^ 2) # Core linear operator (-k^4)
    
    kx_max = maximum(abs.(kx))
    dealias_mask = abs.(kx) .<= (2/3) * kx_max

    # Calculate Bifurcation Mu
    mu4 = 3 * mass^2 + (2 * π * 4 / (2 * Lx))^2
    mu5 = 3 * mass^2 + (2 * π * 5 / (2 * Lx))^2
    mu = FloatQD((1 - alpha_val) * mu4 + alpha_val * mu5)

    # High-Precision Initial Condition
    noise = FloatQD(0.0002) .* (FloatQD.(rand(Nx)) .- FloatQD(0.5))
    noise .-= mean(noise)
    u0 = mass .+ noise
    u_hat0 = fft(u0)

    # Initialize Cache Object
    cache = CHCache(mu, dealias_mask, Laplacian_hat, zeros(FloatQD, Nx))

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
        
        p2 = plot(t_stairs, k_stairs, linetype=:steppost, linewidth=2, xscale=:log10,
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
end

end # end module

@time CahnHilliardSimulation.run_simulation(0.1)