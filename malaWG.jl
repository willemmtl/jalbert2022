using Distributions, ForwardDiff, LinearAlgebra

include("iGMRF.jl")

"""

"""
function malaWG(niter::Integer, h::Real, θ₀::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF)

    m = F.G.m₁ * F.G.m₂

    θ = zeros(m + 1, niter)
    θ[:, 1] = θ₀

    acc = falses(niter)

    for j = 2:niter

        # Generate κᵤ
        θ[1, j] = rand(fcκᵤ(F, θ[2:end, j-1]))
        # Update iGMRF
        F = iGMRF(F.G.m₁, F.G.m₂, θ[1, j])

        # Generate μ
        y = rand(dLangevin(θ[2:end, j-1], h=h, Y=Y, F=F))

        logπ̃(θ::Vector{<:Real}) = fcμ(θ, Y=Y, F=F)
        logq(y::Vector{<:Real}, x::Vector{<:Real}) = qLangevin(y, x, h=h, Y=Y, F=F)

        lr = logπ̃(y) + logq(θ[2:end, j-1], y) - logπ̃(θ[2:end, j-1]) - logq(y, θ[2:end, j-1])

        if log(rand()) < lr
            acc[j] = true
            θ[2:end, j] = y
        else
            θ[2:end, j] = θ[2:end, j-1]
        end

    end

    accRate = count(acc) / (niter - 1) * 100
    println("Taux d'acceptation: ", round(accRate, digits=2), " %")

    return θ
end


"""
    qLangevin(y, x; h, Y, F)

Langevin distribution evaluated at y knowing the parameters x.

# Arguments

- `y::Vector{<:Real}`: Value at which the distribution is evaluated.
- `x::Vector{<:Real}`: Parameters of the Langevin distribution.
- `h::Real`: Instrumental variance
- `Y::Vector{Vector{Float64}}`: Observations.
- `F::iGMRF`: last updated iGMRF.
"""
function qLangevin(y::Vector{<:Real}, x::Vector{<:Real}; h::Real, Y::Vector{Vector{Float64}}, F::iGMRF)

    return logpdf(dLangevin(x, h=h, Y=Y, F=F), y)

end


"""
    dLangevin(θ; h, Y, F)

Langevin distribution = instrumental law of the MALA algorithm.

# Arguments

- `θ::Vector{<:Real}`: Parameters.
- `h::Real`: Instrumental variance
- `Y::Vector{Vector{Float64}}`: Observations.
- `F::iGMRF`: last updated iGMRF.
"""
function dLangevin(θ::Vector{<:Real}; h::Real, Y::Vector{Vector{Float64}}, F::iGMRF)

    
    logπ̃(θ::Vector{<:Real}) = fcμ(θ, Y=Y, F=F)
    ∇logπ̃(θ::Vector{<:Real}) = ForwardDiff.gradient(logπ̃, θ)

    return MvNormal(θ + h * ∇logπ̃(θ) / 2, h*I)

end


"""
    fcμ(μ; Y, F)

Compute the log-density of the parameters' partial posteriori law.

# Arguments

- `μ::Vector{<:Real}`: Parameters [μ₁, ..., μₘ].
- `Y::Vector{Vector{Float64}}`: Observations.
- `F::iGMRF`: last updated iGMRF.
"""
function fcμ(μ::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF)

    return sum(loglikelihood.(GeneralizedExtremeValue.(μ, 1, 0), Y)) - F.κᵤ * μ' * F.G.W * μ / 2

end