using Distributions, ForwardDiff, LinearAlgebra

include("iGMRF.jl")

"""

"""
function mala(niter::Integer, h::Real, θ₀::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF)

    m = F.G.m₁ * F.G.m₂

    θ = zeros(m + 1, niter)
    θ[:, 1] = θ₀

    # Taux d'acceptation
    acc = falses(niter)

    for j = 2:niter

        y = rand(dLangevin(θ[:, j-1], h=h, Y=Y, F=F))

        logπ̃(θ::Vector{<:Real}) = logposteriori(θ, Y=Y, F=F)
        logq(y::Vector{<:Real}, x::Vector{<:Real}) = qLangevin(y, x, h=h, Y=Y, F=F)

        lr = logπ̃(y) + logq(θ[:, j-1], y) - logπ̃(θ[:, j-1]) - logq(y, θ[:, j-1])

        if log(rand()) < lr
            acc[j] = true
            θ[:, j] = y
        else
            θ[:, j] = θ[:, j-1]
        end

    end

    accRate = count(acc) / (niter - 1) * 100
    println("Taux d'acceptation: ", round(accRate, digits=2), " %")

    return θ
end


function qLangevin(y::Vector{<:Real}, x::Vector{<:Real}; h::Real, Y::Vector{Vector{Float64}}, F::iGMRF)

    return logpdf(dLangevin(x, h=h, Y=Y, F=F), y)

end


function dLangevin(θ::Vector{<:Real}; h::Real, Y::Vector{Vector{Float64}}, F::iGMRF)

    
    logπ̃(θ::Vector{<:Real}) = logposteriori(θ, Y=Y, F=F)
    ∇logπ̃(θ::Vector{<:Real}) = ForwardDiff.gradient(logπ̃, θ)

    return MvNormal(θ + h * ∇logπ̃(θ) / 2, h*I)

end


"""
    logposteriori(θ; Y, F)

Compute the log-density of the parameters' partial posteriori joint law.

# Arguments

- `θ::Vector{<:Real}`: Parameters [κᵤ, μ₁, ..., μₘ].
- `Y::Vector{Vector{Float64}}`: Observations.
- `F::iGMRF`: iGMRF giving the structure matrix.
"""
function logposteriori(θ::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF)

    κᵤ = θ[1]
    μ = θ[2:end]
    m = F.G.m₁ * F.G.m₂

    return sum(loglikelihood.(GeneralizedExtremeValue.(μ, 1, 0), Y)) + (m - F.r) * log(κᵤ) / 2 - κᵤ * μ' * F.G.W * μ / 2 + logpdf(Gamma(1, 100), κᵤ)
end