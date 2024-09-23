using Mamba, Distributions, ForwardDiff, LinearAlgebra

include("../iGMRF.jl")

"""
    nuts(niter, h, θ₀; Y, F)

Perform a NUTS Within Gibbs algorithm.

# Arguments

- `niter::Integer`: Number of gibbs iterations.
- `h::Real`: Instrumental variance of MALA.
- `θ₀::Vector{<:Real}`: Initial values.
- `Y::Vector{Vector{Float64}}`: Observations.
- `F::iGMRF`: iGMRF (no need to be updated).
- `nchains::Integer` : Number of chaines for each parameter.
"""
function nuts(niter::Integer, θ₀::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF)

    m = F.G.m₁ * F.G.m₂

    # Initialisation
    names = [["κᵤ"]; ["μ$i" for i=1:m]]
    θ = Chains(niter, m+1, names=names)
    θ[1, :, 1] = θ₀

    ## Log-functional form of Posterior(μ, κᵤ | Y) + Gradient Vector
    logπ̃grad = function(θ::DenseVector)
        logπ̃(θ::DenseVector) = (
            sum(loglikelihood.(GeneralizedExtremeValue.(θ[2:end], 1, 0), Y)) 
            + (m - F.r) * log(θ[1]) / 2 
            - θ[1] * θ[2:end]' * F.G.W * θ[2:end] / 2 
            + logpdf(Gamma(1, 100), θ[1])
        )
        grad = ForwardDiff.gradient(logπ̃, θ)
        logπ̃(θ), grad
    end
    
    update = NUTSVariate(θ₀, logπ̃grad, target=0.4)
    
    for j = 2:niter
        sample!(update, adapt=true)
        θ[j, :, 1] = update
    end

    return θ
end