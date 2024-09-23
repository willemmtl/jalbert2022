using Mamba, Distributions, ForwardDiff, LinearAlgebra

include("../iGMRF.jl")

"""
    amWG(niter, h, θ₀; Y, F, nchains)

Perform an Adaptative Metropolis Within Gibbs algorithm.

# Arguments

- `niter::Integer`: Number of gibbs iterations.
- `h::Real`: Instrumental variance of MALA.
- `θ₀::Vector{<:Real}`: Initial values.
- `Y::Vector{Vector{Float64}}`: Observations.
- `F::iGMRF`: iGMRF (no need to be updated).
- `nchains::Integer` : Number of chaines for each parameter.
"""
function amWG(niter::Integer, h::Real, θ₀::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF, nchains::Integer)

    m = F.G.m₁ * F.G.m₂
    κᵤ = zeros(niter, nchains)
    κᵤ[1, :] = fill(θ₀[1], nchains)
    
    names = ["μ$i" for i=1:m]
    μ = Chains(niter, m, names=names, chains=nchains)
    μ[1, :, :] = repeat(θ₀[2:end], 1, nchains)

    for numc = 1:nchains
        for j = 2:niter
            # Generate μ
            ## Log-transformed Posterior(μ | κᵤ) + Constant and Gradient Vector
            logπ̃grad = function(μ::DenseVector)
                logπ̃(μ::DenseVector) = sum(loglikelihood.(GeneralizedExtremeValue.(μ, 1, 0), Y)) - κᵤ[j-1, numc] * μ' * F.G.W * μ / 2
                # grad = ForwardDiff.gradient(logπ̃, μ)
                logπ̃(μ)#, grad
            end
            
            theta = AMWGVariate(μ.value[j-1, :, numc], h, logπ̃grad)
            sample!(theta)
            μ[j, :, numc] = theta

            # Generate κᵤ
            κᵤ[j, numc] = rand(fcκᵤ(F, μ.value[j, :, numc]))
        end
    end
    
    return κᵤ, μ
end