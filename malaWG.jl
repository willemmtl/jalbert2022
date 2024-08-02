using Mamba, Distributions, ForwardDiff, LinearAlgebra

include("iGMRF.jl")

"""

"""
function malaWG(niter::Integer, h::Real, θ₀::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF)

    m = F.G.m₁ * F.G.m₂

    κᵤ = zeros(niter)
    κᵤ[1] = θ₀[1]

    μ = Chains(niter, m, names = ["μ1", "μ2", "μ3", "μ4", "μ5", "μ6", "μ7", "μ8", "μ9"])
    μ[1, :, 1] = zeros(m)

    for j = 2:niter

        # Generate κᵤ
        κᵤ[j] = rand(fcκᵤ(F, μ.value[j-1, :, 1]))

        # Generate μ
        ## Log-transformed Posterior(μ | κᵤ) + Constant and Gradient Vector
        logπ̃grad = function(μ::DenseVector)
            logπ̃(μ::DenseVector) = sum(loglikelihood.(GeneralizedExtremeValue.(μ, 1, 0), Y)) - κᵤ[j] * μ' * F.G.W * μ / 2
            grad = ForwardDiff.gradient(logπ̃, μ)
            logπ̃(μ), grad
        end
        
        theta = MALAVariate(μ.value[j-1, :, 1], h, logπ̃grad)
        sample!(theta)
        μ[j, :, 1] = theta
        
    end

    return hcat(κᵤ, μ.value)
end