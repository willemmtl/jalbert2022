using Mamba, Distributions, ForwardDiff, LinearAlgebra

include("iGMRF.jl")

"""
    malaWG(niter, h, θ₀; Y, F)

Perform a MALA Within Gibbs algorithm.

# Arguments

- `niter::Integer`: Number of gibbs iterations.
- `h::Real`: Instrumental variance of MALA.
- `θ₀::Vector{<:Real}`: Initial values.
- `Y::Vector{Vector{Float64}}`: Observations.
- `F::iGMRF`: iGMRF (no need to be updated).
- `nchains::Integer` : Number of chaines for each parameter.
"""
function malaWG(niter::Integer, h::Real, θ₀::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF, nchains::Integer)

    m = F.G.m₁ * F.G.m₂
    κᵤ = zeros(niter, nchains)
    κᵤ[1, :] = fill(θ₀[1], nchains)
    
    μ = Chains(niter, m, names = ["μ1", "μ2", "μ3", "μ4", "μ5", "μ6", "μ7", "μ8", "μ9"], chains=nchains)
    μ[1, :, :] = repeat(θ₀[2:end], 1, nchains)

    acc = [[false] for _ in 1:nchains]

    for numc = 1:nchains
        for j = 2:niter

            # Generate κᵤ
            κᵤ[j, numc] = rand(fcκᵤ(F, μ.value[j-1, :, numc]))

            # Generate μ
            ## Log-transformed Posterior(μ | κᵤ) + Constant and Gradient Vector
            logπ̃grad = function(μ::DenseVector)
                logπ̃(μ::DenseVector) = sum(loglikelihood.(GeneralizedExtremeValue.(μ, 1, 0), Y)) - κᵤ[j, numc] * μ' * F.G.W * μ / 2
                grad = ForwardDiff.gradient(logπ̃, μ)
                logπ̃(μ), grad
            end
            
            theta = MALAVariate(μ.value[j-1, :, numc], h, logπ̃grad)
            sampleCandidate!(theta, acc[numc])
            μ[j, :, numc] = theta

        end
    end
    
    for numc = 1:nchains
        println("Taux d'acceptation chaîne $numc: ", count(acc[numc]) * 100 / (niter-1))
    end
    
    return κᵤ, μ
end


sampleCandidate!(v::MALAVariate, acc::Vector{<:Real}) = sampleCandidate!(v, v.tune.logfgrad, acc)

function sampleCandidate!(v::MALAVariate, logfgrad::Function, acc::Vector{<:Real})
  tune = v.tune

  L = sqrt(tune.epsilon) * tune.SigmaL
  Linv = inv(L)
  M2 = 0.5 * L * L'

  logf0, grad0 = logfgrad(v.value)
  y = v + M2 * grad0 + L * randn(length(v))
  logf1, grad1 = logfgrad(y)

  q0 = -0.5 * sum(abs2, Linv * (v - y - M2 * grad1))
  q1 = -0.5 * sum(abs2, Linv * (y - v - M2 * grad0))

  if rand() < exp((logf1 - q1) - (logf0 - q0))
    v[:] = y
    push!(acc, true)
  else
    push!(acc, false)
  end

  v
end