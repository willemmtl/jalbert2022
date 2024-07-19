using Distributions, ForwardDiff, LinearAlgebra

include("iGMRF.jl")

"""

"""
function malaDecomposedBis(niter::Integer, h::Vector{<:Real}, θ₀::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF)

    m = F.G.m₁ * F.G.m₂

    H = vcat([h[1]], fill(h[2], m))

    θ = zeros(m + 1, niter)
    θ[:, 1] = θ₀

    # Taux d'acceptation
    acc = falses(m + 1, niter)

    logπ̃(θ::Vector{<:Real}) = logposteriori(θ, Y=Y, F=F)
    ∇logπ̃(θ::Vector{<:Real}) = ForwardDiff.gradient(logπ̃, θ)

    for j = 2:niter

        for i = 1:(m+1)
            
            y = rand(Normal(θ[i, j-1] + H[i] * ∇logπ̃(θ[:, j-1])[i] / 2, H[i]))
            
            θy = θ[:, j-1]
            θy[i] = y

            qy_θ = logpdf(Normal(θ[i, j-1] + H[i] * ∇logπ̃(θ[:, j-1])[i] / 2, H[i]), y)
            qθ_y = logpdf(Normal(y + H[i] * ∇logπ̃(θy)[i] / 2, H[i]), θ[i, j-1])

            lr = logπ̃(θy) + qθ_y - logπ̃(θ[:, j-1]) - qy_θ
            
            if log(rand()) < lr
                acc[i, j] = true
                θ[i, j] = y
            else
                θ[i, j] = θ[i, j-1]
            end
        end

    end

    accRates = count(acc, dims=2) ./ (niter - 1) .* 100
    println("Taux d'acceptation de κᵤ: ", round(accRates[1], digits=2), " %")
    for i = 2:m+1
        println("Taux d'acceptation de μ$(i-1): ", round(accRates[i], digits=2), " %")
    end

    return θ
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