using Distributions, ForwardDiff, LinearAlgebra

include("iGMRF.jl")

"""

"""
function malaDecomposed(niter::Integer, h::Vector{<:Real}, θ₀::Vector{<:Real}; Y::Vector{Vector{Float64}}, F::iGMRF)

    m = F.G.m₁ * F.G.m₂

    H = vcat([h[1]], fill(h[2], m))

    θ = zeros(m + 1, niter)
    θ[:, 1] = θ₀

    # Taux d'acceptation
    acc = falses(m + 1, niter)

    for j = 2:niter

        for i = 1:(m+1)
            
            dL(param::Real) = dLangevin(param, i, θ=θ[:, j-1], H=H, Y=Y, F=F)
            y = rand(dL(θ[i, j-1]))

            logπ̃(var::Real) = logposteriori(var, i, θ=θ[:, j-1], Y=Y, F=F)
            logq(y::Real, x::Real) = logpdf(dL(x), y)
            
            lr = logπ̃(y) + logq(θ[i, j-1], y) - logπ̃(θ[i, j-1]) - logq(y, θ[i, j-1])

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
    dLangevin(param, i; θ, H, Y, F)

Langevin distribution for the i-th parameter.

# Arguments

- `param::Real`: Parameter's value of the distribution.
- `i::Integer`: Index of this parameter in the parameters' vector.
- `θ::Vector{<:Real}`: All parameters [κᵤ, μ₁, ..., μₘ] (needed for ∂logπ̃).
- `H::Vector{<:Real}`: Instrumental variances.
- `Y::Vector{Vector{Float64}}`: All observations.
- `F::iGMRF`: iGMRF giving the grid structure.
"""
function dLangevin(param::Real, i::Integer; θ::Vector{<:Real}, H::Vector{<:Real}, Y::Vector{Vector{Float64}}, F::iGMRF)

    
    logπ̃(var::Real) = logposteriori(var, i, θ=θ, Y=Y, F=F)
    ∂logπ̃(var::Real) = ForwardDiff.derivative(logπ̃, var)

    return Normal(param + H[i] * ∂logπ̃(param) / 2, H[i])

end


"""
    logposteriori(var, i; θ, Y, F)

Build a partial target function toward the i-th component of the parameter vector.

This function has been made in order to compute partial derivatives.

# Arguments

- `var::Real`: Parameter's value to which the target function is evaluated.
- `i::Integer`: Index of this parameter in the parameters' vector.
- `θ::Vector{<:Real}`: All parameters [κᵤ, μ₁, ..., μₘ].
- `Y::Vector{Vector{Float64}}`: All observations.
- `F::iGMRF`: iGMRF giving the grid structure.
"""
function logposteriori(var::Real, i::Integer; θ::Vector{<:Real}, Y::Vector{Vector{Float64}}, F::iGMRF)

    # println("θ = ", θ)
    # println("vcat(θ[1:i-1], [var], θ[i+1:end]) = ", vcat(θ[1:i-1], [var], θ[i+1:end]))

    return logposteriori(vcat(θ[1:i-1], [var], θ[i+1:end]), Y=Y, F=F)

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