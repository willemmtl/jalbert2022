using Distributions: loglikelihood

include("metropolis.jl")
include("GMRF.jl")

"""
    gibbs(niter, y; δ², κᵤ₀, μ₀, W)

Perform bayesian inference on the hierarchical model's parameters through the Gibbs algorithm.

Generation of the GEV's location parameter is performed through a one-iteration Metropolis algorithm.
Print the acceptation rate of the GEV's location parameter for each grid cell.

# Arguments

- `niter::Integer`: The number of iterations for the gibbs algorithm.
- `y::Array{Float64,3}`: Three-dimension array containing the observations of each grid cell.
- `δ²::Real`: Instrumental variance for the one-iteration Metropolis algorithm.
- `κᵤ₀::Real`: Initial value of the inferred GMRF's precision parameter.
- `μ₀::Vector{<:Real}`: Initial values of the inferred GEV's location parameter.
- `W::SparseMatrixCSC`: Structure matrix of the inferred GMRF.
"""
function gibbs(niter::Integer, y::Array{Float64,3}; δ²::Real, κᵤ₀::Real, μ₀::Vector{<:Real}, W::SparseMatrixCSC)

    m = length(μ₀)
    nobs = size(y, 3)
    y = reshape(y, m, nobs)

    κᵤ = zeros(niter)
    κᵤ[1] = κᵤ₀
    μ = zeros(m, niter)
    μ[:, 1] = μ₀

    acc = falses(m, niter)

    for i = 2:niter
        # Generate μᵢ | κᵤ, μ₋ᵢ
        for k = 1:m
            μ[k, i], acc[k, i] = updateμₖ(k, i, μ, δ², y[k, :], κᵤ[i-1], W)
        end
        # Generate κᵤ
        κᵤ[i] = rand(fcκᵤ(μ[:, i], W=W))
    end

    accRates = count(acc, dims=2) ./ (niter - 1) .* 100
    for k = 1:m
        println("Taux d'acceptation μ$k: ", round(accRates[k], digits=2), " %")
    end

    return κᵤ, μ
end


"""
    updateμₖ(k, i, μ, δ², y, κᵤ)

Update the inferred GEV's location parameter at iteration i of the gibbs algorithm.

Consist in a one-iteration Metropolis algorithm.

# Arguments

- `k::Integer`: Numero of the current cell.
- `i::Integer`: Numero of the current gibbs iteration.
- `μ::Matrix{<:Real}`: Current trace of the location parameter for each grid cell.
- `δ²::Real`: Instrumental variance.
- `y::Vector{<:Real}`: Observations for cell k.
- `κᵤ::Real`: Last updated value of the precision parameter.
"""
function updateμₖ(k::Integer, i::Integer, μ::Matrix{<:Real}, δ²::Real, y::Vector{<:Real}, κᵤ::Real, W::SparseMatrixCSC)

    μ̃ = rand(Normal(μ[k, i-1], δ²))
    μ₀ = μ[k, i-1]

    logL = datalevelloglike(μ̃, y) - datalevelloglike(μ₀, y)
    lf = latentlevelloglike(k, i, μ̃; κᵤ=κᵤ, W=W, μ=μ) - latentlevelloglike(k, i, μ₀; κᵤ=κᵤ, W=W, μ=μ)
    lr = logL + lf

    if lr > log(rand())
        return μ̃, true
    else
        return μ[k, i-1], false
    end
end


"""
    datalevelloglike(μ, y)

Compute the log-likelihhod at the data level evaluated at `μ` knowing the observations `y`.

# Arguments

- `μ::Real`: Location parameter.
- `y::Vector{<:Real}`: Observations.
"""
function datalevelloglike(μ::Real, y::Vector{<:Real})

    return loglikelihood(GeneralizedExtremeValue(μ, 1.0, 0.0), y)

end


"""
    latentlevelloglike(k, i, value; κᵤ, W, μ)

Compute the log-likelihhod at the latent level evaluated at `value` knowing the neighbors' location parameter.

# Arguments

- `k::Integer`: Numero of the current cell.
- `i::Integer`: Numero of the current gibbs iteration.
- `value`: Value for which the log-likelihood is evaluated.
- `κᵤ::Real`: Last updated value of the precision parameter.
- `W::SparseMatrixCSC`: Structure matrix of the inferred GMRF.
- `μ::Matrix{<:Real}`: Last updated trace of the location parameter for each grid cell.
"""
function latentlevelloglike(k::Integer, i::Integer, value::Real; κᵤ::Real, W::SparseMatrixCSC, μ::Matrix{<:Real})
    return logpdf(fcIGMRF(k, i, κᵤ=κᵤ, W=W, μ=μ), value)
end


"""
    fcIGMRF(k, i; κᵤ, W, μ)

Compute the probability density of the full conditional function of the GEV's location parameter due to the iGMRF.

# Arguments

- `k::Integer`: Numero of the current cell.
- `i::Integer`: Numero of the current gibbs iteration.
- `κᵤ::Real`: Last updated value of the precision parameter.
- `W::SparseMatrixCSC`: Structure matrix of the inferred GMRF.
- `μ::Matrix{<:Real}`: Last updated trace of the location parameter for each grid cell.
"""
function fcIGMRF(k::Integer, i::Integer; κᵤ::Real, W::SparseMatrixCSC, μ::Matrix{<:Real})

    # μ partially updated
    μUpdated = vcat(μ[1:k, i], μ[k+1:end, i-1])

    W̄ = W - spdiagm(diag(W))

    Q = κᵤ * Array(diag(W))
    b = -κᵤ * (W̄ * μUpdated)

    return NormalCanon(b[k], Q[k])

end


"""
    fcκᵤ(μ; W)

Compute the probability density of the full conditional function of the iGMRF's precision parameter.

# Arguments

- `μ::Matrix{<:Real}`: Last updated value of the location parameter for each grid cell.
- `W::SparseMatrixCSC`: Structure matrix of the inferred GMRF.
"""
function fcκᵤ(μ::Vector{<:Real}; W::SparseMatrixCSC)

    μ̄ = neighborsMutualEffect(W, μ)

    m = size(μ, 1)
    α = m / 2 + 1
    β = sum(dot(diag(W), (μ .- μ̄) .^ 2)) / 2 + 1 / 100

    return Gamma(α, 1 / β)

end


"""
    neighborsMutualEffect(W, u)

Compute the neighbors effect's term for each grid cell at the same time.

# Arguments

- `W::SparseMatrixCSC`: Structure matrix of the inferred GMRF.
- `μ::Vector{<:Real}`: Value of the location parameter for each grid cell.
"""
function neighborsMutualEffect(W::SparseMatrixCSC, μ::Vector{<:Real})

    W⁻ = W - spdiagm(diag(W))
    W⁻μ = W⁻ * μ

    return -spdiagm(1 ./ diag(W)) * W⁻μ

end