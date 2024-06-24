using ForwardDiff
using Distributions: loglikelihood

include("iGMRF.jl")

"""
    gibbs(niter, y; δ², κᵤ₀, μ₀, W)

Perform bayesian inference on the hierarchical model's parameters through the Gibbs algorithm.

Generation of the GEV's location parameter is performed through a one-iteration Metropolis algorithm.
Print the acceptation rate of the GEV's location parameter for each grid cell.

# Arguments

- `niter::Integer`: The number of iterations for the gibbs algorithm.
- `y::Vector{Vector{Float64}}`: Vector containing the observations of each grid cell.
- `m₁::Integer`: Number of rows of the grid.
- `m₂::Integer`: Number of columns of the grid.
- `δ²::Real`: Instrumental variance for the one-iteration Metropolis algorithm.
- `κᵤ₀::Real`: Initial value of the inferred GMRF's precision parameter.
- `μ₀::Vector{<:Real}`: Initial values of the inferred GEV's location parameter.
"""
function gibbs(niter::Integer, y::Vector{Vector{Float64}}; m₁::Integer, m₂::Integer, δ²::Real, κᵤ₀::Real, μ₀::Vector{<:Real})

    m = m₁ * m₂

    κᵤ = zeros(niter)
    κᵤ[1] = κᵤ₀
    μ = zeros(m, niter)
    μ[:, 1] = μ₀

    acc = falses(m, niter)

    for i = 2:niter
        # Generate μᵢ | κᵤ, μ₋ᵢ
        F = iGMRF(m₁, m₂, κᵤ[i-1])
        μ[:, i], acc[:, i] = sampleμ(F, μ, i, δ²=δ², y=y)
        # Generate κᵤ
        κᵤ[i] = rand(fcκᵤ(μ[:, i], W=F.G.W))
    end

    accRates = count(acc, dims=2) ./ (niter - 1) .* 100
    for k = 1:m
        println("Taux d'acceptation μ$k: ", round(accRates[k], digits=2), " %")
    end

    return κᵤ, μ
end


"""
    sampleμ(F, μ, iInteger; δ², y)

Sample μ from the last updated iGMRF.

# Arguments

- `F::iGMRF`: iGMRF with its last updated parameters.
- `μ::Matrix{<:Real}`: Trace of each inferred GEV's location parameter.
- `i::Integer`: Numero of iteration.
- `δ²::Real`: Instrumental variance for the one-iteration Metropolis algorithm.
- `y::Vector{Vector{Float64}}`: Vector containing the observations of each grid cell.
"""
function sampleμ(F::iGMRF, μ::Matrix{<:Real}, i::Integer; δ²::Real, y::Vector{Vector{Float64}})

    μꜝ = μ[:, i-1]

    acc = falses(F.G.m₁ * F.G.m₂)

    qLangevin = instrumentalMala(F, μꜝ, y, δ²)
    μ̃ = rand.(qLangevin)

    for j in eachindex(F.G.condIndSubsets)
        ind = F.G.condIndSubsets[j]
        accepted = subsetMala(F, μꜝ, μ̃, ind)
        setindex!(μꜝ, μ̃[ind][accepted], ind[accepted])
        acc[ind[accepted]] .= true
    end

    return μꜝ, acc

end


"""
    subsetMala(F, μꜝ, μ̃, logL, ind)

Apply the one-iteration Metropolis algorithm to all of the cells in the same grid partition.

# Arguments

- `F::iGMRF`: iGMRF with its last updated parameters.
- `μꜝ::Vector{<:Real}`: The current state of μ (which is updated for each partition before being returned).
- `μ̃::Vector{<:Real}`: Candidates for the Metropolis algorithm.
- `logL::Vector{<:Real}`: Data-level log-likelihood difference for each cell (between candidates and last value).
- `ind::Vector{<:Integer}`: Cells' indices of the current partition.
"""
function subsetMala(F::iGMRF, μꜝ::Vector{<:Real}, μ̃::Vector{<:Real}, ind::Vector{<:Integer})

    pd = fcIGMRF(F, μꜝ)[ind]

    logπ̃(μ::Vector{<:Real}) = logpdf.(pd, μ[ind]) .+ datalevelloglike.(μ, y)[ind]
    logq(μᵢ₋₁::Vector{<:Real}, μ::Vector{<:Real}) = logpdf.(instrumentalMala(F, μᵢ₋₁, y, δ²), μ)[ind]

    lr = logπ̃(μ̃) .+ logq(μꜝ, μ̃) .- (logπ̃(μꜝ) .+ logq(μ̃, μꜝ))

    return lr .> log.(rand(length(ind)))

end


"""
    instrumentalMala(F, μ, y)

Compute the Instrumental Density for the MALA algorithm.
"""
function instrumentalMala(F::iGMRF, μ::Vector{<:Real}, y::Vector{Vector{Float64}}, δ²::Real)

    f(μ::Vector{<:Real}) = datalevelloglike.(μ, y) + latentlevelloglike(F, μ)
    ∇f = [ForwardDiff.derivative(f[i], μ[i]) for i in 1:length(μ)]

    return Normal.(μ + δ² / 2 * ∇f, δ²)

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
    latentlevelloglike(F, μ)

Compute the log-likelihhod at the data level evaluated at `μ` knowing the iGMRF `F`.
"""
function latentlevelloglike(F::iGMRF, μ::Vector{<:Real})

    pd = fcIGMRF(F, μ)

    return logpdf.(pd, μ)

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