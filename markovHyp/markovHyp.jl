using Mamba
using Distributions: loglikelihood

include("../iGMRF.jl")

"""
    gibbs(niter, y; δ², κᵤ₀, μ₀, W)

Perform bayesian inference through Gibbs algorithm.

Generation of the GEV's location parameter is performed through a one-iteration Metropolis algorithm.
Exploit Markov's hypothesis to save computation costs.

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

    for j = 2:niter
        # Generate μᵢ | κᵤ, μ₋ᵢ
        F = iGMRF(m₁, m₂, κᵤ[j-1])
        μ[:, j] = updateμ(F, μ, j, δ²=δ², y=y)
        # Generate κᵤ
        κᵤ[j] = rand(fcκᵤ(μ[:, j], W=F.G.W))
    end

    # Concatenate κᵤ's and μ's traces
    θ = vcat(reshape(κᵤ, (1, size(κᵤ, 1))), μ)
    # Build the Chains object
    names = [["κᵤ"]; ["μ$i" for i=1:m]]
    sim = Chains(copy(θ'), names=names)

    return sim
end


"""
    updateμ(F, μ, j; δ², y)

Perform one-step-Metropolis subset by subset.

# Arguments

- `F::iGMRF`: Inferred iGMRF with precision parameter's last update.
- `μ::Matrix{<:Real}`: Trace of all the location parameters.
- `j::Integer`: Current step of Gibbs sampler.
- `δ²::Real`: Instrumental variance for the one-iteration Metropolis algorithm.
- `y::Vector{Vector{Float64}}`: Vector containing the observations of each grid cell.
"""
function updateμ(F::iGMRF, μ::Matrix{<:Real}, j::Integer; δ²::Real, y::Vector{Vector{Float64}})

    μꜝ = μ[:, j-1]
    μ̃ = rand.(Normal.(μꜝ, δ²))

    logL = datalevelloglike.(μ̃, y) - datalevelloglike.(μꜝ, y)
    for j in eachindex(F.G.condIndSubsets)
        ind = F.G.condIndSubsets[j]
        accepted = subsetMetropolis(F, μꜝ, μ̃, logL, ind)
        setindex!(μꜝ, μ̃[ind][accepted], ind[accepted])
    end

    return μꜝ

end


"""
    subsetMetropolis(F, μꜝ, μ̃, logL, ind)

Perform one-step-Metropolis over a given subset.

# Arguments

- `F::iGMRF`: Inferred iGMRF with precision parameter's last update.
- `μꜝ::Vector{<:Real}`: Current state of all parameters.
- `μ̃::Vector{<:Real}`: Candidates for all parameters.
- `logL::Vector{<:Real}`: Data-level log-likelihood difference between μꜝ and μ̃ for each cell.
- `ind::Vector{<:Integer}`: Indices of current subset's cells.
"""
function subsetMetropolis(F::iGMRF, μꜝ::Vector{<:Real}, μ̃::Vector{<:Real}, logL::Vector{<:Real}, ind::Vector{<:Integer})

    pd = fcIGMRF(F, μꜝ)[ind]

    lf = logpdf.(pd, μ̃[ind]) .- logpdf.(pd, μꜝ[ind])

    lr = logL[ind] .+ lf

    return lr .> log.(rand(length(ind)))

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
    fcIGMRF(F, μ₀)

Compute the probability density of the full conditional function of the GEV's location parameter due to the iGMRF.

# Arguments

- `F::iGMRF`: Inferred iGMRF with the last update of the precision parameter.
- `μ::Vector{<:Real}`: Last updated location parameters.
"""
function fcIGMRF(F::iGMRF, μ::Vector{<:Real})

    Q = F.κᵤ * Array(diag(F.G.W))
    b = -F.κᵤ * (F.G.W̄ * μ)

    return NormalCanon.(b, Q)

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