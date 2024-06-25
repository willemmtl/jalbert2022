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
function gibbs(niter::Integer, data::Vector{Vector{Float64}}; m₁::Integer, m₂::Integer, δ²::Real, κᵤ₀::Real, μ₀::Vector{<:Real})

    m = m₁ * m₂

    κᵤ = zeros(niter)
    κᵤ[1] = κᵤ₀
    μ = zeros(m, niter)
    μ[:, 1] = μ₀

    acc = falses(m, niter)

    for i = 2:niter
        # Generate μᵢ | κᵤ, μ₋ᵢ
        F = iGMRF(m₁, m₂, κᵤ[i-1])
        μ[:, i], acc[:, i] = sampleμ(F, μ, i, δ²=δ², data=data)
        # Generate κᵤ
        κᵤ[i] = rand(fcκᵤ(F, μ[:, i]))
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
function sampleμ(F::iGMRF, μ::Matrix{<:Real}, i::Integer; δ²::Real, data::Vector{Vector{Float64}})

    μꜝ = μ[:, i-1]

    acc = falses(F.G.m₁ * F.G.m₂)

    dLangevin = instrumentalMala(F, μꜝ, data, δ²)
    μ̃ = rand.(dLangevin)

    for j in eachindex(F.G.condIndSubsets)
        ind = F.G.condIndSubsets[j]
        accepted = subsetMala(F, μꜝ, μ̃, ind, data=data)
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
function subsetMala(F::iGMRF, μꜝ::Vector{<:Real}, μ̃::Vector{<:Real}, ind::Vector{<:Integer}; data::Vector{Vector{Float64}})

    logπ̃(μ::Vector{<:Real}) = logposteriori(μ, ind; F=F, data=data)
    logq(μᵢ₋₁::Vector{<:Real}, μ::Vector{<:Real}) = logpdf.(instrumentalMala(F, μᵢ₋₁, data, δ²), μ)[ind]

    lr = logπ̃(μ̃) .+ logq(μꜝ, μ̃) .- (logπ̃(μꜝ) .+ logq(μ̃, μꜝ))

    return lr .> log.(rand(length(ind)))

end


"""
    instrumentalMala(F, μ, y, δ²)

Compute the Instrumental Density for the MALA algorithm.
"""
function instrumentalMala(F::iGMRF, μ::Vector{<:Real}, data::Vector{Vector{Float64}}, δ²::Real)

    logπ̃(μ::Vector{<:Real}) = logposteriori(μ; F=F, data=data)
    ∇logπ̃(μ::Vector{<:Real}) = diag(ForwardDiff.jacobian(logπ̃, μ))

    return Normal.(μ + δ² / 2 * ∇logπ̃(μ), δ²)

end


"""
    logposteriori(μ; F, data)

Computes the "log-distribution" of the target law for the whole grid.

# Arguments

- `μ::Vector{<:Real}`: Location parameters.
- `F::iGMRF`: Inferred iGMRF.
- `data::Vector{Vector{Float64}}`: Observations for every cells
"""
function logposteriori(μ::Vector{<:Real}; F::iGMRF, data::Vector{Vector{Float64}})

    return datalevelloglike.(μ, data) .+ latentlevelloglike(F, μ)

end


"""
    logposteriori(μ, ind; F, data)

Computes the "log-distribution" of the target law for an entire partition.

# Arguments

- `μ::Vector{<:Real}`: Location parameters.
- `ind::Vector{<:Integer}`: Indices of the current partition.
- `F::iGMRF`: Inferred iGMRF.
- `data::Vector{Vector{Float64}}`: Observations for every cells
"""
function logposteriori(μ::Vector{<:Real}, ind::Vector{<:Integer}; F::iGMRF, data::Vector{Vector{Float64}})

    return datalevelloglike.(μ, data)[ind] .+ latentlevelloglike(F, μ, ind)

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