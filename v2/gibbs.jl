using Distributions

include("metropolis.jl")
include("GMRF.jl")

function gibbs(niter::Integer, y::Array{Float64,3}; itMetropolis::Integer, δ²::Vector{<:Real}, κᵤ₀::Real, u₀::Vector{<:Real}, μ₀::Vector{<:Real}, W::SparseMatrixCSC)

    # Nombre de cellules
    m = length(μ₀)
    # Insuffisance de rang de W
    r = 1
    # Redimensionnement des données
    nobs = size(y, 3)
    y = reshape(y, m, nobs)

    # Initialisation
    κᵤ = zeros(niter)
    κᵤ[1] = κᵤ₀
    u = zeros(m, niter)
    u[:, 1] = u₀
    μ = zeros(m, niter)
    μ[:, 1] = μ₀

    # Variance instrumentale pour Metropolis
    δ²κ, δ²u, δ²μ = δ²

    # Itérations
    for i = 2:niter
        # On génère κᵤ
        f̃κᵤ(κᵤ::Real) = fcκᵤ(κᵤ, W=W, u=u[:, i-1])
        # κᵤ[i] = metropolisUnivariate(itMetropolis, θ₀=κᵤ[i-1], δ²=δ²κ, f̃=f̃κᵤ, printAcc=false)[end]
        κᵤ[i] = 100
        # On génère u
        u[:, i] = generateSpatialEffect(itMetropolis, θ₀=u[:, i-1], m=m, r=r, κ=κᵤ[i], W=W, σ²=δ²u)
        # On génère les μᵢ | κᵤ, u
        Ū = generateŪ(W, u[:, i])
        for k = 1:9
            f̃μ(μ::Real) = fcμ(μ, y=y[k, :], κᵤ=κᵤ[i], Wᵢᵢ=W[k, k], ūᵢ=Ū[k])
            μ[k, i] = metropolisUnivariate(itMetropolis, θ₀=μ[k, i-1], δ²=δ²μ, f̃=f̃μ, printAcc=false)[end]
        end
    end

    return κᵤ, u, μ
end

"""
Fonction de densité de la loi conditionnelle complète de μ.
"""
function fcμ(μ::Real; y::Array, κᵤ::Real, Wᵢᵢ::Real, ūᵢ::Real)
    return sum(logpdf.(GeneralizedExtremeValue.(μ, 1.0, 0.0), y)) + logpdf(Normal(ūᵢ, 1 / (κᵤ * Wᵢᵢ)), μ)
end

"""
Fonction de log-densité de la loi conditionnelle complète de κᵤ
"""
function fcκᵤ(κᵤ::Real; W::SparseMatrixCSC, u::Vector{<:Real})
    if κᵤ < 0
        return -Inf
    else
        Ū = generateŪ(W, u)
        return sum(logpdf.(Normal.(Ū, 1 ./ (κᵤ .* diag(W))), u)) - κᵤ / 100
    end
end

"""
Calcul intermédiaire qui sera utile par la suite
"""
function generateŪ(W::SparseMatrixCSC, u::Vector{<:Real})
    # Annule la diagonale de W⁻
    W⁻ = W - spdiagm(diag(W))
    # Somme pondérée par les uⱼ
    W⁻u = W⁻ * u
    # Multiplication par l'inverse de la diagonale de W
    return -spdiagm(1 ./ diag(W)) * W⁻u
end
