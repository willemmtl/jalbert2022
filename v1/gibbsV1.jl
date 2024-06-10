using Distributions: loglikelihood

include("metropolis.jl")
include("GMRF.jl")

function gibbs(niter::Integer, y::Array{Float64,3}; δ²::Real, κᵤ₀::Real, μ₀::Vector{<:Real}, W::SparseMatrixCSC)

    # Nombre de cellules
    m = length(μ₀)
    # Redimensionnement des données
    nobs = size(y, 3)
    y = reshape(y, m, nobs)

    # Initialisation
    κᵤ = zeros(niter)
    κᵤ[1] = κᵤ₀
    μ = zeros(m, niter)
    μ[:, 1] = μ₀

    # Taux d'acceptation des μ
    acc = falses(m, niter)

    # Itérations
    for i = 2:niter
        # On génère les μᵢ | κᵤ, μ₋ᵢ
        for k = 1:9
            μ[k, i], acc[k, i] = updateμₖ(k, i, μ, δ², y[k, :], κᵤ[i-1])
        end
        # μ[:, i] = [0.0211641 0.0153844 -0.117718 0.077531 0.00983722 -0.0508229 0.0955774 -0.0244998 -0.0264531]
        # On génère κᵤ
        μ̄ᵢ = neighborsMutualEffect(W, μ[:, i])
        κᵤ[i] = rand(fcκᵤ(W, μ[:, i], μ̄ᵢ))
    end

    accRates = count(acc, dims=2) ./ (niter - 1) .* 100
    for k = 1:m
        println("Taux d'acceptation μ$k: ", round(accRates[k], digits=2), " %")
    end

    return κᵤ, μ
end

"""
Tentative de réorganisaion des itérations --> mêmes résultats...
"""
function updateμₖ(k::Integer, i::Integer, μ::Matrix{<:Real}, δ²::Real, y::Vector{<:Real}, κᵤ::Real)
    # On génère un candidat pour Ui
    # Et donc pour μ
    μ̃ = rand(Normal(μ[k, i-1], δ²))
    # État précédent
    μ₀ = μ[k, i-1]
    # Calcul de la différence de log-vraisemblance au niveau des données
    logL = loglikelihood(GeneralizedExtremeValue(μ̃, 1.0, 0.0), y) - loglikelihood(GeneralizedExtremeValue(μ₀, 1.0, 0.0), y)
    # Calcul de la différence de log-vraisemblance au niveau latent
    lf = logpdf(iGMRFfc(k, i, κᵤ, W, μ), μ̃) - logpdf(iGMRFfc(k, i, κᵤ, W, μ), μ[k, i-1])
    # Somme des deux différences
    lr = logL + lf
    # Acceptation
    if lr > log(rand())
        return μ̃, true
    else
        return μ[k, i-1], false
    end
end

"""
Loi conditionnelle complète du iGMRF
"""
function iGMRFfc(k::Integer, i::Integer, κᵤ::Real, W::SparseMatrixCSC, μ::Matrix{<:Real})

    # μ partially updated
    μUpdated = vcat(μ[1:k, i], μ[k+1:end, i-1])

    # Structure matrix without its diagonal
    W̄ = W - spdiagm(diag(W))

    # Parameters of the canonical Normal
    Q = κᵤ * Array(diag(W))
    b = -κᵤ * (W̄ * μUpdated)

    pd = NormalCanon(b[k], Q[k])

    return pd

end

"""
Distribution conditionnelle complète pour κᵤ.
"""
function fcκᵤ(W::SparseMatrixCSC, μ::Vector{<:Real}, μ̄::Vector{<:Real})
    m = size(μ, 1)
    α = m / 2 + 1
    β = sum(dot(diag(W), (μ .- μ̄) .^ 2)) / 2 + 1 / 100
    return Gamma(α, 1 / β)
end

"""
Calcul des effets mutuels des voisins.
"""
function neighborsMutualEffect(W::SparseMatrixCSC, u::Vector{<:Real})
    # Annule la diagonale de W⁻
    W⁻ = W - spdiagm(diag(W))
    # Somme pondérée par les uⱼ
    W⁻u = W⁻ * u
    # Multiplication par l'inverse de la diagonale de W
    return -spdiagm(1 ./ diag(W)) * W⁻u
end

"""
Calcul des effets des voisins sur la cellule k
A partir des paramètres les plus à jour possibles.
"""
function neighborsEffect(i::Integer, k::Integer, W::SparseMatrixCSC, μ::Matrix{<:Real})
    # Considérons qu'on s'intéresse à μₖ
    # Pour p < k, les μₚ ont déjà été mis à jour (μ[p, i])
    # On utilise la version précédente pour les p > k (μ[p, i-1])
    μUpdated = vcat(μ[1:k, i], μ[k+1:end, i-1])
    # Somme pondérée par les coefficients de la matrice W
    # μ[k, i] = 0 ce qui annule le coefficient Wₖₖ
    return -1 / W[k, k] * sum(μUpdated' * W[k, :])
end