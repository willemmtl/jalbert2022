using Distributions

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
            μ̄ₖ = neighborsEffect(i, k, W, μ)
            f̃μ(μ::Real) = fcμ(μ, y=y[k, :], κᵤ=κᵤ[i-1], Wₖₖ=W[k, k], μ̄ₖ=μ̄ₖ)
            μ[k, i], acc[k, i] = metropolisOneStep(μ[k, i-1], δ², f̃μ)
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
Fonction de densité de la loi conditionnelle complète de μ.
"""
function fcμ(μ::Real; y::Array, κᵤ::Real, Wₖₖ::Real, μ̄ₖ::Real)
    return sum(logpdf.(GeneralizedExtremeValue(μ, 1.0, 0.0), y)) + logpdf(Normal(μ̄ₖ, 1 / (κᵤ * Wₖₖ)), μ)
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

# ------------------------------------------------

"""
Tentative de réorganisaion des itérations --> mêmes résultats...
"""
function updateμₖ(k::Integer, i::Integer, u::Matrix{<:Real}, δ²::Real, y::Vector{<:Real}, κᵤ::Real)
    # On génère un candidat pour Ui
    ũ = rand(Normal(u[k, i-1], δ²))
    # On en déduit un candidat pour μ
    μ̃ = ũ
    μ₀ = u[k, i-1]
    # Calcul de la différence de log-vraisemblance au niveau des données
    logL = loglikelihood(GeneralizedExtremeValue(μ̃, 1.0, 0.0), y) - loglikelihood(GeneralizedExtremeValue(μ₀, 1.0, 0.0), y)
    # Calcul de la différence de log-vraisemblance au niveau latent
    ūₖ = neighborsEffect(i, k, W, u)
    Wₖₖ = W[k, k]
    lf = logpdf(Normal(ūₖ, 1 / (κᵤ * Wₖₖ)), ũ) - logpdf(Normal(ūₖ, 1 / (κᵤ * Wₖₖ)), u[k, i-1])
    # Somme des deux différences
    lr = logL + lf
    # Acceptation
    if lr > log(rand())
        return ũ, true
    else
        return u[k, i-1], false
    end
end


"""
Fonction de densité de la loi conditionnelle complète de U.
"""
function fcU(u::Real; κᵤ::Real, Wᵢᵢ::Real, i::Integer, k::Integer)
    ūᵢ = neighborsEffect(i, k, W, μ)
    return loglikelihood(Normal(ūᵢ, 1 / (κᵤ * Wᵢᵢ)), u)
end