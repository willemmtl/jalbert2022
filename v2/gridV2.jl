using Random, Distributions, SparseArrays

include("metropolis.jl")
include("GMRF.jl")

function generateData(grid_params::Array{<:Real}, nobs::Integer)

    # Array qui contiendra les observations pour chaque cellule
    grid_obs = zeros(size(grid_params, 1), size(grid_params, 2), nobs)

    for i = 1:size(grid_params, 1)
        for j = 1:size(grid_params, 2)
            # Récupération des paramètres de la cellule courante
            gev_params = grid_params[i, j, :]
            # Génération des observations pour la cellule courante
            grid_obs[i, j, :] = rand(GeneralizedExtremeValue(gev_params...), nobs)
        end
    end

    return grid_obs
end

function generateTargetGridV2(; m₁::Integer, m₂::Integer, r::Integer, κ::Vector{<:Real}, W::SparseMatrixCSC, σ²::Vector{<:Real})
    # Déballage des κ
    κᵤ, κᵥ = κ
    # Déballage des σ²
    σ²ᵤ, σ²ᵥ = σ²
    # Paramètre de position
    μ = generateGEVParamV2(m₁=m₁, m₂=m₂, r=r, κ=κᵤ, W=W, σ²=σ²ᵤ)
    # Paramètre d'échelle
    ϕ = generateGEVParamV2(m₁=m₁, m₂=m₂, r=r, κ=κᵥ, W=W, σ²=σ²ᵥ)
    # Paramètre de forme
    ξ = rand(Beta(6, 9), 3, 3) .- 0.5
    # Concatène les paramètres pour former la grille finale m₁xm₂x3
    return cat(μ, exp.(ϕ), ξ, dims=3)
end

function generateGEVParamV2(; m₁::Integer, m₂::Integer, r::Integer, κ::Real, W::SparseMatrixCSC, σ²::Real)
    # Nombre total de cellules
    m = m₁ * m₂
    # Génère les effets spatiaux
    s = generateSpatialEffect(10000, θ₀=zeros(m), m=m, r=r, κ=κ, W=W, σ²=σ²)
    # Il n'y a pas de variable explicative
    # On renvoie donc directement les effets spatiaux
    return reshape(s, m₁, m₂)'
end