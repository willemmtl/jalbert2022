using Random, Distributions, SparseArrays

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

function generateTargetGridV1(; m₁::Integer, m₂::Integer, κᵤ::Real, W::SparseMatrixCSC)
    # Paramètre de position
    μ = generateGEVParamV1(m₁=m₁, m₂=m₂, κᵤ=κᵤ, W=W)
    # Paramètre d'échelle
    ϕ = zeros(m₁, m₂)
    # Paramètre de forme
    ξ = zeros(m₁, m₂)
    # Concatène les paramètres pour former la grille finale m₁xm₂x3
    return cat(μ, exp.(ϕ), ξ, dims=3)
end

function generateGEVParamV1(; m₁::Integer, m₂::Integer, κᵤ::Real, W::SparseMatrixCSC)
    # Nombre total de cellules
    m = m₁ * m₂
    # Génère les effets spatiaux
    s = generateIGMRF(κᵤ, W, m₁, m₂)
    # Il n'y a pas de variable explicative
    # On renvoie donc directement les effets spatiaux
    return reshape(s, m₁, m₂)'
end