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

function generateTargetGridV1(F::iGMRF)
    # Paramètre de position
    μ = generateGEVParamV1(F)
    # Paramètre d'échelle
    ϕ = zeros(F.m₁, F.m₂)
    # Paramètre de forme
    ξ = zeros(F.m₁, F.m₂)
    # Concatène les paramètres pour former la grille finale m₁xm₂x3
    return cat(μ, exp.(ϕ), ξ, dims=3)
end

function generateGEVParamV1(F::iGMRF)
    # Nombre total de cellules
    m = m₁ * m₂
    # Génère les effets spatiaux
    s = sample(F)
    # Il n'y a pas de variable explicative
    # On renvoie donc directement les effets spatiaux
    return reshape(s, m₁, m₂)'
end