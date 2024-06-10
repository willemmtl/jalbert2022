using Distributions, LinearAlgebra

function metropolisMultivariate(niter::Integer; θ₀::Vector{<:Real}, σ²::Real, f̃::Function, printAcc::Bool)
    # Taille du vecteur des parametres
    n = length(θ₀)
    # Matrice qui contiendra la trace
    θ = zeros(n, niter)
    θ[:, 1] = θ₀
    # Vecteur des acceptations
    acc = falses(niter)
    # Iterations
    for i = 2:niter
        # Generation d'un candidat
        # Loi Normale Multivariée centrée à l'état précédent et de variance σ²I
        y = rand(MvNormal(θ[:, i-1], σ² * I))
        # Acceptation
        logr = f̃(y) - f̃(θ[:, i-1])
        if log(rand()) < logr
            θ[:, i] = y
            acc[i] = true
        else
            θ[:, i] = θ[:, i-1]
        end
    end
    # Affichage du taux d'acceptation
    if printAcc
        println("Taux d'acceptation : ", count(acc) ./ (niter - 1) .* 100, "%")
    end
    # Renvoie la trace
    return θ
end

function metropolisUnivariate(niter::Integer; θ₀::Real, δ²::Real, f̃::Function, printAcc::Bool)
    # Résultat de chaque itération
    θ = zeros(Float64, niter)

    # Valeurs initiales
    θ[1] = θ₀
    # Taux d'acceptation
    acc = falses(niter)

    for i = 2:niter
        # Génère un candidat à l'aide de la distribution instrumentale
        # Une normale centrée à l'état précédent et de variance donnée
        y = rand(Normal(θ[i-1], δ²))
        # Probabilité d'acceptation (log)
        r = f̃(y) - f̃(θ[i-1])
        # Phase d'acceptation
        if log(rand()) < r
            θ[i] = y
            acc[i] = true
        else
            θ[i] = θ[i-1]
        end
    end

    if printAcc
        println("Taux d'acceptation : ", count(acc) ./ (niter - 1) .* 100, "%")
    end

    return θ
end

"""
Metropolis à une seule itération.
"""
function metropolisOneStep(θᵢ₋₁::Real, δ²::Real, f̃::Function)
    # Génère un candidat
    # Une normale centrée à l'état précédent et de variance δ² donnée
    y = rand(Normal(θᵢ₋₁, δ²))
    # Probabilité d'acceptation (log)
    r = f̃(y) - f̃(θᵢ₋₁)
    # Phase d'acceptation
    if log(rand()) < r
        return y, true
    else
        return θᵢ₋₁, false
    end
end