using SparseArrays

function buildStructureMatrix(m₁::Integer, m₂::Integer)
    # Vecteur des voisins horizontaux
    v = ones(Int64, m₁)
    v[end] = 0
    V = repeat(v, outer=m₂)
    pop!(V)
    # Vecteur des voisins verticaux
    U = ones(Int64, m₁ * (m₂ - 1))
    # Nb total de cellules
    m = m₁ * m₂
    # Matrice du voisinage
    D = sparse(1:(m-1), 2:m, V, m, m) + sparse(1:(m-m₁), (m₁+1):m, U, m, m)
    D = D + D'
    # Compte le nombre de voisins de chaque cellule
    nbs = fill(Int[], m)
    for i = 1:m
        # Détermine les 1 de la colonne i, i.e. les voisins de la cellule i
        nbs[i] = findall(!iszero, D[:, i])
    end
    # Compte le nombre de voisins et remplit la diagonale
    return -D + spdiagm(0 => length.(nbs))
end

function generateSpatialEffect(niter::Integer; θ₀::Vector{<:Real}, m::Integer, r::Integer, κ::Real, W::SparseMatrixCSC, σ²::Real)
    # Pour U
    f̃(θ::Vector{<:Real}) = log(κ) * (m - r) / 2 - κ * (θ' * W * θ) / 2
    u = metropolisMultivariate(niter, θ₀=θ₀, σ²=σ², f̃=f̃, printAcc=false)[:, end]
    # Renvoie le dernier échantillon
    return u
end