using SparseArrays, LinearAlgebra

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

"""
Génère un iGMRF à partir d'un GMRF sous contrainte linéaire.
"""
function generateIGMRF(κᵤ::Real, W::SparseMatrixCSC, m₁::Integer, m₂::Integer)
    # Paramètres
    m = m₁ * m₂
    r = 1
    # 1er vecteur propre de Q
    e₁ = ones(m, 1)
    A = e₁
    # Création de la matrice de précision (impropre ??)
    Q = κᵤ * W + e₁ * e₁' # ??
    # Factorisation de cholesky
    C = cholesky(Q)
    L = C.L
    # On génère z ∼ N(0, 1)
    z = randn(m)
    # On résout Lᵀv = z
    v = L' \ z
    # On calcule x = μ + v
    x = v # ??
    # On résout Vₙₓₖ = Q⁻¹Aᵀ
    V = C \ A
    # On calule Wₖₓₖ = AV
    W = A' * V
    # On calule Uₖₓₙ = W⁻¹Vᵀ
    U = W \ (V')
    # On calcule c = Ax - e
    c = A' * x
    # On calcule x* = x - Uᵀc
    y = x - U' * c

    return y
end