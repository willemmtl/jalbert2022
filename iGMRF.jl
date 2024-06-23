using SparseArrays, LinearAlgebra

include("grid.jl")


struct iGMRF
    G::GridStructure
    r::Int64
    κᵤ::Float64
end


"""
    iGMRF(m₁, m₂, κᵤ)

Create a iGMRF object given the grid dimensions a the precision parameter.

# Arguments

- `m₁::Integer`: Number of rows of the grid.
- `m₂::Integer`: Number of columns of the grid.
- `κᵤ::Real`: Precision parameter.
"""
function iGMRF(m₁::Integer, m₂::Integer, κᵤ::Real)::iGMRF
    W = buildStructureMatrix(m₁, m₂)
    W̄ = W - spdiagm(diag(W))

    condIndSubsets = markCondIndSubsets(m₁, m₂)

    G = GridStructure(m₁, m₂, condIndSubsets, W, W̄)

    return iGMRF(G, 1, κᵤ)
end


"""
    buildStructureMatrix(m₁, m₂)

Build the 1-order iGMRF's structure matrix given the grid dimensions.

# Arguments

- `m₁::Integer`: Number of rows of the grid.
- `m₂::Integer`: Number of columns of the grid.
"""
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
    markCondIndSubsets(m₁, m₂)

Partition the grid in two parts, marking each cell with '1' or '2'.

The partitioning follows the Markov hypothesis : two cells within the same subset are conditionally independant.
This partitioning aims at accelerating computings.

# Arguments

- `m₁::Integer`: Number of rows of the grid.
- `m₂::Integer`: Number of columns of the grid.
"""
function markCondIndSubsets(m₁::Integer, m₂::Integer)::Vector{Vector{Integer}}

    condIndSubsetIndex = 2 * ones(Int64, m₁, m₂)
    condIndSubsetIndex[1:2:end, 1:2:end] .= 1
    condIndSubsetIndex[2:2:end, 2:2:end] .= 1

    return Array[findall(vec(condIndSubsetIndex) .== i) for i = 1:2]

end


"""
    sample(F)

Generate an iGMRF sample.

Use the GMRF under linear constraints method.

# Arguments

- `F::iGMRF`: iGMRF .
"""
function sample(F::iGMRF)::Vector{<:Real}
    # Paramètres
    m = F.G.m₁ * F.G.m₂
    # 1er vecteur propre de Q
    e₁ = ones(m, 1)
    A = e₁
    # Création de la matrice de précision (impropre ??)
    Q = F.κᵤ * F.G.W + e₁ * e₁' # ??
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
    Wₖₓₖ = A' * V
    # On calule Uₖₓₙ = W⁻¹Vᵀ
    U = Wₖₓₖ \ (V')
    # On calcule c = Ax - e
    c = A' * x
    # On calcule x* = x - Uᵀc
    y = x - U' * c

    return y
end


"""
    fcIGMRF(F, μ₀)

Compute the probability density of the full conditional function of the GEV's location parameter due to the iGMRF.

# Arguments

- `F::iGMRF`: Inferred iGMRF with the last update of the precision parameter.
- `μ::Vector{<:Real}`: Last updated location parameters.
"""
function fcIGMRF(F::iGMRF, μ::Vector{<:Real})

    Q = F.κᵤ * Array(diag(F.G.W))
    b = -F.κᵤ * (F.G.W̄ * μ)

    return NormalCanon.(b, Q)

end