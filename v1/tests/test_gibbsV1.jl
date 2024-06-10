using Test, Distributions

include("../gibbsV1.jl");
include("../GMRF.jl");

@testset "gibbsV1.jl" begin

    @testset "neighborsEffect(i, k, W, μ)" begin
        # Matrice de structure
        W = buildStructureMatrix(2, 2)
        # Trace de μ partiellement mis à jour
        μ = zeros(4, 3)
        μ[:, 1] = ones(4)
        μ[1, 2] = 2
        μ[2, 2] = 3
        # Calcul de μ₃ à l'itération 2 de gibbs
        @test neighborsEffect(2, 3, W, μ) ≈ 3 / 2
    end

    @testset "neighborsMutualEffect(W, u)" begin
        # Matrice de structure
        W = buildStructureMatrix(2, 2)
        # μ précédemment calculés
        μ = [1, 2, 3, 4]
        # Calcul des μ̄
        @test neighborsMutualEffect(W, μ) == [2.5, 2.5, 2.5, 2.5]
    end

    @testset "fcκᵤ(W, μ, μ̄)" begin
        W = buildStructureMatrix(2, 2)
        μ = ones(4)
        μ̄ = zeros(4)
        @test fcκᵤ(W, μ, μ̄) == Gamma(3, 100 / 401)
    end

end