using Test, Distributions

include("../gibbs.jl");
include("../GMRF.jl");

@testset "gibbs.jl" begin

    @testset "neighborsMutualEffect(W, μ)" begin

        W = buildStructureMatrix(2, 2)

        μ = [1, 2, 3, 4]

        @test neighborsMutualEffect(W, μ) == [2.5, 2.5, 2.5, 2.5]

    end

    @testset "fcκᵤ(μ, W)" begin

        W = buildStructureMatrix(2, 2)
        μ = [1, 2, 3, 4]

        @test fcκᵤ(μ, W=W) == Gamma(3, 100 / 501)

    end

    @testset "fcIGMRF(k, i; κᵤ, W, μ)" begin

        k = 3
        i = 2

        κᵤ = 1
        W = buildStructureMatrix(2, 2)

        μ = zeros(4, 3)
        μ[:, 1] = ones(4)
        μ[1, 2] = 2
        μ[2, 2] = 3

        @test fcIGMRF(k, i; κᵤ=κᵤ, W=W, μ=μ) == NormalCanon(3, 2)

    end

    @testset "instrumentalMala(μ, f, δ²)" begin

        μ = 1
        f(x::Real) = x^2
        δ² = 2

        @test instrumentalMala(μ, f, δ²) == Normal(3, 2)

    end

end