using Test, Distributions

include("../gibbs.jl");


@testset "gibbs.jl" begin

    @testset "neighborsMutualEffect(W, μ)" begin

        W = buildStructureMatrix(2, 2)

        μ = [1, 2, 3, 4]

        @test neighborsMutualEffect(W, μ) == [2.5, 2.5, 2.5, 2.5]

    end

    @testset "fcκᵤ(μ, W)" begin

        W = buildStructureMatrix(2, 2)
        μ = ones(4)

        @test fcκᵤ(μ, W=W) == Gamma(3, 100 / 401)

    end

end