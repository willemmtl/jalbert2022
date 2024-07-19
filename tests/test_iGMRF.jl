using Test, Distributions

include("../iGMRF.jl");

@testset "iGMRF.jl" begin

    @testset "neighborsMutualEffect(F, μ)" begin

        F = iGMRF(2, 2, 1)

        μ = [1, 2, 3, 4]

        @test neighborsMutualEffect(F, μ) == [2.5, 2.5, 2.5, 2.5]

    end

    @testset "fcκᵤ(F, μ)" begin

        F = iGMRF(2, 2, 1)
        μ = [1, 1, 5, 3]

        @test fcκᵤ(F, μ) == Gamma(3, 100 / 1401)

    end

end