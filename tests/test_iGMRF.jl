using Test, Distributions

include("../iGMRF.jl");


@testset "iGMRF.jl" begin

    @testset "fcIGMRF(F, μ)" begin

        F = iGMRF(2, 2, 1)
        μ = [1, 2, 2, 4]

        @test fcIGMRF(F, μ) == [NormalCanon(4, 2), NormalCanon(5, 2), NormalCanon(5, 2), NormalCanon(4, 2)]

    end

end