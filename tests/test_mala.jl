using Test, Distributions

include("../iGMRF.jl");
include("../mala.jl");


@testset "mala.jl" begin

    @testset "logposteriori(θ; Y, F)" begin

        F = iGMRF(2, 2, 1)
        θ = [1, 2, 0, 1, 3]
        Y = [
            [3.0, 7.0],
            [1.0, 6.0],
            [8.0, 4.0],
            [2.0, 5.0]
        ]

        @test logposteriori(θ, Y=Y, F=F) ≈ (
            (logpdf(GeneralizedExtremeValue(2, 1, 0), 3) + logpdf(GeneralizedExtremeValue(2, 1, 0), 7))
            + (logpdf(GeneralizedExtremeValue(0, 1, 0), 1) + logpdf(GeneralizedExtremeValue(0, 1, 0), 6))
            + (logpdf(GeneralizedExtremeValue(1, 1, 0), 8) + logpdf(GeneralizedExtremeValue(1, 1, 0), 4))
            + (logpdf(GeneralizedExtremeValue(3, 1, 0), 2) + logpdf(GeneralizedExtremeValue(3, 1, 0), 5))
            - 9
            + logpdf(Gamma(1, 100), 1)
        )

    end

end