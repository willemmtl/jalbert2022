using Test

@testset "utils.jl" begin
    
    @testset "reshape(v, m₁, m₂)" begin
        
        v = [1, 2, 3, 4, 5, 6, 7, 8, 9]

        r = reshape(v, 3, 3)

        @test r'[1, 2] == 2
        @test r'[1, 3] == 3
        @test r'[2, 1] == 4
        @test r'[2, 3] == 6
        @test r'[3, 2] == 8

    end

end