using Statistics

include("iGMRF.jl");
include("dataGen.jl");
include("malaWG.jl");

function evaluate(N; nobs::Integer, niter::Integer)

    Random.seed!(400);

    m₁ = 3;
    m₂ = 3;
    κᵤ = 100.0;
    F = iGMRF(m₁, m₂, κᵤ);
    grid_target = generateTargetGrid(F);
    data = generateData(grid_target, nobs);

    h = 0.0006;
    m = m₁ * m₂;
    θ₀ = vcat([10], zeros(m));
    F = iGMRF(3, 3, 1);

    κ̂ᵤ = zeros(N);
    distances = zeros(N);
    times = zeros(N);

    for n=1:N
        startTime = time()
        θ = malaWG(niter, h, θ₀, Y=data, F=F);
        endTime = time()
        times[n] = endTime - startTime

        μ̂ = θ[2:end, end];
        κ̂ᵤ[n] = θ[1, end];

        distances[n] = norm(reshape(μ̂[:, end], m₁, m₂) .- grid_target[:, :, 1], 2) / m
    end

    println("κ̂ᵤ = ", mean(κ̂ᵤ))
    println("Distance = ", mean(distances))
    println("Temps d'exécution = ", mean(times))
end

evaluate(10, nobs=1000, niter=10000);