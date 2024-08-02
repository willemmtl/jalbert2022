using Statistics

include("iGMRF.jl");
include("dataGen.jl");
include("malaWG.jl");

function evaluate(N; nobs::Integer, niter::Integer, m₁::Integer, m₂::Integer, warming_size::Integer)

    Random.seed!(400);

    m = m₁ * m₂;
    κᵤ = 100.0;
    F = iGMRF(m₁, m₂, κᵤ);
    grid_target = generateTargetGrid(F);
    data = generateData(grid_target, nobs);

    h = 0.0005;
    θ₀ = vcat([10], zeros(m));
    F = iGMRF(m₁, m₂, 10);

    κ̂ᵤ = zeros(N);
    distances = zeros(N);
    times = zeros(N);

    for n=1:N
        startTime = time()
        θ = malaWG(niter, h, θ₀, Y=data, F=F);
        endTime = time()
        times[n] = endTime - startTime

        μ̂ = mean(θ[2:end, warming_size:end], dims=2);
        κ̂ᵤ[n] = mean(θ[1, warming_size:end]);

        distances[n] = norm(reshape(μ̂, m₁, m₂)' .- grid_target[:, :, 1], 2) / m
    end

    println("κ̂ᵤ = ", mean(κ̂ᵤ))
    println("Distance = ", mean(distances))
    println("Temps d'exécution = ", mean(times))
end

evaluate(10, nobs=1000, niter=1000, m₁=6, m₂=6, warming_size=200);