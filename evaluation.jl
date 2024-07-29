using Statistics

include("GMRF.jl");
include("grid.jl");
include("gibbs.jl");

function evaluate(N; nobs::Integer, niter::Integer, m₁::Integer, m₂::Integer, warming_size::Integer)

    Random.seed!(400);

    m = m₁ * m₂;
    κᵤ = 100.0;
    F = iGMRF(m₁, m₂, κᵤ);
    grid_target = generateTargetGridV1(F);
    data = generateData(grid_target, nobs);

    δ² = 0.07
    κᵤ₀ = 10
    μ₀ = zeros(m)
    W = buildStructureMatrix(m₁, m₂);

    κ̂ᵤ = zeros(N);
    distances = zeros(N);
    times = zeros(N);

    for n=1:N
        startTime = time()
        κᵤ, μ = gibbs(niter, data, δ²=δ², κᵤ₀=κᵤ₀, μ₀=μ₀, W=W);
        endTime = time()
        times[n] = endTime - startTime

        μ̂ = mean(μ[:, warming_size:end], dims=2);
        κ̂ᵤ[n] = mean(κᵤ[warming_size:end]);

        distances[n] = norm(reshape(μ̂, m₁, m₂) .- grid_target[:, :, 1], 2) / m
    end

    println("κ̂ᵤ = ", mean(κ̂ᵤ))
    println("Distance = ", mean(distances))
    println("Temps d'exécution = ", mean(times))
end

evaluate(10, nobs=1000, niter=1000, m₁=6, m₂=6, warming_size=200);