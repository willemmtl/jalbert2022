using Statistics

include("iGMRF.jl");
include("dataGen.jl");
include("amWG.jl");
include("malaWG.jl");
include("nutsWG.jl");

function evaluate(N; nobs::Integer, niter::Integer, m₁::Integer, m₂::Integer, warming_size::Integer, translation::Real=0.0)

    Random.seed!(400);

    m = m₁ * m₂;
    κᵤ = 100.0;
    F = iGMRF(m₁, m₂, κᵤ);
    grid_target = generateTargetGrid(F);
    grid_target[:, :, 1] = grid_target[:, :, 1] .+ translation;
    data = generateData(grid_target, nobs);

    h = 0.25;
    θ₀ = vcat([10], fill(10, m));
    F = iGMRF(m₁, m₂, 10);

    ESS = zeros(N);
    ESSperSecond = zeros(N);
    κ̂ᵤ = zeros(N);
    distances = zeros(N);
    times = zeros(N);

    for n=1:N
        startTime = time()
        κᵤ, μ = nutsWG(niter, θ₀, Y=data, F=F, nchains=1);
        endTime = time()
        times[n] = endTime - startTime

        # Warming time removed
        μSampling = μ[warming_size:end, :, :];
        # Change rate
        println("Taux d'acceptation de la simulation $n: ", round(mean(changerate(μSampling).value), digits=3))
        # Effective sample size
        ESS[n] = mean(summarystats(μSampling).value[:, 5])
        # ESS per second
        ESSperSecond[n] = ESS[n] / times[n]
        # Estimates
        μ̂ = mean(μSampling.value, dims=1);
        κ̂ᵤ[n] = mean(κᵤ[warming_size:end, 1]);

        distances[n] = norm(reshape(μ̂, m₁, m₂)' .- grid_target[:, :, 1], 2) / m
    end

    println("κ̂ᵤ = ", round(mean(κ̂ᵤ), digits=3))
    println("Distance = ", round(mean(distances), digits=6))
    println("Temps d'exécution = ", round(mean(times), digits=3))
    println("ESS = ", round(mean(ESS), digits=0), " +/- ", round(std(ESS), digits=0))
    println("ESS / s = ", round(mean(ESSperSecond), digits=0), " +/- ", round(std(ESSperSecond), digits=0))
end

nobs = 100;
niter = 100;

evaluate(10, nobs=nobs, niter=niter, m₁=10, m₂=10, warming_size=Int(0.2 * niter), translation=10);