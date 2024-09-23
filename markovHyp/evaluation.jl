using Statistics

include("../iGMRF.jl");
include("../dataGen.jl");
include("markovHyp.jl");

function evaluate(N; nobs::Integer, niter::Integer, m₁::Integer, m₂::Integer, δ²::Real, warming_size::Integer, translation::Real=0.0)

    Random.seed!(400);

    m = m₁ * m₂;
    κᵤ = 100.0;
    F = iGMRF(m₁, m₂, κᵤ);
    grid_target = generateTargetGrid(F);
    grid_target[:, :, 1] = grid_target[:, :, 1] .+ translation;
    data = generateData(grid_target, nobs);

    κᵤ₀ = 10;
    μ₀ = fill(10, m);
    F = iGMRF(m₁, m₂, 10);

    μESS = zeros(N);
    μESSperSecond = zeros(N);
    κ̂ᵤ = zeros(N);
    distances = zeros(N);
    times = zeros(N);

    for n=1:N
        startTime = time()
        θ = gibbs(niter, data, m₁=m₁, m₂=m₂, δ²=δ², κᵤ₀=κᵤ₀, μ₀=μ₀);
        endTime = time()
        times[n] = endTime - startTime

        # Warming time removed
        θsampling = θ[warming_size:end, :, :];
        μSampling = θ[warming_size:end, 2:end, :];
        # Change rate
        # Performed only on μ because κᵤ's change rate is 1.0
        println("Taux d'acceptation moyen de la simulation $n: ", round(mean(changerate(μSampling).value), digits=3))
        # Effective sample size
        μESS[n] = mean(summarystats(μSampling).value[:, 5])
        # ESS per second
        μESSperSecond[n] = μESS[n] / times[n]
        # Estimates
        κ̂ᵤ[n] = mean(θsampling.value[:, 1, 1]);
        
        μ̂ = mean(θsampling.value[:, 2:end, 1], dims=1);
        distances[n] = norm(reshape(μ̂, m₁, m₂)' .- grid_target[:, :, 1], 2) / m
    end

    println("κ̂ᵤ = ", round(mean(κ̂ᵤ), digits=3))
    println("Distance = ", round(mean(distances), digits=6))
    println("Temps d'exécution = ", round(mean(times), digits=3))
    println("μESS = ", round(mean(μESS), digits=0), " +/- ", round(std(μESS), digits=0))
    println("μESS / s = ", round(mean(μESSperSecond), digits=0), " +/- ", round(std(μESSperSecond), digits=0))
end

δ² = 0.22;
nobs = 100;
niter = 10000;

evaluate(10, nobs=nobs, niter=niter, m₁=10, m₂=10, δ²=δ², warming_size=Int(0.2 * niter), translation=10);