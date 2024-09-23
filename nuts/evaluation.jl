using Statistics

include("../iGMRF.jl");
include("../dataGen.jl");
include("NUTS.jl");

function evaluate(N; nobs::Integer, niter::Integer, m₁::Integer, m₂::Integer, warming_size::Integer, translation::Real=0.0)

    Random.seed!(400);

    m = m₁ * m₂;
    κᵤ = 100.0;
    F = iGMRF(m₁, m₂, κᵤ);
    grid_target = generateTargetGrid(F);
    grid_target[:, :, 1] = grid_target[:, :, 1] .+ translation;
    data = generateData(grid_target, nobs);

    θ₀ = vcat([10], fill(10, m));
    F = iGMRF(m₁, m₂, 10);

    κᵤESS = zeros(N);
    μESS = zeros(N);
    κᵤESSperSecond = zeros(N);
    μESSperSecond = zeros(N);
    κ̂ᵤ = zeros(N);
    distances = zeros(N);
    times = zeros(N);

    for n=1:N
        startTime = time()
        θ = nuts(niter, θ₀, Y=data, F=F);
        endTime = time()
        times[n] = endTime - startTime

        # Warming time removed
        θsampling = θ[warming_size:end, :, :];
        κᵤSampling = θ[warming_size:end, 1, :];
        μSampling = θ[warming_size:end, 2:end, :];
        # Change rate
        println("Taux d'acceptation de la simulation $n: ", round(mean(changerate(θsampling).value), digits=3))
        # Effective sample size
        κᵤESS[n] = mean(summarystats(κᵤSampling).value[:, 5])
        μESS[n] = mean(summarystats(μSampling).value[:, 5])
        # ESS per second
        κᵤESSperSecond[n] = κᵤESS[n] / times[n]
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
    println("κᵤESS = ", round(mean(κᵤESS), digits=0), " +/- ", round(std(κᵤESS), digits=0))
    println("μESS / s = ", round(mean(μESSperSecond), digits=0), " +/- ", round(std(μESSperSecond), digits=0))
    println("κᵤESS / s = ", round(mean(κᵤESSperSecond), digits=0), " +/- ", round(std(κᵤESSperSecond), digits=0))
end

nobs = 100;
niter = 1000;

evaluate(10, nobs=nobs, niter=niter, m₁=10, m₂=10, warming_size=Int(0.2 * niter), translation=10);