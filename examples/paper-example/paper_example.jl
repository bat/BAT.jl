using BAT
using DensityInterface
using Distributions
using Plots
using StatsBase
using Statistics
using SpecialFunctions
using ArraysOfArrays
using TypedTables
using CSV
import Cuba, AdvancedHMC, ForwardDiff
using AutoDiffOperators
#using AHMI

BAT.set_batcontext(ad = ADSelector(ForwardDiff))


function log_pdf_poisson(λ::T, k::U) where {T<:Real,U<:Real}
    R = float(promote_type(T,U))
    if λ >= 0 && k >= 0 && isinteger(k)
        a = iszero(k) ? R(k) : R(log(λ) * k)
        lg = logabsgamma(k + 1)[1]
        R(a - λ - lg)
    else
        R(-Inf)
    end
end

function bkg_signal_pdf(
    ν_B::Real,
    ν_S::Real,
    λ::Real,
    S_μ::Real,
    S_σ::Real,
    E::Real)

    res = (ν_B * pdf(Exponential(λ), E) + ν_S * pdf(Normal(S_μ, S_σ), E)) / (ν_B + ν_S)
    return res
end

function log_bkg_signal_pdf(
    ν_B::Real,
    ν_S::Real,
    λ::Real,
    S_μ::Real,
    S_σ::Real,
    E::Real)

    return log(bkg_signal_pdf(ν_B, ν_S, λ, S_μ, S_σ, E))
end

function μ_log_normal(
    γ::Real,
    σ::Real)

    return log(γ) - σ^2 / 2.0
end

function fit_function_sum_all(p, E)

    ν_B = summary_dataset_table.exposure .* p.B
    ν_S = summary_dataset_table.exposure .* summary_dataset_table.efficiency .* p.S

    return sum(summary_dataset_table.exposure .* bkg_signal_pdf.(ν_B, ν_S, p.λ, p.S_μ, p.S_σ, E)) / sum(summary_dataset_table.exposure)

end


# We can define a likelihood via logfuncdensity
function make_likelihood_bkg(summary_dataset_table, sample_table)
    logfuncdensity(function(p)
        ll_a = sum(log_pdf_poisson.(p.B .* summary_dataset_table.exposure, summary_dataset_table.event_counts))
        ll_b = sum(-log(p.λ) .- sample_table.E ./ p.λ)
        return ll_a + ll_b
    end)
end


# We can also define a likelihood as a struct that supports `densitykind` and `logdensityof`:
struct SignalBkgLikelihood{DS,ST}
    summary_dataset_table::DS
    sample_table::ST
end

@inline DensityInterface.DensityKind(::SignalBkgLikelihood) = IsDensity()

function DensityInterface.logdensityof(likelihood::SignalBkgLikelihood, p)
    summary_dataset_table =  likelihood.summary_dataset_table
    sample_table =  likelihood.sample_table

    ν_B = summary_dataset_table.exposure .* p.B
    ν_S = summary_dataset_table.exposure .* summary_dataset_table.efficiency .* p.S
    help_B = ν_B[sample_table.dataset]
    help_S = ν_S[sample_table.dataset]

    ll_a = sum(log_pdf_poisson.(ν_B .+ ν_S, summary_dataset_table.event_counts))
    ll_b = sum(log_bkg_signal_pdf.(help_B, help_S, p.λ, p.S_μ, p.S_σ, sample_table.E))

    ll_a + ll_b
end


N_datasets = 5
E_min = 0.0
E_max = 200.0
signal_events = 3
true_λ = 50.0

ΔE = E_max - E_min

summary_dataset_table = CSV.read("summary_dataset_table.csv", Table)
sample_table = CSV.read("sample_table.csv", Table)

function make_child_prior(N)
    v -> begin
        return distprod(B = fill(LogNormal(μ_log_normal(v.m_B, v.σ_B), v.σ_B), N))
    end
end

parent_prior_bkg = distprod(
    σ_B = Uniform(0.1, 1.0),
    m_B = Uniform(10^-10, 1e-1 * ΔE),
    λ = Uniform(10^-10, 100.0)
)

parent_prior_bkg_signal = distprod(
    S = Uniform(0.0, 10.0),
    S_μ = 100.0,
    S_σ = 2.0,
    σ_B = Uniform(0.1, 1.0),
    m_B = Uniform(10^-10, 1e-1 * ΔE),
    λ = Uniform(10^-10, 100.0)
)

prior_bkg =  distbind(make_child_prior(length(summary_dataset_table)), parent_prior_bkg, merge)

prior_bkg_signal = distbind(make_child_prior(length(summary_dataset_table)), parent_prior_bkg_signal, merge)

posterior_bkg = PosteriorMeasure(make_likelihood_bkg(summary_dataset_table, sample_table), prior_bkg)

posterior_bkg_signal = PosteriorMeasure(SignalBkgLikelihood(summary_dataset_table, sample_table), prior_bkg_signal)

nchains = 4
nsteps = 10^5

algorithm = MCMCSampling(proposal = HamiltonianMC(), nchains = nchains, nsteps = nsteps)

samples_bkg = bat_sample(posterior_bkg, algorithm).result
eval_bkg = EvaluatedMeasure(posterior_bkg, samples = samples_bkg)

@show evidence_bkg_bridge = bat_integrate(eval_bkg, BridgeSampling()).result
@show evidence_bkg_cuba = bat_integrate(eval_bkg, VEGASIntegration(maxevals = 10^6, rtol = 0.005)).result

samples_bkg_signal = bat_sample(posterior_bkg_signal, algorithm).result
eval_bkg_signal = EvaluatedMeasure(posterior_bkg_signal, samples = samples_bkg_signal)

@show evidence_bkg_signal_bridge = bat_integrate(eval_bkg_signal, BridgeSampling()).result
@show evidence_bkg_signal_cuba = bat_integrate(eval_bkg_signal, VEGASIntegration(maxevals = 10^6, rtol = 0.005)).result

#@show BF_exponential_bridge = evidence_bkg_signal_bridge / evidence_bkg_bridge
@show BF_exponential_cuba = evidence_bkg_signal_cuba / evidence_bkg_cuba

@show bkg_sig_marginal_modes = bat_marginalmode(samples_bkg_signal).result

p_1 = plot(size=(800,500), layout=(2,2), labelfontsize=12, tickfontsize=10, legendfontsize=7)
#upper left
p_1 = plot!(samples_bkg_signal, :S, subplot=1, label = "Posterior")
#p_1 = plot!(parent_prior_bkg_signal, :S, subplot=1, label = "Prior", linecolor = "blue")
#lower right
p_1 = plot!(samples_bkg_signal, :λ, subplot=4, label = "Posterior", legend=false)
#p_1 = plot!(parent_prior_bkg_signal, :λ, subplot=4, label = "Prior", linecolor = "blue")
#upper right
p_1 = plot!(samples_bkg_signal, (:S, :λ), subplot=2, st = :histogram, legend=false, colorbar=false)
#lower left
p_1 = plot!(samples_bkg_signal, (:S,:λ), subplot=3, legend=true)

savefig(p_1, "prior_posterior.pdf")
savefig(p_1, "prior_posterior.png")



p_2 = plot(size=(800,500), layout=(2,2), labelfontsize=12, tickfontsize=10, legendfontsize=7)
#upper left
p_2 = plot!(samples_bkg_signal, :m_B, subplot=1, label = "Posterior")
#p_2 = plot!(parent_prior_bkg_signal, :m_B, subplot=1, label = "Prior", linecolor = "blue")
#lower right
p_2 = plot!(samples_bkg_signal, :σ_B, subplot=4, label = "Posterior", legend=false)
#p_2 = plot!(parent_prior_bkg_signal, :σ_B, subplot=4, label = "Prior", linecolor = "blue")
#upper right
p_2 = plot!(samples_bkg_signal, (:m_B, :σ_B), subplot=2, st = :histogram, legend=false, colorbar=false)
#lower left
p_2 = plot!(samples_bkg_signal, (:m_B,:σ_B), subplot=3, legend=true)

savefig(p_2, "prior_posterior_hierarchical.pdf")
savefig(p_2, "prior_posterior_hierarchical.png")


p_hist = plot(size=(800,500), layout=(1,1), labelfontsize=12, tickfontsize=10, legendfontsize=7)
p_hist = histogram!(sample_table.E, bins = range(0.0, stop=maximum(sample_table.E)+20., length=100), title = "", xlabel = "Energy [keV]", ylabel = "Counts", label = "", box = :on, grid = :off)

savefig(p_hist, "total_hist.pdf")
savefig(p_hist, "total_hist.png")

p_fit_sum = plot(size=(800,500), layout=(1,1), labelfontsize=12, tickfontsize=10, legendfontsize=7)
p_fit_sum = plot!(range(0.0, (maximum(sample_table.E)+20), length=500), fit_function_sum_all, samples_bkg_signal, box = :on, grid = :off, xlabel = "Energy [keV]", ylabel = "Background distribution", legend = :topright)
p_fit_sum = histogram!(twinx(), xticks=([], []), sample_table.E, bins = range(0.0, stop=maximum(sample_table.E)+20., length=100), ylabel = "Counts", label = "Binned data", box = :on, grid = :off, fillalpha = 0.4, linealpha = 0.4, legend = :topleft)

savefig(p_fit_sum, "detector_sum.pdf")
savefig(p_fit_sum, "detector_sum.png")
