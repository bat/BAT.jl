using BAT
using ValueShapes
using Distributions
using Plots
using StatsBase
using Statistics
using SpecialFunctions
using ArraysOfArrays
using TypedTables
using CSV


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



function make_likelihood_bkg(summary_dataset_table, sample_table)
    p -> begin
        ll_a = sum(log_pdf_poisson.(p.B .* summary_dataset_table.exposure, summary_dataset_table.event_counts))
        ll_b = sum(-log(p.λ) .- sample_table.E ./ p.λ)
        return LogDVal(ll_a + ll_b)
    end
end

function make_likelihood_bkg_signal(summary_dataset_table, sample_table)
    p -> begin
        ν_B = summary_dataset_table.exposure .* p.B
        ν_S = summary_dataset_table.exposure .* summary_dataset_table.efficiency .* p.S
        help_B = ν_B[sample_table.dataset]
        help_S = ν_S[sample_table.dataset]

        ll_a = sum(log_pdf_poisson.(ν_B .+ ν_S, summary_dataset_table.event_counts))
        ll_b = sum(log_bkg_signal_pdf.(help_B, help_S, p.λ, p.S_μ, p.S_σ, sample_table.E))

        return LogDVal(ll_a + ll_b)
    end
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
        return NamedTupleDist(B = fill(LogNormal(μ_log_normal(v.mean_B, v.σ_B), v.σ_B), N))
    end
end

parent_prior_bkg = NamedTupleDist(
    σ_B = Uniform(0.1, 1.0),
    mean_B = Uniform(10^-8, 1e-1 * ΔE),
    λ = Uniform(10^-7, 100.0)
)

parent_prior_bkg_signal = NamedTupleDist(
    S = Uniform(0.0, 10.0),
    S_μ = 100.0,
    S_σ = 2.0,
    σ_B = Uniform(0.1, 1.0),
    mean_B = Uniform(10^-8, 1e-1 * ΔE),
    λ = Uniform(10^-7, 100.0)
)

prior_bkg =  HierarchicalDistribution(make_child_prior(length(summary_dataset_table)), parent_prior_bkg)

prior_bkg_signal = HierarchicalDistribution(make_child_prior(length(summary_dataset_table)), parent_prior_bkg_signal)

posterior_bkg = PosteriorDensity(make_likelihood_bkg(summary_dataset_table, sample_table), prior_bkg)

posterior_bkg_signal = PosteriorDensity(make_likelihood_bkg_signal(summary_dataset_table, sample_table), prior_bkg_signal)

nchains = 4
nsteps = 10^6

algorithm = MCMCSampling(mcalg = MetropolisHastings(), nchains = nchains, nsteps = nsteps)

samples_bkg = bat_sample(posterior_bkg, algorithm).result

@show evidence_bkg = bat_integrate(samples_bkg).result

samples_bkg_signal = bat_sample(posterior_bkg_signal, algorithm).result

@show evidence_bkg_signal = bat_integrate(samples_bkg_signal).result

@show BF_exponential = evidence_bkg_signal / evidence_bkg

@show bkg_sig_marginal_modes = bat_marginalmode(samples_bkg_signal).result

pyplot(size=(800,500), layout=(2,2), legendfontsize=7)
#upper left
p_1 = plot(samples_bkg_signal, :S, subplot=1, label = "Posterior")
p_1 = plot!(parent_prior_bkg_signal, :S, subplot=1, label = "Prior", linecolor = "blue")
#lower right
p_1 = plot!(samples_bkg_signal, :λ, subplot=4, label = "Posterior", legend=false)
p_1 = plot!(parent_prior_bkg_signal, :λ, subplot=4, label = "Prior", linecolor = "blue")
#upper right
p_1 = plot!(samples_bkg_signal, (:S, :λ), subplot=2, st = :histogram, legend=false, colorbar=false)
#lower left
p_1 = plot!(samples_bkg_signal, (:S,:λ), subplot=3, legend=true)

savefig(p_1, "prior_posterior.pdf")
savefig(p_1, "prior_posterior.png")



pyplot(size=(800,500), layout=(2,2), legendfontsize=7)
#upper left
p_2 = plot(samples_bkg_signal, :mean_B, subplot=1, label = "Posterior")
p_2 = plot!(parent_prior_bkg_signal, :mean_B, subplot=1, label = "Prior", linecolor = "blue")
#lower right
p_2 = plot!(samples_bkg_signal, :σ_B, subplot=4, label = "Posterior", legend=false)
p_2 = plot!(parent_prior_bkg_signal, :σ_B, subplot=4, label = "Prior", linecolor = "blue")
#upper right
p_2 = plot!(samples_bkg_signal, (:mean_B, :σ_B), subplot=2, st = :histogram, legend=false, colorbar=false)
#lower left
p_2 = plot!(samples_bkg_signal, (:mean_B,:σ_B), subplot=3, legend=true)

savefig(p_2, "prior_posterior_hierarchical.pdf")
savefig(p_2, "prior_posterior_hierarchical.png")


pyplot(size=(800,500), layout=(1,1), labelfontsize=20, tickfontsize=17, legendfontsize=20)
p_hist = histogram(sample_table.E, bins = range(0.0, stop=maximum(sample_table.E)+20., length=100), title = "", xlabel = "Energy [keV]", ylabel = "Counts", label = "", box = :on, grid = :off)
savefig(p_hist, "total_hist.pdf")
savefig(p_hist, "total_hist.png")

pyplot(size=(800,500), layout=(1,1), labelfontsize=20, tickfontsize=17, legendfontsize=20)
p_fit_sum = plot(range(0.0, (maximum(sample_table.E)+20), length=500), fit_function_sum_all, samples_bkg_signal, box = :on, grid = :off, xlabel = "", ylabel = "Background distribution", legend = :topright)
p_fit_sum = histogram!(twinx(), sample_table.E, bins = range(0.0, stop=maximum(sample_table.E)+20., length=100), xlabel = "Energy [keV]", ylabel = "Counts", label = "Binned data", box = :on, grid = :off, fillalpha = 0.4, linealpha = 0.4, legend = :topleft)

savefig(p_fit_sum, "detector_sum.pdf")
savefig(p_fit_sum, "detector_sum.png")
