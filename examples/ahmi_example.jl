using BAT

#Model definition to generate samples from a n-dim gaussian shell
struct GaussianShellDensity<:AbstractDensity
    lambda::Vector{Float64}
    r::Float64
    sigma::Float64
    dimensions::Int64
end
BAT.nparams(model::GaussianShellDensity) = model.dimensions

#optional define exec_capabilities of our likelihood
BAT.exec_capabilities(::typeof(BAT.unsafe_density_logval), target::GaussianShellDensity, params::AbstractArray) = ExecCapabilities(0, true, 0, true)

#define likelihood for the Gaussian Shell
function BAT.unsafe_density_logval(target::GaussianShellDensity, params::AbstractArray{Float64, 1}, exec_context::ExecContext = ExecContext())
    diff::Float64 = 0
    for i in eachindex(params)
        diff += (target.lambda[i] - params[i]) * (target.lambda[i] - params[i])
    end
    diff = sqrt(diff)
    expo::Float64 = exp(-(diff - target.r) * (diff - target.r) / (2 * target.sigma^2))
    return log(1.0 / sqrt(2 * pi * target.sigma^2) * expo)
end

algorithm = MetropolisHastings()
#algorithm = MetropolisHastings(MHAccRejProbWeights{Float64}())

#define model and #dimensions
dim = 2
model = GaussianShellDensity(zeros(dim), 5.0, 2.0, dim)

#define boundaries
lo_bounds = [-30.0 for i = 1:dim]
hi_bounds = [ 30.0 for i = 1:dim]
bounds = HyperRectBounds(lo_bounds, hi_bounds, reflective_bounds)

chainspec = MCMCSpec(algorithm, model, bounds)
chains = 8
nsamples = 10^5

#define function to generate samples
sample() = rand(chainspec, nsamples, chains)


#Harmonic Mean Integration
#True integral value for 2D Gaussian Shell I = 31.4411
#True integral value for 10D Gaussian Shell I = 1.1065e9


#BAT.jl samples
bat_samples = sample()
data = HMIData(bat_samples)
hm_integrate!(data)

using Plots; pyplot()
plot(data, rscale = 0.25)
