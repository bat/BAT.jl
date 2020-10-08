using BAT, ValueShapes

#Model definition to generate samples from a n-dim gaussian shell
struct GaussianShellDensity<:AbstractDensity
    lambda::Vector{Float64}
    r::Float64
    sigma::Float64
    dimensions::Int64
end
ValueShapes.totalndof(model::GaussianShellDensity) = model.dimensions

#define likelihood for the Gaussian Shell
function BAT.eval_logval_unchecked(target::GaussianShellDensity, v::AbstractArray{Float64, 1})
    diff::Float64 = 0
    for i in eachindex(v)
        diff += (target.lambda[i] - v[i]) * (target.lambda[i] - v[i])
    end
    diff = sqrt(diff)
    expo::Float64 = exp(-(diff - target.r) * (diff - target.r) / (2 * target.sigma^2))
    return log(1.0 / sqrt(2 * pi * target.sigma^2) * expo)
end

algorithm = MetropolisHastings()
#algorithm = MetropolisHastings(ARPWeighting{Float64}())

#define model and #dimensions
dim = 2
model = GaussianShellDensity(zeros(dim), 5.0, 2.0, dim)

#define boundaries
lo_bounds = [-30.0 for i = 1:dim]
hi_bounds = [ 30.0 for i = 1:dim]
bounds = BAT.HyperRectBounds(lo_bounds, hi_bounds, BAT.reflective_bounds)



#Harmonic Mean Integration
#True integral value for 2D Gaussian Shell I = 31.4411
#True integral value for 10D Gaussian Shell I = 1.1065e9


#BAT.jl samples
bat_samples = bat_sample(PosteriorDensity(model, bounds), (10^5, 8), algorithm).result
data = BAT.HMIData(bat_samples)
BAT.hm_integrate!(data)

using Plots; pyplot()
plot(data, rscale = 0.25)
