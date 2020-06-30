# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    ModeAsDefined <: AbstractModeEstimator

Constructors:

    ModeAsDefined()

Get the mode as defined by the density, resp. the underlying distribution
(if available), via `StatsBase.mode`.
"""
struct ModeAsDefined <: AbstractModeEstimator end
export ModeAsDefined


function bat_findmode_impl(posterior::AnyPosterior, algorithm::ModeAsDefined)
    (result = StatsBase.mode(posterior),)
end

function bat_findmode_impl(posterior::DistributionDensity, algorithm::ModeAsDefined)
    (result = StatsBase.mode(posterior.dist),)
end



"""
    MaxDensitySampleSearch <: AbstractModeEstimator

Constructors:

    MaxDensitySampleSearch()

Estimate the mode as the variate with the highest posterior density value
within a given set of samples.
"""
struct MaxDensitySampleSearch <: AbstractModeEstimator end
export MaxDensitySampleSearch


function bat_findmode_impl(posterior::DensitySampleVector, algorithm::MaxDensitySampleSearch)
    v, i = _get_mode(posterior)
    (result = v, mode_idx = i)
end



"""
    MaxDensityNelderMead <: AbstractModeEstimator

Constructors:

    MaxDensityNelderMead()

Estimate the mode of the posterior using Nelder-Mead optimization (via
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)).
"""
struct MaxDensityNelderMead <: AbstractModeEstimator end
export MaxDensityNelderMead

function bat_findmode_impl(posterior::AnyPosterior, algorithm::MaxDensityNelderMead; initial_mode = missing)
    shape = varshape(posterior)
    x = _get_initial_mode(posterior, initial_mode)
    conv_posterior = convert(AbstractDensity, posterior)
    r = Optim.maximize(p -> logvalof(conv_posterior, p), x, Optim.NelderMead())
    (result = shape(Optim.minimizer(r.res)), info = r)
end


_get_initial_mode(posterior::AnyPosterior, ::Missing) =
    _get_initial_mode(posterior, rand(sampler(getprior(posterior))))

_get_initial_mode(posterior::AnyPosterior, samples::DensitySampleVector) =
    _get_initial_mode(posterior, unshaped(bat_findmode(samples).result))

_get_initial_mode(posterior::AnyPosterior, x::AbstractArray{<:Real}) = Array(x)
_get_initial_mode(posterior::AnyPosterior, x::Array{<:Real}) = x

function _get_initial_mode(posterior::AnyPosterior, x)
    shape = varshape(posterior)
    x_unshaped = Vector{<:Real}(undef, shape)
    shape(x_unshaped)[] = stripscalar(x)
    x_unshaped
end



"""
    MaxDensityLBFGS <: AbstractModeEstimator

Constructors:

    MaxDensityLBFGS()

Estimate the mode of the posterior using LBFGS optimization (via
[Optim.jl](https://github.com/JuliaNLSolvers/Optim.jl)). The gradient
of the posterior is computer by forward-mode auto-differentiation (via
[ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl)).
"""
struct MaxDensityLBFGS <: AbstractModeEstimator end
export MaxDensityLBFGS


function bat_findmode_impl(posterior::AnyPosterior, algorithm::MaxDensityLBFGS; initial_mode = missing)
    shape = varshape(posterior)
    x = _get_initial_mode(posterior, initial_mode)
    conv_posterior = convert(AbstractDensity, posterior)
    r = Optim.maximize(p -> logvalof(conv_posterior, p), x, Optim.LBFGS(); autodiff = :forward)
    (result = shape(Optim.minimizer(r.res)), info = r)
end



function bat_marginalmode_impl(samples::DensitySampleVector; nbins::Union{Integer, Symbol} = 200)

    shape = varshape(samples)
    flat_samples = flatview(unshaped.(samples.v))
    n_params = size(flat_samples)[1]
    nt_samples = ntuple(i -> flat_samples[i,:], n_params)
    marginalmode_params = Vector{Float64}()

    for param in Base.OneTo(n_params)
        if typeof(nbins) == Symbol
            number_of_bins = Plots._auto_binning_nbins(nt_samples, param, mode=nbins)
        else
            number_of_bins = nbins
        end

        marginalmode_param = find_localmodes(bat_marginalize(samples, param, nbins=number_of_bins).result)

        if length(marginalmode_param[1]) > 1
            @warn "More than one bin with the same weight is found. Returned the first one"
        end
        push!(marginalmode_params, marginalmode_param[1][1])
    end
    (result = shape(marginalmode_params),)
end
