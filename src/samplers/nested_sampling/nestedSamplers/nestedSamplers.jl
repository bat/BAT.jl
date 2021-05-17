using NestedSamplers
using MCMCChains: Chains

include("ns_bounds.jl")
include("ns_proposals.jl")


@with_kw struct UsingNestedSamplers <: AbstractSamplingAlgorithm
    
    num_live_points::Int = 1000
    
    bound::NSBound = MultiEllipsoidBound()
    
    proposal::NSProposal = AutoProposal() 

    enlarge::Float64 = 1.25

    # Not sure about what this does yet
    # update_interval =
    
    min_ncall::Int64 = 2*num_live_points
    
    min_eff::Float64 = 0.10

    # The following four are the possible convergence criteria
    dlogz::Float64 = 0.1
    max_iters = Inf
    max_ncalls = Inf
    maxlogl = Inf
end
export UsingNestedSamplers


############################################################################################################
# Here are transformation functions to convert datatypes
############################################################################################################
function const2normal(c::ValueShapes.ConstValueDist)
    return Distributions.Normal(c.value, 0)
end

function const2normal(c)
    return c
end

function prior2array(posterior)
    p = collect(values(BAT.getprior(posterior).dist._internal_distributions))
    return const2normal.(p) # durch den Punkt wird es für jedes Element des Arrays durchgeführt
end

function chain2batsamples(chain::Chains, shape)
    weights = chain.value.data[:, end] # Das letzte Vektorelement sind die weights
    logvals = zeros(length(weights))

    samples = [chain.value.data[i, 1:end-1] for i in 1:length(chain.value.data[:, 1])] # Die vorherigen Elemente sind die Samples
    return shape.(BAT.DensitySampleVector(samples, logvals, weight = weights)) # = batsamples
end

function batPosterior2nestedModel(posterior; num_live_points, bound::NSBound, proposal::NSProposal, enlarge, min_ncall, min_eff)
    function nestedLikelihood(x) # evtl posterior mit let
        ks = keys(BAT.varshape(posterior))
        nx = (;zip(ks, x)...)
        return BAT.eval_logval_unchecked(BAT.getlikelihood(posterior),nx) # Wert aus Likelihood extrahieren
    end

    priors = prior2array(posterior) # NS erwartet ein Array als Prior, dieses können wir aus dem BAT-Posterior extrahieren

    model = NestedModel(nestedLikelihood, priors);

    bounding = NSbounding(bound)
    prop = NSprop(proposal)
    sampler = Nested(BAT.getprior(posterior).dist._internal_shape._flatdof, num_live_points; # Die Anzahl der Freiheitsgerade wird aus dem Posterior extrahiert
                    bounding, # Art der "Grenze" des Volumens
                    prop, # Algoritmus zum finden neuer Punkte
                    enlarge, # Linearer Faktor, welcher bounds vergrößert
                    #update_interval, # ?
                    min_ncall, # mindestanzahl an Iterationen (vor der ersten Grenze?)
                    min_eff # maximale Effizenz (vor der ersten Grenze?)
                    ) 

    return model, sampler
end

############################################################################################################
# Here is the bat_sample implementation for the NestedSamplers package
############################################################################################################
function bat_sample_impl(rng::AbstractRNG, target::AnyDensityLike, algorithm::UsingNestedSamplers)
    posterior = target
    model, sampler = batPosterior2nestedModel(posterior; 
        algorithm.num_live_points, algorithm.bound, algorithm.proposal,
        algorithm.enlarge, algorithm.min_ncall, algorithm.min_eff
    )
    
    chain, state = sample(model, sampler; 
        dlogz = algorithm.dlogz, maxiter = algorithm.max_iters,
        maxcall = algorithm.max_ncalls, maxlogl = algorithm.maxlogl, chain_type=Chains
    )
    
    res = chain2batsamples(chain, BAT.varshape(BAT.getprior(posterior)));
    return ( ########### Here is more to do i think
        result = res,
    )

end

# For comparison here is a copy of the ultranestfunction
# function bat_sample_impl(
#     rng::AbstractRNG,
#     target::AnyDensityLike,
#     algorithm::ReactiveNestedSampling
# )
#     density_notrafo = convert(AbstractDensity, target)
#     shaped_density, trafo = bat_transform(algorithm.trafo, density_notrafo)
#     vs = varshape(shaped_density)
#     density = unshaped(shaped_density)
# 
#     bounds = var_bounds(density)
#     if !(all(isequal(0), bounds.vol.lo) && all(isequal(1), bounds.vol.hi))
#         throw(ArgumentError("ReactiveNestedSampling only supports (transformed) densities defined on the unit hypercube"))
#     end
# 
#     function vec_ultranest_logpstr(V_rowwise::AbstractMatrix{<:Real})
#         map(logdensityof(density), nestedview(copy(V_rowwise')))
#     end
# 
#     ndims = totalndof(vs)
#     paramnames = all_active_names(varshape(density_notrafo))
# 
#     smplr = UltraNest.ultranest.ReactiveNestedSampler(
#         paramnames, vec_ultranest_logpstr, vectorized = true,
#         num_test_samples = algorithm.num_test_samples,
#         draw_multiple = algorithm.draw_multiple,
#         num_bootstraps = algorithm.num_bootstraps,
#         ndraw_min = algorithm.ndraw_min,
#         ndraw_max = algorithm.ndraw_max
#     )
# 
#     unest_result = smplr.run(
#         log_interval = algorithm.log_interval < 0 ? nothing : algorithm.log_interval,
#         show_status = algorithm.show_status,
#         #viz_callback = algorithm.# viz_callback,
#         dlogz = algorithm.dlogz,
#         dKL = algorithm.dKL,
#         frac_remain = algorithm.frac_remain,
#         Lepsilon = algorithm.Lepsilon,
#         min_ess = algorithm.min_ess,
#         max_iters = algorithm.max_iters < 0 ? nothing : algorithm.max_iters,
#         max_ncalls = algorithm.max_ncalls < 0 ? nothing : algorithm.max_ncalls,
#         max_num_improvement_loops = algorithm.max_num_improvement_loops,
#         min_num_live_points = algorithm.min_num_live_points,
#         cluster_num_live_points = algorithm.cluster_num_live_points,
#         insertion_test_window = algorithm.insertion_test_window,
#         insertion_test_zscore_threshold = algorithm.insertion_test_zscore_threshold
#     )
#     
#     r = convert(Dict{String, Any}, unest_result)
# 
#     unest_wsamples = convert(Dict{String, Any}, r["weighted_samples"])
#     v_trafo_us = nestedview(convert(Matrix{Float64}, unest_wsamples["points"]'))
#     logvals_trafo = convert(Vector{Float64}, unest_wsamples["logl"])
#     weight = convert(Vector{Float64}, unest_wsamples["weights"])
#     samples_trafo = DensitySampleVector(vs.(v_trafo_us), logvals_trafo, weight = weight)
#     samples_notrafo = inv(trafo).(samples_trafo)
# 
#     uwv_trafo_us = nestedview(convert(Matrix{Float64}, r["samples"]'))
#     uwlogvals_trafo = map(logdensityof(density), uwv_trafo_us)
#     uwsamples_trafo = DensitySampleVector(vs.(uwv_trafo_us), uwlogvals_trafo)
#     uwsamples_notrafo = inv(trafo).(uwsamples_trafo)
# 
#     logz = convert(BigFloat, r["logz"])::BigFloat
#     logzerr = convert(BigFloat, r["logzerr"])::BigFloat
#     logintegral = Measurements.measurement(logz, logzerr)
# 
#     ess = convert(Float64, r["ess"])
# 
#     return (
#         result = samples_notrafo, result_trafo = samples_trafo, trafo = trafo,
#         uwresult = uwsamples_notrafo, uwresult_trafo = uwsamples_trafo,
#         logintegral = logintegral, ess = ess,
#         info = r
#     )
# end