# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Included via @require: "ultranest.jl"

using NestedSamplers
using MCMCChains: Chains

include("nestedSamplers/nestedSamplers.jl")

########################################################
# Umwandlungsfunktionen
########################################################
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
    return bat_samples = shape.(BAT.DensitySampleVector(samples, logvals, weight = weights))
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

########################################################
# Nested Samplers Code
########################################################
# function bat_sample(posterior; num_live_points::Int64=1000, bound::NSBound=EllipsoidBound(), proposal::NSProposal=Uniformly()) # später optionale Parameter, welche den genauen algorithmus beeinflussen
function test_bat_sample(posterior; num_live_points::Int64=1000, bound::NSBound=MultiEllipsoidBound(), proposal::NSProposal=Uniformly(),
                        enlarge::Float64=1.25, min_ncall::Int64=2*num_live_points, min_eff::Float64=0.10,
                        dlogz::Float64=0.1, maxiter=Inf, maxcall=Inf, maxlogl=Inf) # This are the convergence criteria
    model, sampler = batPosterior2nestedModel(posterior; num_live_points, bound, proposal, enlarge, min_ncall, min_eff)

    chain, state = sample(model, sampler; dlogz, maxiter, maxcall, maxlogl, chain_type=Chains)
    
    return chain2batsamples(chain, BAT.varshape(BAT.getprior(posterior)));
    #return (result= ..., )
end
