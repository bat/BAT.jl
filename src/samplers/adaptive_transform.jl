# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type AbstractAdaptiveTransform end


struct CustomTransform{F} <: AbstractAdaptiveTransform 
    f::F
end

CustomTransform() = CustomTransform(identity)

init_adaptive_transform(at::CustomTransform, ::AbstractMeasure, ::BATContext) = at.f



struct NoAdaptiveTransform <: AbstractAdaptiveTransform end

init_adaptive_transform(::NoAdaptiveTransform, ::AbstractMeasure, ::BATContext) = identity

struct AdaptiveTransformChain{AT<:AbstractAdaptiveTransform} <: AbstractAdaptiveTransform
    f::Tuple{Vararg{AT}}
end
export AdaptiveTransformChain

function init_adaptive_transform(
    adaptive_transform::AdaptiveTransformChain, 
    target::AbstractMeasure, 
    context::BATContext
)
    initialized_trafos = Vector{Function}()

    for trafo in adaptive_transform.f
        trafo_init = init_adaptive_transform(trafo, target, context)
        push!(initialized_trafos, trafo_init)
    end

    return fchain(initialized_trafos)
end


function _iterate_trafo_with_interm((f_1, itr_state), fs, current_0, proposed_0)
    current = (x = transform_samples(f_1, current_0.z), z = current_0.z)
    proposed = (x = transform_samples(f_1, proposed_0.z), z = proposed_0.z)

    intermediate_results = FunctionChains._similar_empty(fs, typeof((current, proposed)))
    FunctionChains._sizehint!(intermediate_results, Base.IteratorSize(fs), fs)
    intermediate_results = FunctionChains._push!!(intermediate_results, (current, proposed))

    next = iterate(fs, itr_state)
    while !isnothing(next)
        f_i, itr_state = next

        current = (x = transform_samples(f_i, current.x), z = current.x)
        proposed = (x = transform_samples(f_i, proposed.x), z = proposed.x)
       
        intermediate_results = FunctionChains._push!!(intermediate_results, (current, proposed))

        next = iterate(fs, itr_state)
    end

    return intermediate_results
end

function trafo_samples_with_interm_results(fc::FunctionChain, current, proposed)
    fs = fchainfs(fc)
    return _iterate_trafo_with_interm(iterate(fs), fs, current, proposed)
end



function _iterate_trafo_with_interm((f_1, itr_state), fs, samples::AbstractVector{<:DensitySampleVector})
    intermediate_results = FunctionChains._similar_empty(fs, typeof(samples))
    FunctionChains._sizehint!(intermediate_results, Base.IteratorSize(fs), fs)

    intermediate_results = FunctionChains._push!!(intermediate_results, samples) 

    trafo_samples = transform_samples.(f_1, samples)
    next = iterate(fs, itr_state)
    while !isnothing(next) 
        intermediate_results = FunctionChains._push!!(intermediate_results, trafo_samples)
        f_i, itr_state = next

        # TODO, MD: Unnecessarily applies the trafo in the last iteration. Fix.
        trafo_samples = transform_samples.(f_i, trafo_samples)
        next = iterate(fs, itr_state)
    end

    return intermediate_results
end

function trafo_samples_with_interm_results(fc::FunctionChain, samples::AbstractVector{<:DensitySampleVector})
    fs = fchainfs(fc)
    return _iterate_trafo_with_interm(iterate(fs), fs, samples)
end


struct TriangularAffineTransform <: AbstractAdaptiveTransform end

# TODO: MD, make typestable
function init_adaptive_transform(adaptive_transform::TriangularAffineTransform, target::AbstractMeasure, ::BATContext)
    n = totalndof(varshape(target))

    M = _approx_cov(target, n)
    b = _approx_mean(target, n)
    s = cholesky(M).L
    g = MulAdd(s, b)

    return g
end


# TODO: Implement DiagonalAffineTransform
struct DiagonalAffineTransform <: AbstractAdaptiveTransform end
