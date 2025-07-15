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
