# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type AbstractAdaptiveTransform end


struct CustomTransform{F} <: AbstractAdaptiveTransform 
    f::F
end

CustomTransform() = CustomTransform(identity)

struct TriangularAffineTransform <: AbstractAdaptiveTransform end

# TODO: MD, make typestable
function init_adaptive_transform(
    adaptive_transform::BAT.TriangularAffineTransform,
    target,
    context
)
    n = totalndof(varshape(target))

    M = _approx_cov(target, n)
    s = cholesky(M).L
    g = Mul(s)

    return g
end


struct DiagonalAffineTransform <: AbstractAdaptiveTransform end
