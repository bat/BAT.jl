abstract type AdaptiveTransformSpec end


struct CustomTransform{F} <: AdaptiveTransformSpec 
    f::F
end

CustomTransform() = CustomTransform(identity)

function init_adaptive_transform(
    adaptive_transform::CustomTransform,
    density,
    context
)
    return adaptive_transform.f
end



struct TriangularAffineTransform <: AdaptiveTransformSpec end

function init_adaptive_transform(
    adaptive_transform::TriangularAffineTransform,
    density,
    context
)
    M = _approx_cov(density)
    s = cholesky(M).L
    g = Mul(s)

    return g
end



struct DiagonalAffineTransform <: AdaptiveTransformSpec end




