abstract type AdaptiveTransformSpec end


struct CustomTransform{F} <: AdaptiveTransformSpec 
    f::F
end

CustomTransform() = CustomTransform(identity)

function init_adaptive_transform(
    rng::AbstractRNG,
    adaptive_transform::CustomTransform,
    density
)
    return adaptive_transform
end



struct TriangularAffineTransform <: AdaptiveTransformSpec end

function init_adaptive_transform(
    rng::AbstractRNG,
    adaptive_transform::TriangularAffineTransform,
    density
)
    M = _approx_cov(density)
    s = cholesky(M).L
    g = Mul(s)

    return g
end



struct DiagonalAffineTransform <: AdaptiveTransformSpec end




