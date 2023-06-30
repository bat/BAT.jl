# TODO AC: File not included as it would overwrite BAT.jl functions


function _cov_with_fallback(d)
    rng = bat_determ_rng()
    smplr = bat_sampler(d)
    T = float(eltype(rand(rng, smplr)))
    n = totalndof(varshape(d))
    C = fill(T(NaN), n, n)
    try
        C[:] = cov(d)
    catch err
        if err isa MethodError
            C[:] = cov(nestedview(rand(rng, smplr, 10^5)))
        else
            throw(err)
        end
    end
    return C
end

_approx_cov(target::Distribution) = _cov_with_fallback(target)
_approx_cov(target::DistLikeMeasure) = _cov_with_fallback(target)
_approx_cov(target::AbstractPosteriorMeasure) = _approx_cov(getprior(target))
_approx_cov(target::BAT.Transformed{<:Any,<:BAT.DistributionTransform}) =
    BAT._approx_cov(target.trafo.target_dist)
_approx_cov(target::Renormalized) = _approx_cov(parent(target))
_approx_cov(target::WithDiff) = _approx_cov(parent(target))

