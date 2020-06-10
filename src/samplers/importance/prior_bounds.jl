function get_bounds(prior::NamedTupleDist)
    bounds = [get_bounds(d) for d in values(prior)]
    return vcat(bounds...)
end


function get_bounds(d::Distribution)
    lo, hi = minimum(d), maximum(d)

    if isinf(lo)
        lo = typeof(lo)(quantile(d, 0.00001))
    end

    if isinf(hi)
        hi = typeof(hi)(quantile(d, 0.99999))
    end

    return lo, hi
end

function get_bounds(d::Product)
    bounds = get_bounds.(d.v)
    return vcat(bounds...)
end

function get_prior_bounds(posterior::AnyPosterior)
    prior = getprior(posterior).dist
    return get_bounds(prior)
end
