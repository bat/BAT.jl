# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    maximize_density(f_logdensity, x_init::AbstractVector{<:Real}, algorithm, context::BATContext)

*BAT-internal, not part of stable public API.*

Maximize the log-density function `f_logdensity`, starting at `x_init`,
using optimization `algorithm`.

The user is responsible for shaping f_logdensity in a way that works well
with the algorithm (typically by transforming it to an unbounded space),
`maximize_density` does not apply automatic space transformations.

Returns a NamedTuple `(result = x_optimal, info = ...)` with the optimum in
the same space and algorithm-specific optimizer information.
"""
function maximize_density end


# Optimization result mapped back to target space, preserving additional
# backend result fields. structdiff preserves the field types declared by the
# backend (e.g. an abstractly typed info field), keeping the return type
# inferrable:
function _optimum_result(r::NamedTuple, f_pretransform)
    inv_trafo = inverse(f_pretransform)
    r_rest = Base.structdiff(r, NamedTuple{(:result,)})
    return merge((result = inv_trafo(r.result), result_trafo = r.result, f_pretransform = f_pretransform), r_rest)
end
