# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    maximize_density(f_logdensity, x_init::AbstractVector{<:Real}, algorithm, context::BATContext)

*BAT-internal, not part of stable public API.*

Maximize the log-density function `f_logdensity`, starting at `x_init`,
using optimization `algorithm`.

The user is responsible for shaping f_logdensity in a way that works well
with the algorithm (typically by transforming it to an unbounded space),
`maximize_density` does not apply automatic space transformations.

Returns a NamedTuple `(result = x_optimal, trace = ..., info = ...)` with
the optimum in the same space, an optional optimization trace and
algorithm-specific optimizer information.

`trace` is `nothing` unless trace recording was requested via the
algorithm (field `store_trace` of [`OptimAlg`](@ref) and
[`OptimizationAlg`](@ref)). When present, it is a `NamedTuple` of
iteration-indexed vectors: `v` (the iterates) and, depending on backend
and optimizer, `logd` (the log-density values at the iterates) and
`grad_logd` (the log-density gradients at the iterates).
"""
function maximize_density end


# Optimization result mapped back to target space, preserving additional
# backend result fields. structdiff preserves the field types declared by the
# backend (e.g. the abstractly typed trace and info fields), keeping the
# return type inferrable. The optimizer trace lives in the transformed
# search space, so it is renamed to trace_trafo, matching result_trafo:
function _optimum_result(r::NamedTuple, f_pretransform)
    inv_trafo = inverse(f_pretransform)
    r_rest = Base.structdiff(r, NamedTuple{(:result, :trace)})
    # The trace field keeps its declared (possibly abstract) type instead
    # of being re-narrowed by literal NamedTuple construction, which would
    # split the union and widen the merged return type:
    trace_part = NamedTuple{(:trace_trafo,), Tuple{fieldtype(typeof(r), :trace)}}((r.trace,))
    return merge(
        (result = inv_trafo(r.result), result_trafo = r.result, f_pretransform = f_pretransform),
        trace_part,
        r_rest
    )
end
