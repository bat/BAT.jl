# This file is a part of BAT.jl, licensed under the MIT License (MIT).

abstract type AbstractAdaptiveTransform end


struct CustomTransform{F} <: AbstractAdaptiveTransform 
    f::F
end

CustomTransform() = CustomTransform(identity)

init_adaptive_transform(at::CustomTransform, ::AbstractMeasure, ::BATContext) = at.f


struct NoAdaptiveTransform <: AbstractAdaptiveTransform end

init_adaptive_transform(::NoAdaptiveTransform, ::AbstractMeasure, ::BATContext) = identity

"""
    struct AdaptiveTransformChain <: AbstractAdaptiveTransform

A chain of adaptive space transformations, applied innermost first:
`x = f[end](...f[1](z)...)`. Tuned via [`MultiTrafoTuning`](@ref), with
one transform tuning per component.

Note: target-moment-based initializations (like
`PriorApproxTransformInit`) are only exact for the outermost component;
inner components should typically use `BAT.UnitTransformInit`.

Constructors:

* ```AdaptiveTransformChain(f::Tuple{Vararg{AbstractAdaptiveTransform}})```
"""
struct AdaptiveTransformChain{FT<:Tuple{Vararg{AbstractAdaptiveTransform}}} <: AbstractAdaptiveTransform
    f::FT
end

export AdaptiveTransformChain

function init_adaptive_transform(
    adaptive_transform::AdaptiveTransformChain,
    target::AbstractMeasure,
    v_init::Union{AbstractVector,Nothing},
    context::BATContext
)
    fs = adaptive_transform.f
    n = length(fs)
    initialized_trafos = Vector{Function}(undef, n)

    # Components are initialized outermost first: the outermost component
    # sees the target-space positions, inner components see the positions
    # pulled back through the already-initialized outer components. The
    # target itself is not pulled back, so target-moment-based
    # initializations are only exact for the outermost component:
    vs = v_init
    for j in n:-1:1
        f_j = init_adaptive_transform(fs[j], target, vs, context)
        initialized_trafos[j] = f_j
        if !isnothing(vs) && j > 1
            vs = inverse(f_j).(vs)
        end
    end

    # A tuple-based chain keeps the composed transform type-stable in the
    # sampling hot loop:
    return fchain((initialized_trafos...,))
end

function init_adaptive_transform(
    adaptive_transform::AdaptiveTransformChain,
    target::AbstractMeasure,
    context::BATContext
)
    return init_adaptive_transform(adaptive_transform, target, nothing, context)
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


"""
    abstract type BAT.AbstractTransformInit

*BAT-internal, not part of stable public API.*

Abstract type for algorithms that initialize adaptive MCMC space
transformations.
"""
abstract type AbstractTransformInit end


"""
    struct BAT.UnitTransformInit <: BAT.AbstractTransformInit

*BAT-internal, not part of stable public API.*

Initializes affine space transformations to the identity map. The natural
initialization for inner components of an [`AdaptiveTransformChain`](@ref),
where the outer components already carry the target geometry.
"""
struct UnitTransformInit <: AbstractTransformInit end

_unit_transform_eltype(::AbstractVector{<:AbstractVector{P}}, ::BATContext) where {P<:Real} = float(P)
_unit_transform_eltype(::Nothing, context::BATContext) = get_precision(context)

function _affine_init_moments(::UnitTransformInit, target::AbstractMeasure, v_init::Union{AbstractVector,Nothing}, context::BATContext)
    n = totalndof(varshape(target))
    T = _unit_transform_eltype(v_init, context)
    return Matrix{T}(I, n, n), zeros(T, n)
end


"""
    struct BAT.PriorApproxTransformInit <: BAT.AbstractTransformInit

*BAT-internal, not part of stable public API.*

Initializes affine space transformations from the approximate covariance
and mean of the prior.
"""
struct PriorApproxTransformInit <: AbstractTransformInit end



"""
    abstract type BAT.AbstractAffineTransform <: BAT.AbstractAdaptiveTransform

*BAT-internal, not part of stable public API.*

Supertype for adaptive affine space transformations `x = A * z + b`,
which differ in the structure they impose on `A`.

# Implementation

Subtypes must have a field `init` holding a
[`BAT.AbstractTransformInit`](@ref) algorithm, and must specialize
`BAT._affine_init_A` to build `A` in their structure from an approximate
covariance. Transform tunings maintain that structure across updates and
specialize on the concrete type in turn (see `BAT._fisher_estimator` for
[`FisherTransformTuning`](@ref)).
"""
abstract type AbstractAffineTransform <: AbstractAdaptiveTransform end


"""
    struct BAT.TriangularAffineTransform <: BAT.AbstractAffineTransform

*BAT-internal, not part of stable public API.*

Adaptive affine space transformation `x = A * z + b` with a
lower-triangular matrix `A`, initialized via an
[`BAT.AbstractTransformInit`](@ref) algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TriangularAffineTransform{I<:AbstractTransformInit} <: AbstractAffineTransform
    "Transform initialization algorithm."
    init::I = PriorApproxTransformInit()
end

"""
    struct BAT.DiagonalAffineTransform <: BAT.AbstractAffineTransform

*BAT-internal, not part of stable public API.*

Adaptive affine space transformation `x = A * z + b` with a diagonal
matrix `A`, initialized via an [`BAT.AbstractTransformInit`](@ref)
algorithm. Cheap to apply and to tune, but blind to correlations.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct DiagonalAffineTransform{I<:AbstractTransformInit} <: AbstractAffineTransform
    "Transform initialization algorithm."
    init::I = PriorApproxTransformInit()
end


"""
    struct BAT.LowRankAffineTransform <: BAT.AbstractAffineTransform

*Experimental feature, not part of stable public API.*

Adaptive affine space transformation `x = A * z + b` with `A` a
diagonal-plus-low-rank Gram factor (`A * A' == D + W * S * W'`,
represented as a MatrixShapedOperators Woodbury operator factor): a
diagonal base geometry plus a correction along the directions where a
diagonal geometry is insufficient. Tuning selects those directions by an
eigenvalue cutoff (see [`FisherTransformTuning`](@ref)), which
regularizes the geometry estimate compared to a full triangular matrix.

Applying the transformation costs O(rank * n_dims). Initialization from
an approximate covariance honors `cutoff` and `max_rank` and may use a
dense decomposition.

Dynamic Fisher tuning currently makes one rank-one correction attempt
for at most 32 dimensions when `cutoff >= 1.5`. It fits from a fixed
window and uses a guard followed by held-out validation. HMC keeps the
diagonal kernel during both. MALA installs the candidate provisionally
during its guard so it can retune and mix, then keeps it after acceptance
or restores the diagonal transform after rejection. The correction must beat
both the frozen diagonal base and its own diagonal projection. Each paired
Fisher-loss comparison needs a positive one-sided 99% normal lower bound with
at least 20 effective observations. This rejects purely diagonal updates
without restricting the shape of a correlation direction. Other settings tune
only the diagonal base.

This is a conservative held-out heuristic, not a finite-sample error-rate
guarantee. Its asymptotic interpretation assumes stationary, mixing validation
chains, finite long-run paired-loss variance, and independent walkers.
Heavy-tailed cases outside those assumptions have only empirical evidence.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct LowRankAffineTransform{I<:AbstractTransformInit} <: AbstractAffineTransform
    "Transform initialization algorithm."
    init::I = PriorApproxTransformInit()

    "Maximum rank of the non-diagonal correction during initialization,
    `0` means no explicit cap. Dynamic Fisher tuning currently attempts
    one rank-one correction."
    max_rank::Int = 0

    "Relative eigenvalue cutoff used during initialization. Dynamic
    Fisher tuning requires `cutoff >= 1.5` and uses its fixed validated
    rank-one policy."
    cutoff::Float64 = 1.5
end

# Adaptive transform initialization may take the initial walker positions
# into account:
function init_adaptive_transform(adaptive_transform::AbstractAdaptiveTransform, target::AbstractMeasure, ::Union{AbstractVector,Nothing}, context::BATContext)
    return init_adaptive_transform(adaptive_transform, target, context)
end

function init_adaptive_transform(adaptive_transform::AbstractAffineTransform, target::AbstractMeasure, v_init::Union{AbstractVector,Nothing}, context::BATContext)
    M, b = _affine_init_moments(adaptive_transform.init, target, v_init, context)
    return MulAdd(_affine_init_A(adaptive_transform, M), b)
end

function init_adaptive_transform(adaptive_transform::AbstractAffineTransform, target::AbstractMeasure, context::BATContext)
    return init_adaptive_transform(adaptive_transform, target, nothing, context)
end

# The matrix part of the initial transform in the structure the adaptive
# transform declares, from an (approximate) covariance estimate. Tuners
# keep that structure across transform updates:
_affine_init_A(::TriangularAffineTransform, M::AbstractMatrix) = cholesky(Positive, M).L

_affine_init_A(::DiagonalAffineTransform, M::AbstractMatrix) = Diagonal(sqrt.(diag(M)))

# The initial geometry keeps the structure the transform declares: a
# diagonal base plus the eigenvalue-thresholded correction of the given
# (approximate) covariance - e.g. the diag-plus-low-rank structure of a
# Pathfinder fit survives instead of being flattened to its diagonal:
function _affine_init_A(at::LowRankAffineTransform, M::AbstractMatrix)
    # The parameter invariants belong to the transform, not to any one
    # tuner that happens to consume it:
    @argcheck at.cutoff > 1
    @argcheck at.max_rank >= 0
    dvec, W, S = _lowrank_decomposition(M, at.cutoff, at.max_rank)
    return _lowrank_gram_factor(dvec, W, S)
end

_lowrank_gram_factor(dvec::AbstractVector, W::AbstractMatrix, S::Symmetric) =
    rowgram_factor(woodbury_operator(Diagonal(dvec), W, S))

# Eigenvalue-thresholded low-rank correction on top of a diagonal base:
# in diagonally standardized coordinates, only directions in which the
# geometry deviates from the base by a factor beyond `cutoff` (or its
# inverse) enter the correction. Returns (W, S, λ_kept, V_kept) of the
# representation G = D + W S Wᵀ with dsq = sqrt.(diag(D)) and (λ, V) the
# eigen pairs of the standardized geometry:
function _lowrank_correction(dsq::AbstractVector, λ::AbstractVector, V::AbstractMatrix, cutoff::Real, max_rank::Integer)
    keep = findall(l -> l > cutoff || l < inv(cutoff), λ)
    if max_rank > 0 && length(keep) > max_rank
        logsize = abs.(log.(max.(λ[keep], floatmin(float(eltype(λ))))))
        keep = keep[sortperm(logsize, rev = true)[1:max_rank]]
    end
    λ_kept = λ[keep]
    V_kept = V[:, keep]
    W = (dsq .* V_kept) .* sqrt.(abs.(λ_kept .- 1))'
    S = Symmetric(Matrix(Diagonal(sign.(λ_kept .- 1))))
    return W, S, λ_kept, V_kept
end

# Diagonal-plus-low-rank representation of a dense SPD geometry estimate:
function _lowrank_decomposition(G_dense::AbstractMatrix, cutoff::Real, max_rank::Integer)
    dvec = diag(G_dense)
    dsq = sqrt.(dvec)
    E = eigen(Symmetric(Matrix(G_dense ./ (dsq .* dsq'))))
    W, S, _, _ = _lowrank_correction(dsq, E.values, E.vectors, cutoff, max_rank)
    return dvec, W, S
end

# Approximate covariance and mean the affine initialization is based on.
# TODO: MD, make typestable
function _affine_init_moments(::PriorApproxTransformInit, target::AbstractMeasure, ::Union{AbstractVector,Nothing}, ::BATContext)
    n = totalndof(varshape(target))
    return _approx_cov(target, n), _approx_mean(target, n)
end
