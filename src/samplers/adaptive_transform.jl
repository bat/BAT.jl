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


"""
    abstract type BAT.AbstractTransformInit

Abstract type for algorithms that initialize adaptive MCMC space
transformations.
"""
abstract type AbstractTransformInit end


"""
    struct BAT.PriorApproxTransformInit <: BAT.AbstractTransformInit

Initializes affine space transformations from the approximate covariance
and mean of the prior.
"""
struct PriorApproxTransformInit <: AbstractTransformInit end


"""
    struct BAT.PathfinderTransformInit <: BAT.AbstractTransformInit

Initializes affine space transformations from local Gaussian target
approximations obtained by running the Pathfinder algorithm (see
[`BAT.pathfinder_gaussian_fit`](@ref)) from each initial walker position.
Requires the [`BATContext`](@ref) to include an `ADSelector`.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PathfinderTransformInit <: AbstractTransformInit
    "Maximum number of L-BFGS iterations."
    maxiters::Int = 1000

    "L-BFGS history length."
    history_length::Int = 6

    "Number of Monte Carlo draws used to estimate the ELBO."
    ndraws_elbo::Int = 5
end


"""
    struct BAT.TriangularAffineTransform <: BAT.AbstractAdaptiveTransform

Adaptive affine space transformation `x = A * z + b` with a
lower-triangular matrix `A`, initialized via an
[`BAT.AbstractTransformInit`](@ref) algorithm.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct TriangularAffineTransform{I<:AbstractTransformInit} <: AbstractAdaptiveTransform
    "Transform initialization algorithm."
    init::I = PriorApproxTransformInit()
end

"""
    struct BAT.DiagonalAffineTransform <: BAT.AbstractAdaptiveTransform

Adaptive affine space transformation `x = A * z + b` with a diagonal
matrix `A`, initialized via an [`BAT.AbstractTransformInit`](@ref)
algorithm. Cheap to apply and to tune, but blind to correlations.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct DiagonalAffineTransform{I<:AbstractTransformInit} <: AbstractAdaptiveTransform
    "Transform initialization algorithm."
    init::I = PriorApproxTransformInit()
end


"""
    struct BAT.LowRankAffineTransform <: BAT.AbstractAdaptiveTransform

Adaptive affine space transformation `x = A * z + b` with `A` a
diagonal-plus-low-rank Gram factor (`A * A' == D + W * S * W'`,
represented as a MatrixShapedOperators Woodbury operator factor): a
diagonal base geometry plus a correction along the directions where a
diagonal geometry is insufficient. Tuning selects those directions by an
eigenvalue cutoff (see [`FisherTransformTuning`](@ref)), which
regularizes the geometry estimate compared to a full triangular matrix.

Note: applying the transformation costs O(rank * n_dims), but the current
geometry fitting still accumulates dense moments (O(n_dims^2) memory), so
the low-rank structure buys regularization and application cost, not yet
fitting cost - a fully projected low-rank estimator is planned.

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct LowRankAffineTransform{I<:AbstractTransformInit} <: AbstractAdaptiveTransform
    "Transform initialization algorithm."
    init::I = PriorApproxTransformInit()

    "Maximum rank of the non-diagonal correction, `0` means unlimited."
    max_rank::Int = 0

    "Relative eigenvalue cutoff: only directions in which the estimated
    geometry deviates from the diagonal base by a factor above `cutoff`
    (or below its inverse) enter the low-rank correction."
    cutoff::Float64 = 1.5
end

const AffineStructureTransform = Union{TriangularAffineTransform,DiagonalAffineTransform,LowRankAffineTransform}

# Adaptive transform initialization may take the initial walker positions
# into account:
function init_adaptive_transform(adaptive_transform::AbstractAdaptiveTransform, target::AbstractMeasure, ::Union{AbstractVector,Nothing}, context::BATContext)
    return init_adaptive_transform(adaptive_transform, target, context)
end

function init_adaptive_transform(adaptive_transform::AffineStructureTransform, target::AbstractMeasure, v_init::Union{AbstractVector,Nothing}, context::BATContext)
    M, b = _affine_init_moments(adaptive_transform.init, target, v_init, context)
    return MulAdd(_affine_init_A(adaptive_transform, M), b)
end

function init_adaptive_transform(adaptive_transform::AffineStructureTransform, target::AbstractMeasure, context::BATContext)
    return init_adaptive_transform(adaptive_transform, target, nothing, context)
end

# The matrix part of the initial transform in the structure the adaptive
# transform declares, from an (approximate) covariance estimate. Tuners
# keep that structure across transform updates:
_affine_init_A(::TriangularAffineTransform, M::AbstractMatrix) = cholesky(Positive, M).L

_affine_init_A(::DiagonalAffineTransform, M::AbstractMatrix) = Diagonal(sqrt.(diag(M)))

# The initial low-rank correction is empty, tuning fills it in:
_affine_init_A(::LowRankAffineTransform, M::AbstractMatrix) =
    _lowrank_gram_factor(diag(M), zeros(eltype(M), size(M, 1), 0), Symmetric(zeros(eltype(M), 0, 0)))

_lowrank_gram_factor(dvec::AbstractVector, W::AbstractMatrix, S::Symmetric) =
    rowgram_factor(woodbury_operator(Diagonal(dvec), W, S))

# Approximate covariance and mean the affine initialization is based on.
# TODO: MD, make typestable
function _affine_init_moments(::PriorApproxTransformInit, target::AbstractMeasure, ::Union{AbstractVector,Nothing}, ::BATContext)
    n = totalndof(varshape(target))
    return _approx_cov(target, n), _approx_mean(target, n)
end

function _affine_init_moments(::PathfinderTransformInit, ::AbstractMeasure, ::Nothing, ::BATContext)
    throw(ArgumentError("PathfinderTransformInit requires initial positions"))
end
