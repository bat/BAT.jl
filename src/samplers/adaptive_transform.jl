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

function _affine_init_moments(::UnitTransformInit, target::AbstractMeasure, ::Union{AbstractVector,Nothing}, ::BATContext)
    n = totalndof(varshape(target))
    return Matrix{Float64}(I, n, n), zeros(n)
end


"""
    struct BAT.PriorApproxTransformInit <: BAT.AbstractTransformInit

*BAT-internal, not part of stable public API.*

Initializes affine space transformations from the approximate covariance
and mean of the prior.
"""
struct PriorApproxTransformInit <: AbstractTransformInit end


"""
    struct BAT.PathfinderTransformInit <: BAT.AbstractTransformInit

*Experimental feature, not part of stable public API.*

Initializes affine space transformations from local Gaussian target
approximations obtained by running the Pathfinder algorithm (see
[`BAT.pathfinder_gaussian_fit`](@ref)) from each initial walker position.

Requires the [`BATContext`](@ref) to include an `ADSelector` and a
gradient-based optimization backend: by default the Optim package must be
loaded, alternatively set `optalg` explicitly.

See [L. Zhang, B. Carpenter, A. Gelman and A. Vehtari, "Pathfinder:
Parallel quasi-Newton variational inference", JMLR 23(306)
(2022)](https://jmlr.org/papers/v23/21-0889.html).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct PathfinderTransformInit{A} <: AbstractTransformInit
    "Density maximization backend that generates the L-BFGS trajectory,
    must record iterates and gradients (see [`maximize_density`](@ref))."
    optalg::A = _default_pathfinder_optalg()

    "L-BFGS history length of the inverse-Hessian estimates."
    history_length::Int = 6

    "Number of Monte Carlo draws used to estimate the ELBO."
    ndraws_elbo::Int = 5
end


"""
    struct BAT.TriangularAffineTransform <: BAT.AbstractAdaptiveTransform

*BAT-internal, not part of stable public API.*

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

*BAT-internal, not part of stable public API.*

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

*Experimental feature, not part of stable public API.*

Adaptive affine space transformation `x = A * z + b` with `A` a
diagonal-plus-low-rank Gram factor (`A * A' == D + W * S * W'`,
represented as a MatrixShapedOperators Woodbury operator factor): a
diagonal base geometry plus a correction along the directions where a
diagonal geometry is insufficient. Tuning selects those directions by an
eigenvalue cutoff (see [`FisherTransformTuning`](@ref)), which
regularizes the geometry estimate compared to a full triangular matrix.

Applying the transformation costs O(rank * n_dims), and the geometry
fitting is projected: it accumulates diagonal moments plus a bounded
window of recent draws and solves the Fisher problem in the joint thin
subspace of the window, so estimation memory and fitting cost are
O(n_dims * window) plus small-matrix work - no dense moments or solves
(the transform initialization from an approximate covariance is the one
remaining dense step).

Constructors:

* ```$(FUNCTIONNAME)(; fields...)```

Fields:

$(TYPEDFIELDS)
"""
@with_kw struct LowRankAffineTransform{I<:AbstractTransformInit} <: AbstractAdaptiveTransform
    "Transform initialization algorithm."
    init::I = PriorApproxTransformInit()

    "Maximum rank of the non-diagonal correction, `0` means no explicit
    cap. Note that the estimable rank is always bounded by the size of
    the estimation window of recent draws (see `FisherTransformTuning`)."
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

function _affine_init_moments(::PathfinderTransformInit, ::AbstractMeasure, ::Nothing, ::BATContext)
    throw(ArgumentError("PathfinderTransformInit requires initial positions"))
end
