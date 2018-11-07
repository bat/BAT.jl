# Generic rand and rand! implementations similar to those in Distributions,
# but with an rng argument:

export BATSampler

abstract type BATSampler{F<:VariateForm,S<:ValueSupport} <: Sampleable{F,S} end


@inline Random.rand(s::BATSampler, args...) = rand(Random.GLOBAL_RNG, s, args...)
@inline Random.rand!(s::BATSampler, args...) = rand!(Random.GLOBAL_RNG, s, args...)

# To avoid ambiguity with Distributions:
@inline Random.rand(s::BATSampler, dims::Int...) = rand(Random.GLOBAL_RNG, s, dims...)
@inline Random.rand!(s::BATSampler, A::AbstractVector) = rand!(Random.GLOBAL_RNG, s, A)
@inline Random.rand!(s::BATSampler, A::AbstractMatrix) = rand!(Random.GLOBAL_RNG, s, A)


function Random.rand!(rng::AbstractRNG, s::BATSampler{Univariate}, A::AbstractArray)
    @inbounds @simd for i in 1:length(A)
        A[i] = rand(rng, s)
    end
    return A
end

Random.rand(rng::AbstractRNG, s::BATSampler{Univariate}, dims::Dims) =
    rand!(rng, s, Array{eltype(s)}(undef, dims))

Random.rand(rng::AbstractRNG, s::BATSampler{Univariate}, dims::Int...) =
    rand!(s, Array{eltype(s)}(undef, dims))


Random.rand(rng::AbstractRNG, s::BATSampler{Multivariate}) =
    rand!(rng, s, Vector{eltype(s)}(undef, length(s)))

Random.rand(rng::AbstractRNG, s::BATSampler{Multivariate}, n::Integer) =
    rand!(rng, s, Matrix{eltype(s)}(undef, length(s), n))

function Random.rand!(rng::AbstractRNG, s::BATSampler{Multivariate}, A::AbstractMatrix)
    _check_rand_compat(s, A)
    @inbounds for i = 1:size(A,2)
        rand!(rng, s, view(A,:,i))
    end
    return A
end

Random.rand(rng::AbstractRNG, s::BATSampler{Multivariate}, n::Int) = rand!(rng, s, Matrix{eltype(s)}(undef, length(s), n))
