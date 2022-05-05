# Code imported from Pathfinder.jl (https://github.com/sethaxen/Pathfinder.jl)
# under MIT License.
# Copyright (c) 2021 Seth Axen <seth.axen@gmail.com> and contributors

# ToDo: Remove as soon as available from a lightweight source.

struct WoodburyPDMat{
    T<:Real,
    TA<:AbstractMatrix{T},
    TB<:AbstractMatrix{T},
    TD<:AbstractMatrix{T},
    TUA<:Union{Diagonal{T},UpperTriangular{T}},
    TQ<:AbstractMatrix{T},
    TUC<:Union{Diagonal{T},UpperTriangular{T}},
} <: PDMats.AbstractPDMat{T}
    A::TA
    B::TB
    D::TD
    UA::TUA
    Q::TQ
    UC::TUC
end

function WoodburyPDMat(
    A::AbstractMatrix{T}, B::AbstractMatrix{T}, D::AbstractMatrix{T}
) where {T}
    cholA = cholesky(A)
    UA = cholA.U
    Q, R = qr(UA' \ B)
    cholC = cholesky(Symmetric(muladd(R, D * R', I)))
    return WoodburyPDMat(A, B, D, UA, Q, cholC.U)
end
function WoodburyPDMat(A, B, D)
    T = Base.promote_eltype(A, B, D)
    return WoodburyPDMat(
        convert(AbstractMatrix{T}, A),
        convert(AbstractMatrix{T}, B),
        convert(AbstractMatrix{T}, D),
    )
end

Base.Matrix(W::WoodburyPDMat) = Matrix(Symmetric(muladd(W.B, W.D * W.B', W.A)))

function Base.AbstractMatrix{T}(W::WoodburyPDMat) where {T}
    return WoodburyPDMat(
        map(k -> convert(AbstractMatrix{T}, getfield(W, k)), fieldnames(typeof(W)))...
    )
end

function Base.getindex(W::WoodburyPDMat, i::Int, j::Int)
    B = W.B
    isempty(B) && return W.A[i, j]
    return @views W.A[i, j] + dot(B[i, :], W.D, B[j, :])
end

Base.adjoint(W::WoodburyPDMat) = W

Base.transpose(W::WoodburyPDMat) = W

function Base.inv(W::WoodburyPDMat)
    invUA = inv(W.UA)
    Anew = invUA * invUA'
    Bnew = invUA * Matrix(W.Q)
    invUC = inv(W.UC)
    Dnew = muladd(invUC, invUC', -I)
    return WoodburyPDMat(Anew, Bnew, Dnew)
end

LinearAlgebra.det(W::WoodburyPDMat) = exp(logdet(W))
LinearAlgebra.logdet(W::WoodburyPDMat) = 2 * (logdet(W.UA) + logdet(W.UC))
function LinearAlgebra.logabsdet(W::WoodburyPDMat)
    l = logdet(W)
    return (l, one(l))
end

function LinearAlgebra.diag(W::WoodburyPDMat)
    D = W.D isa Diagonal ? W.D : Symmetric(W.D)
    return diag(W.A) + map(b -> dot(b, D, b), eachrow(W.B))
end

function LinearAlgebra.lmul!(W::WoodburyPDMat, x::StridedVecOrMat)
    UA = W.UA
    UC = W.UC
    Q = W.Q
    k = minimum(size(W.B))
    lmul!(UA, x)
    lmul!(Q', x)
    x1 = x isa AbstractVector ? view(x, 1:k) : view(x, 1:k, :)
    lmul!(UC, x1)
    lmul!(UC', x1)
    lmul!(Q, x)
    lmul!(UA', x)
    return x
end

function LinearAlgebra.mul!(y::AbstractVector, W::WoodburyPDMat, x::StridedVecOrMat)
    return lmul!(W, copyto!(y, x))
end
function LinearAlgebra.mul!(y::AbstractMatrix, W::WoodburyPDMat, x::StridedVecOrMat)
    return lmul!(W, copyto!(y, x))
end

function Base.:*(W::WoodburyPDMat, c::Real)
    c > 0 || return Matrix(W) * c
    return WoodburyPDMat(W.A * c, W.B * one(c), W.D * c)
end

PDMats.dim(W::WoodburyPDMat) = size(W.A, 1)

function PDMats.invquad(W::WoodburyPDMat, x::AbstractVector{T}) where {T}
    v = W.Q' * (W.UA' \ x)
    n, m = size(W.B)
    k = min(m, n)
    return @views sum(abs2, W.UC' \ v[1:k]) + sum(abs2, v[(k + 1):n])
end

function PDMats.invquad!(r::AbstractArray, W::WoodburyPDMat, x::StridedMatrix{T}) where {T}
    v = lmul!(W.Q', W.UA' \ x)
    k = minimum(size(W.B))
    @views ldiv!(W.UC', v[1:k, :])
    colwise_sumsq!(r, v)
    return r
end

function PDMats.quad!(r::AbstractArray, W::WoodburyPDMat, x::StridedMatrix{T}) where {T}
    v = lmul!(W.Q', W.UA * x)
    k = minimum(size(W.B))
    @views lmul!(W.UC, v[1:k, :])
    colwise_sumsq!(r, v)
    return r
end

function PDMats.quad(W::WoodburyPDMat, x::AbstractVector{T}) where {T}
    v = W.Q' * (W.UA * x)
    n, m = size(W.B)
    k = min(m, n)
    return @views sum(abs2, W.UC * v[1:k]) + sum(abs2, v[(k + 1):n])
end

function PDMats.unwhiten!(
    r::StridedVecOrMat{T}, W::WoodburyPDMat, x::StridedVecOrMat{T}
) where {T}
    k = minimum(size(W.B))
    copyto!(r, x)
    @views lmul!(W.UC', x isa AbstractVector ? r[1:k] : r[1:k, :])
    lmul!(W.Q, r)
    lmul!(W.UA', r)
    return r
end

function invunwhiten!(
    r::StridedVecOrMat{T}, W::WoodburyPDMat, x::StridedVecOrMat{T}
) where {T}
    k = minimum(size(W.B))
    copyto!(r, x)
    @views ldiv!(W.UC, x isa AbstractVector ? r[1:k] : r[1:k, :])
    lmul!(W.Q, r)
    ldiv!(W.UA, r)
    return r
end

# adapted from https://github.com/JuliaStats/PDMats.jl/blob/master/src/utils.jl
function colwise_sumsq!(r::AbstractArray, a::AbstractMatrix)
    n = length(r)
    @assert n == size(a, 2)
    @inbounds for j in 1:n
        v = zero(eltype(a))
        @simd for i in axes(a, 1)
            v += abs2(a[i, j])
        end
        r[j] = v
    end
    return r
end
