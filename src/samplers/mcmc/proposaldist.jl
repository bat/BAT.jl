# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function proposaldist_logpdf(
    pdist::Distribution{Multivariate,Continuous},
    v_proposed::AbstractVector{<:Real},
    v_current::AbstractVector{<:Real}
)
    logpdf(pdist, v_proposed - v_current)
end


function proposal_rand!(
    rng::AbstractRNG,
    pdist::Distribution{Multivariate,Continuous},
    v_proposed::AbstractVector{<:Real},
    v_current::AbstractVector{<:Real}
)
    v_proposed .= v_current + rand(rng, pdist)
end


function mv_proposaldist(T::Type{<:AbstractFloat}, d::TDist, varndof::Integer)
    Σ = PDMat(Matrix(I(varndof) * one(T)))
    df = only(Distributions.params(d))
    μ = Fill(zero(eltype(Σ)), varndof)
    Distributions.GenericMvTDist(convert(T, df), μ, Σ)
end
