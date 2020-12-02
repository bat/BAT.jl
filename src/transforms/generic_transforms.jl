# This file is a part of BAT.jl, licensed under the MIT License (MIT).


uniform_cdf(x::Real, a::Real, b::Real) = (x - a) / (b - a)
uniform_invcdf(u::Real, a::Real, b::Real) = (b - a) * u + a
uniform_invcdf_ladj(u::Real, a::Real, b::Real) = log(b - a)


# ToDo: Cache inv(b - a)
struct UniformCDFTrafo{T<:Real} <: VariateTransform{Univariate,UnitSpace,MixedSpace}
    a::T
    b::T
end

ValueShapes.varshape(trafo::UniformCDFTrafo) = ScalarShape{Real}()

target_space(trafo::UniformCDFTrafo) = UnitSpace()
source_space(trafo::UniformCDFTrafo) = MixedSpace()

function apply_vartrafo_impl(trafo::UniformCDFTrafo, src_v::Real, prev_ladj::Real)
    trg_v = uniform_cdf(src_v, trafo.a, trafo.b)
    if isnan(prev_ladj)
        var_trafo_result(trg_v, src_v)
    else
        trafo_ladj = - uniform_invcdf_ladj(trg_v, trafo.a, trafo.b)
        var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
    end
end

function apply_vartrafo_impl(inv_trafo::InverseVT{<:UniformCDFTrafo}, src_v::Real, prev_ladj::Real)
    trafo = inv_trafo.orig
    trg_v = uniform_invcdf(src_v, trafo.a, trafo.b)
    if isnan(prev_ladj)
        var_trafo_result(trg_v, src_v)
    else
        trafo_ladj = uniform_invcdf_ladj(src_v, trafo.a, trafo.b)
        var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
    end
end



logistic_cdf(x::Real, mu::Real, theta::Real) = inv(exp(-(x - mu) / theta) + one(x))
logistic_invcdf(u::Real, mu::Real, theta::Real) = log(u / (one(u) - u)) * theta + mu
logistic_invcdf_ladj(u::Real, mu::Real, theta::Real) = -log((u - u^2) / theta)


# ToDo: Cache inv(theta)
struct LogisticCDFTrafo{T<:Real} <: VariateTransform{Univariate,UnitSpace,InfiniteSpace}
    mu::T
    theta::T
end

ValueShapes.varshape(trafo::LogisticCDFTrafo) = ScalarShape{Real}()

target_space(trafo::LogisticCDFTrafo) = UnitSpace()
source_space(trafo::LogisticCDFTrafo) = InfiniteSpace()

function apply_vartrafo_impl(trafo::LogisticCDFTrafo, src_v::Real, prev_ladj::Real)
    trg_v = logistic_cdf(src_v, trafo.mu, trafo.theta)
    if isnan(prev_ladj)
        var_trafo_result(trg_v, src_v)
    else
        trafo_ladj = - logistic_invcdf_ladj(trg_v, trafo.mu, trafo.theta)
        var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
    end
end

function apply_vartrafo_impl(inv_trafo::InverseVT{<:LogisticCDFTrafo}, src_v::Real, prev_ladj::Real)
    trafo = inv_trafo.orig
    trg_v = logistic_invcdf(src_v, trafo.mu, trafo.theta)
    if isnan(prev_ladj)
        var_trafo_result(trg_v, src_v)
    else
        trafo_ladj = logistic_invcdf_ladj(src_v, trafo.mu, trafo.theta)
        var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
    end
end



exponential_cdf(x::Real, theta::Real) = one(x) - exp(x / -theta)
exponential_invcdf(u::Real, theta::Real) = -theta * log(one(u) - u)
exponential_cdf_ladj(x::Real, theta::Real) = (x / -theta) - log(theta)


# ToDo: Cache inv(theta) and log(theta)
struct ExponentialCDFTrafo{T<:Real} <: VariateTransform{Univariate,UnitSpace,MixedSpace}
    theta::T
end

ValueShapes.varshape(trafo::ExponentialCDFTrafo) = ScalarShape{Real}()

target_space(trafo::ExponentialCDFTrafo) = UnitSpace()
source_space(trafo::ExponentialCDFTrafo) = MixedSpace()

function apply_vartrafo_impl(trafo::ExponentialCDFTrafo, src_v::Real, prev_ladj::Real)
    trg_v = exponential_cdf(src_v, trafo.theta)
    if isnan(prev_ladj)
        var_trafo_result(trg_v, src_v)
    else
        trafo_ladj = exponential_cdf_ladj(src_v, trafo.theta)
        var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
    end
end

function apply_vartrafo_impl(inv_trafo::InverseVT{<:ExponentialCDFTrafo}, src_v::Real, prev_ladj::Real)
    trafo = inv_trafo.orig
    trg_v = exponential_invcdf(src_v, trafo.theta)
    if isnan(prev_ladj)
        var_trafo_result(trg_v, src_v)
    else
        trafo_ladj = - exponential_cdf_ladj(trg_v, trafo.theta)
        var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
    end
end



# ToDo: Cache inv(theta)
struct ScaledLogTrafo{T<:Real} <: VariateTransform{Univariate,InfiniteSpace,MixedSpace}
    theta::T
end

ValueShapes.varshape(trafo::ScaledLogTrafo) = ScalarShape{Real}()

target_space(trafo::ScaledLogTrafo) = InfiniteSpace()
source_space(trafo::ScaledLogTrafo) = MixedSpace()

function apply_vartrafo_impl(trafo::ScaledLogTrafo, src_v::Real, prev_ladj::Real)
    trg_v = log(src_v / trafo.theta)
    if isnan(prev_ladj)
        var_trafo_result(trg_v, src_v)
    else
        trafo_ladj = - abs(trg_v + log(trafo.theta))
        var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
    end
end

function apply_vartrafo_impl(inv_trafo::InverseVT{<:ScaledLogTrafo}, src_v::Real, prev_ladj::Real)
    trafo = inv_trafo.orig
    trg_v = exp(src_v) * trafo.theta
    if isnan(prev_ladj)
        var_trafo_result(trg_v, src_v)
    else
        trafo_ladj = abs(src_v + log(trafo.theta))
        var_trafo_result(trg_v, src_v, trafo_ladj, prev_ladj)
    end
end



const InfiniteDist = Union{
    StandardUvNormal, StandardMvNormal,
    Cauchy, Gumbel, Laplace, Logistic, NoncentralT, Normal, MvNormal, NormalCanon, TDist
}

const PositiveDist = Union{
    BetaPrime, Chi, Chisq, Erlang, Exponential, FDist, Frechet, Gamma, InverseGamma,
    InverseGaussian, Kolmogorov, LogNormal, NoncentralChisq, NoncentralF, Rayleigh, Weibull
}

const UnitDist = Union{
    StandardUvUniform, StandardMvUniform,
    Beta, KSOneSided, NoncentralBeta
}

const UCTruncatedDist = Truncated{<:Distribution{Univariate,Continuous}}

const StdUvDist = Union{StandardUvUniform, StandardUvNormal}
const StdMvDist = Union{StandardMvUniform, StandardMvNormal}


getspace(d::InfiniteDist) = InfiniteSpace()
getspace(d::UnitDist) = UnitSpace()
getspace(d::Distribution) = MixedSpace()


getvartrafo(space::InfiniteSpace, dist::Normal) = IdentityVT(InfiniteSpace(), ScalarShape{Real}())
getvartrafo(space::InfiniteSpace, dist::PositiveDist) = BAT.ScaledLogTrafo(1)
getvartrafo(space::InfiniteSpace, dist::Uniform) = inv(LogisticCDFTrafo(0, 1)) ∘ UniformCDFTrafo(minimum(dist), maximum(dist))
getvartrafo(space::InfiniteSpace, dist::UnitDist) = inv(LogisticCDFTrafo(0, 1))
getvartrafo(space::InfiniteSpace, dist::AbstractMvNormal) = IdentityVT(InfiniteSpace(), ArrayShape{Real}(length(dist)))
# getvartrafo(space::InfiniteSpace, dist::Distributions.AbstractMvLogNormal) = ...
