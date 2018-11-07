# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function _rand_gamma_mt(rng::AbstractRNG, ::Type{T}, shape::Real) where {T<:Real}
    (shape <= 0) && throw(ArgumentError("Require shape > 0, got $shape"))

    α = T(shape)

    if (α <= 1)
        return _rand_gamma_mt(rng, T, α + 1) * convert(T, rand(rng))^(1/α)
    else
        k = T(3)
        d = α - 1/k;
        c = 1 / (k * √d);  # == 1 / √(k^2 * α - k)

        while true
            x = randn(rng, T)
            cx1 = c*x + 1
            if (0 < cx1)  # -1/c < x
                h_x = d * cx1^3  # hx(x) = d * (1 + c*x)^3
                u = convert(T, rand(rng));

                v = cx1^3;
                dv = d*v  # == h(x) = d * (1 + c*x)^3
                if (u > 0)
                    (u < 1 - T(0.0331) * x^4) && return dv
                    (log(u) < x^2/2 + (d - dv + d * log(v))) && return dv
                end
            end
        end
    end
end

export BATGammaMTSampler

struct BATGammaMTSampler{T} <: BATSampler{Univariate,Continuous}
    shape::T
    scale::T
end

BATGammaMTSampler(d::Gamma) = BATGammaMTSampler(shape(d), scale(d))

Base.eltype(s::BATGammaMTSampler{T}) where {T} = T

Random.rand(rng::AbstractRNG, s::BATGammaMTSampler) = s.scale * _rand_gamma_mt(rng, float(typeof(s.shape)), s.shape)

bat_sampler(d::Gamma) = BATGammaMTSampler(d)
