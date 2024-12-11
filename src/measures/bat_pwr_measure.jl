# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    BATPwrMeasure

*BAT-internal, not part of stable public API.*
"""
struct BATPwrMeasure{M,D<:Dims} <: BATMeasure
    parent::M
    sz::D
end

function BATMeasure(m::MeasureBase.PowerMeasure{<:AbstractMeasure,<:Tuple{Vararg{Base.OneTo}}})
    BATPwrMeasure(batmeasure(_pwr_base(m)), _pwr_size(m))
end

MeasureBase.powermeasure(m::BATMeasure, dims::Dims) = _bat_pwrmeasure(m, dims)
MeasureBase.powermeasure(m::BATMeasure, axes::Tuple{Vararg{Base.OneTo}}) = _bat_pwrmeasure(m, map(length(axes)))
MeasureBase.powermeasure(m::BATMeasure, ::Tuple{}) = m

_bat_pwrmeasure(m::BATMeasure, dims::Tuple{Vararg{Integer}}) = BATPwrMeasure(m, dims)
_bat_pwrmeasure(m::BATMeasure, dims::Tuple{<:Integer}) = BATPwrMeasure(m, dims)
_bat_pwrmeasure(m::BATDistMeasure{<:UnivariateDistribution}, dims::Tuple{<:Integer}) = batmeasure(product_distribution(Fill(Distribution(m), only(dims))))
_bat_pwrmeasure(::BATDistMeasure{<:StandardUvUniform}, dims::Tuple{<:Integer}) = batmeasure(StandardMvUniform(only(dims)))
_bat_pwrmeasure(::BATDistMeasure{<:StandardUvNormal}, dims::Tuple{<:Integer}) = batmeasure(StandardMvNormal(only(dims)))


_pwr_base(m::BATPwrMeasure) = m.parent
_pwr_axes(m::BATPwrMeasure) = map(Base.OneTo, m.sz)
_pwr_size(m::BATPwrMeasure) = m.sz


function _cartidxs(axs::Tuple{Vararg{AbstractUnitRange,N}}) where {N}
    CartesianIndices(map(_dynamic, axs))
end

function Base.rand(gen::GenContext, m::BATPwrMeasure)
    cunit = get_compute_unit(gen)
    adapt(cunit, map(_ -> rand(rng, m.parent), _cartidxs(m.axes)))
end

function Base.rand(gen::GenContext, m::BATPwrMeasure{<:BATDistMeasure})
    X = rand(get_rng(gen), m.parent.dist, size(marginals(m))...)
    reshaped_X = _reshape_rand_n_output(X)
    gen_adapt(gen, reshaped_X)
end

function Base.rand(gen::GenContext, m::BATPwrMeasure{<:DensitySampleMeasure})
    dims = size(marginals(m))
    n = prod(dims)
    idxs = _rand_subsample_idxs(gen, m, n)
    result_vec = gen_adapt(gen, m.parent._smpls.v[idxs])
    return reshape(result_vec, dims)
end



function MeasureBase.testvalue(::Type{T}, m::BATPwrMeasure) where {T}
    Fill(testvalue(T, _pwr_base(m)), _pwr_size(m))
end

function MeasureBase.testvalue(m::BATPwrMeasure)
    Fill(testvalue(_pwr_base(m)), _pwr_size(m))
end

MeasureBase.marginals(m::BATPwrMeasure) = Fill(_pwr_base(m), _pwr_size(m))

@inline function DensityInterface.logdensityof(m::BATPwrMeasure, x)
    @assert size(x) == _pwr_size(m)
    m_base = _pwr_base(m)
    sum(x) do x_i
        logdensity_def(m_base, x_i)
    end
end


@inline function MeasureBase.insupport(m::BATPwrMeasure, x)
    @assert size(x) == _pwr_size(m)
    m_base = _pwr_base(m)
    insupp = broadcast(x) do x_i
        # https://github.com/SciML/Static.jl/issues/36
        dynamic(insupport(m_base, x_i))
    end
    _all(insupp)
end

MeasureBase.getdof(m::BATPwrMeasure) = getdof(_pwr_base(m)) * prod(_pwr_size(m))

MeasureBase.massof(m::BATPwrMeasure) = massof(_pwr_base(m))^prod(_pwr_size(m))

MeasureBase.params(m::BATPwrMeasure) = params(_pwr_base(m))
