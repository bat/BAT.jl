# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ParameterMapping{
    VFV<:AbstractVector{<:Real},
    VFI<:AbstractVector{Int},
    VVI<:AbstractVector{Int}
}
    fixed_idxs::VFI
    fixed_values::VFV
    variable_idxs::VVI
end

export ParameterMapping


function ParameterMapping(bounds::HyperRectBounds)
    vol = bounds.vol
    lo_hi_cmp = vol.lo .≈ vol.hi
    fixed_idxs = findall(x -> x == true, lo_hi_cmp)
    fixed_values = vol.lo[fixed_idxs]
    variable_idxs = findall(x -> x == false, lo_hi_cmp)
    ParameterMapping(fixed_idxs, fixed_values, variable_idxs)
end


export map_params, map_params!

function map_params!(
    mapped_params::AbstractVector{<:Real},
    parmap::ParameterMapping,
    params::AbstractVector{<:Real}
)
    mapped_params[parmap.fixed_idxs] = parmap.fixed_values
    mapped_params[parmap.variable_idxs] = params
    mapped_params
end

function map_params!(
    mapped_params::VectorOfSimilarVectors{<:Real},
    parmap::ParameterMapping,
    params::VectorOfSimilarVectors{<:Real}
)
    flatview(mapped_params)[parmap.fixed_idxs, :] = parmap.fixed_values
    flatview(mapped_params)[parmap.variable_idxs, :] = flatview(params)
    mapped_params
end

function map_params(parmap::ParameterMapping, params::AbstractVector{<:Real})
    mapped_params = similar(params, size(parmap.fixed_idxs, 1) + size(parmap.variable_idxs, 1))
    map_params!(mapped_params, parmap, params)
end

function map_params(parmap::ParameterMapping, params::VectorOfSimilarVectors{<:Real})
    mapped_params = VectorOfSimilarVectors(similar(flatview(params), size(parmap.fixed_paramsidxs, 1) + size(parmap.variable_idxs, 1), size(flatview(params), 2)))
    map_params!(mapped_params, parmap, params)
end


export invmap_params, invmap_params!

function invmap_params!(
    params::AbstractVector{<:Real},
    parmap::ParameterMapping,
    mapped_params::AbstractVector{<:Real}
)
    params[:] = view(mapped_params, parmap.variable_idxs)
    params
end

function invmap_params!(
    params::VectorOfSimilarVectors{<:Real},
    parmap::ParameterMapping,
    mapped_params::VectorOfSimilarVectors{<:Real}
)
    flatview(params)[:, :] = view(flatview(mapped_params), parmap.variable_idxs, :)
    params
end

function invmap_params(parmap::ParameterMapping, mapped_params::AbstractVector{<:Real})
    params = similar(mapped_params, size(parmap.variable_idxs, 1))
    invmap_params!(params, parmap, mapped_params)
end

function invmap_params(parmap::ParameterMapping, mapped_params::VectorOfSimilarVectors{<:Real})
    params = VectorOfSimilarVectors(similar(flatview(mapped_params), size(parmap.variable_idxs, 1), size(flatview(mapped_params), 2)))
    invmap_params!(params, parmap, mapped_params)
end


function invmap_params(parmap::ParameterMapping, bounds::HyperRectBounds)
    idxs = parmap.variable_idxs
    vol = HyperRectVolume(bounds.vol.lo[idxs], bounds.vol.hi[idxs])
    bt = bounds.bt[idxs]
    HyperRectBounds(vol, bt)
end

function invmap_params(parmap::ParameterMapping, bounds::NoParamBounds)
    NoParamBounds(size(parmap.variable_idxs, 1))
end
