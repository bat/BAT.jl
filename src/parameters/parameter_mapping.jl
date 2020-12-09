# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct VarMapping{
    VFV<:AbstractVector{<:Real},
    VFI<:AbstractVector{Int},
    VVI<:AbstractVector{Int}
}
    fixed_idxs::VFI
    fixed_values::VFV
    variable_idxs::VVI
end


function VarMapping(bounds::HyperRectBounds)
    vol = bounds.vol
    lo_hi_cmp = vol.lo .≈ vol.hi
    fixed_idxs = findall(v -> v == true, lo_hi_cmp)
    fixed_values = vol.lo[fixed_idxs]
    variable_idxs = findall(v -> v == false, lo_hi_cmp)
    VarMapping(fixed_idxs, fixed_values, variable_idxs)
end


function apply_varmap!(
    mapped_v::AbstractVector{<:Real},
    varmap::VarMapping,
    v::AbstractVector{<:Real}
)
    mapped_v[varmap.fixed_idxs] = varmap.fixed_values
    mapped_v[varmap.variable_idxs] = v
    mapped_v
end

function apply_varmap!(
    mapped_v::VectorOfSimilarVectors{<:Real},
    varmap::VarMapping,
    v::VectorOfSimilarVectors{<:Real}
)
    flatview(mapped_v)[varmap.fixed_idxs, :] = varmap.fixed_values
    flatview(mapped_v)[varmap.variable_idxs, :] = flatview(v)
    mapped_v
end

function apply_varmap(varmap::VarMapping, v::AbstractVector{<:Real})
    mapped_v = similar(v, size(varmap.fixed_idxs, 1) + size(varmap.variable_idxs, 1))
    apply_varmap!(mapped_v, varmap, v)
end

function apply_varmap(varmap::VarMapping, v::VectorOfSimilarVectors{<:Real})
    mapped_v = VectorOfSimilarVectors(similar(flatview(v), size(varmap.fixed_varidxs, 1) + size(varmap.variable_idxs, 1), size(flatview(v), 2)))
    apply_varmap!(mapped_v, varmap, v)
end


function invapply_varmap!(
    v::AbstractVector{<:Real},
    varmap::VarMapping,
    mapped_v::AbstractVector{<:Real}
)
    v[:] = view(mapped_v, varmap.variable_idxs)
    v
end

function invapply_varmap!(
    v::VectorOfSimilarVectors{<:Real},
    varmap::VarMapping,
    mapped_v::VectorOfSimilarVectors{<:Real}
)
    flatview(v)[:, :] = view(flatview(mapped_v), varmap.variable_idxs, :)
    v
end

function invapply_varmap(varmap::VarMapping, mapped_v::AbstractVector{<:Real})
    v = similar(mapped_v, size(varmap.variable_idxs, 1))
    invapply_varmap!(v, varmap, mapped_v)
end

function invapply_varmap(varmap::VarMapping, mapped_v::VectorOfSimilarVectors{<:Real})
    v = VectorOfSimilarVectors(similar(flatview(mapped_v), size(varmap.variable_idxs, 1), size(flatview(mapped_v), 2)))
    invapply_varmap!(v, varmap, mapped_v)
end


function invapply_varmap(varmap::VarMapping, bounds::HyperRectBounds)
    idxs = varmap.variable_idxs
    vol = HyperRectVolume(bounds.vol.lo[idxs], bounds.vol.hi[idxs])
    HyperRectBounds(vol)
end

function invapply_varmap(varmap::VarMapping, bounds::NoVarBounds)
    NoVarBounds(size(varmap.variable_idxs, 1))
end
