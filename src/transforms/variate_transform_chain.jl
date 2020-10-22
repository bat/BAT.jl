# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct VariateTransformChain{
    VF <: VariateForm,
    ST <: VariateSpace,
    SF <: VariateSpace,
    N,
    NTT<:NTuple{N,VariateTransform{VF}}
} <: VariateTransform{VF,ST,SF}
    transforms::NTT
end

function VariateTransformChain(transforms::NTuple{N,VariateTransform{VF}}) where {N,VF}
    shape = varshape(first(values(transforms)))
    @argcheck all(isequal(shape), map(varshape, values(transforms)))
    ST = typeof(target_space(last(transforms)))
    SF = typeof(source_space(last(transforms)))
    NTT = typeof(transforms)
    VariateTransformChain{VF,ST,SF,N,NTT}(transforms)
end


import Base.∘

# ToDo: Utilize specialized merge operations at ends of chains:
∘(a::VariateTransform, b::VariateTransform) = VariateTransformChain((b, a))
∘(a::VariateTransform, b::VariateTransformChain) = VariateTransformChain((b, a.transforms...))
∘(a::VariateTransformChain, b::VariateTransform) = VariateTransformChain((b.transforms..., a))
∘(a::VariateTransformChain, b::VariateTransformChain) = VariateTransformChain((b.transforms..., a.transforms...))

∘(a::VariateTransform{VF}, b::IdentityVT{VF}) where VF = (@argcheck varshape(a) == varshape(b); a)
∘(a::IdentityVT{VF}, b::VariateTransform{VF}) where VF = (@argcheck varshape(a) == varshape(b); b)
∘(a::IdentityVT{VF}, b::IdentityVT{VF}) where VF = (@argcheck varshape(a) == varshape(b); a)


target_space(trafo::VariateTransformChain) = target_space(last(orig.transforms))
source_space(trafo::VariateTransformChain) = source_space(last(orig.transforms))


function _apply_trafo_chain(v::Any, prev_ladj::Real)
    (v = v, ladj = prev_ladj)
end

function _apply_trafo_chain(v::Any, prev_ladj::Real, t_first::VariateTransform, ts::VariateTransform...)
    r1 = apply_vartrafo_impl(t_first, v, prev_ladj)
    r2 = _apply_trafo_chain(r1.v, r1.ladj, ts...)
    (v = r2.v, ladj = r2.ladj)
end


Base.inv(trafo::VariateTransformChain) = VariateTransformChain(reverse(map(inv, trafo.transforms)))

ValueShapes.varshape(trafo::VariateTransformChain) = varshape(first(values(trafo.transforms)))

apply_vartrafo_impl(trafo::VariateTransformChain, v::Any, prev_ladj::Real) = _apply_trafo_chain(v, prev_ladj, trafo.transforms...)

apply_vartrafo_impl(trafo::InvVT{<:VariateTransformChain}, v::Any, prev_ladj::Real) = apply_vartrafo_impl(inv(trafo.orig), v, prev_ladj)


# ToDo: Better fix for ambiguity:
function apply_vartrafo_impl(trafo::VariateTransformChain{Univariate}, v::AbstractArray{<:Real,0}, prev_ladj::Real)
    r = apply_vartrafo_impl(trafo, v[], prev_ladj)
    (v = fill(r.v), ladj = r.ladj)
end
