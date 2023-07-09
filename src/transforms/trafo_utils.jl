# This file is a part of BAT.jl, licensed under the MIT License (MIT).


broadcast_trafo(f::Any, v_src::Any) = f.(v_src)


function broadcast_trafo(::typeof(identity), v_src::Any)
    v_src
end

function broadcast_trafo(
    trafo::Any,
    v_src::Union{ArrayOfSimilarVectors{<:Real},ShapedAsNTArray}
)
    vs_trg = trafo(elshape(v_src))
    R = eltype(unshaped(trafo(first(v_src)), vs_trg))
    v_src_us = unshaped.(v_src)
    trafo_us = unshaped(trafo)

    n = length(eachindex(v_src_us))
    v_trg_unshaped = nestedview(similar(flatview(v_src_us), R, totalndof(vs_trg), n))
    @assert axes(v_trg_unshaped) == axes(v_src)
    @assert v_trg_unshaped isa ArrayOfSimilarArrays
    @threads for i in eachindex(v_trg_unshaped, v_src)
        v = trafo_us(v_src_us[i])
        v_trg_unshaped[i] = v
    end
    vs_trg.(v_trg_unshaped)
end


function broadcast_trafo(
    ::typeof(identity),
    s_src::DensitySampleVector
)
    s_src
end

function broadcast_trafo(
    trafo::Any,
    s_src::DensitySampleVector
)
    vs_trg = trafo(elshape(s_src.v))
    R = eltype(unshaped(trafo(first(s_src.v)), vs_trg))
    s_src_us = unshaped.(s_src)
    trafo_us = unshaped(trafo)

    n = length(eachindex(s_src_us))
    s_trg_unshaped = DensitySampleVector((
        nestedview(similar(flatview(s_src_us.v), R, totalndof(vs_trg), n)),
        zero(s_src_us.logd),
        deepcopy(s_src_us.weight),
        deepcopy(s_src_us.info),
        deepcopy(s_src_us.aux),
    ))
    @assert axes(s_trg_unshaped) == axes(s_src)
    @assert s_trg_unshaped.v isa ArrayOfSimilarArrays
    @threads for i in eachindex(s_trg_unshaped, s_src)
        v, ladj = with_logabsdet_jacobian(trafo_us, s_src_us.v[i])
        s_trg_unshaped.v[i] = v
        s_trg_unshaped.logd[i] = s_src_us.logd[i] - ladj
    end
    vs_trg.(s_trg_unshaped)
end


function broadcast_arbitrary_trafo(
    trafo::Any,
    smpls::DensitySampleVector
)
    dummy_x = first(smpls.v)
    dummy_y = trafo(dummy_x)
    y_shape = valshape(dummy_y)
    unshaped_Y = VectorOfSimilarVectors(FunctionChain((trafo, inverse(y_shape))).(smpls.v))
    Y = y_shape.(unshaped_Y)
    DensitySampleVector((
        Y,
        Fill(NaN, length(eachindex(smpls.logd))),
        deepcopy(smpls.weight),
        deepcopy(smpls.info),
        deepcopy(smpls.aux),
    ))
end
