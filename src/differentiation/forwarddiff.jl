# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ForwardDiffAD

*Experimental feature, not part of stable public API.*

Constructors:

* ```ForwardDiffAD()```
"""
struct ForwardDiffAD <: DifferentiationAlgorithm end
export ForwardDiffAD


struct ForwardDiffGradient{
    F<:Function,
    VS<:AbstractValueShape,
    GS<:AbstractValueShape,
} <: GradientFunction
    _unshaped_f::F
    _input_shape::VS
    _grad_shape::GS
end


function ForwardDiffGradient(f::Function)
    input_shape = varshape(f)
    unshaped_f = unshaped(f)
    grad_shape = gradient_shape(input_shape)
    ForwardDiffGradient(unshaped_f, input_shape, grad_shape)
end


function _run_forwarddiff!(grad_f_x::AbstractVector{<:Real}, ::Type{T}, f::Function, x::AbstractVector{<:Real}) where {T<:Real}
    result = DiffResults.MutableDiffResult(zero(T), (grad_f_x,))

    # chunk = ForwardDiff.Chunk(x)
    # config = ForwardDiff.GradientConfig(f, x, chunk)
    # ForwardDiff.gradient!(result, f, x, config)
    ForwardDiff.gradient!(result, f, x)

    @assert DiffResults.gradient(result) === grad_f_x
    DiffResults.value(result)
end


function (gf::ForwardDiffGradient)(v::Any)
    input_shape = gf._input_shape
    v_shaped = fixup_variate(input_shape, v)
    v_unshaped = unshaped(v_shaped)
    R = density_logval_type(v_unshaped, default_dlt())

    grad_f_unshaped = similar(v_unshaped)

    value = R(_run_forwarddiff!(grad_f_unshaped, R, gf._unshaped_f, v_unshaped))

    (value, gf._grad_shape(grad_f_unshaped))
end


function (gf::ForwardDiffGradient)(::typeof(!), grad_f::Any, v::Any)
    input_shape = gf._input_shape
    v_shaped = fixup_variate(input_shape, v)
    v_unshaped = unshaped(v_shaped)
    R = density_logval_type(v_unshaped, default_dlt())

    result = if isnothing(grad_f)
        R(gf._unshaped_f(v_unshaped))
    else
        grad_f_unshaped = unshaped(grad_f, gf._grad_shape)
        R(_run_forwarddiff!(grad_f_unshaped, R, gf._unshaped_f, v_unshaped))
    end
end


function bat_valgrad_impl(f::Function, algorithm::ForwardDiffAD)
    (result = ForwardDiffGradient(f),)
end
