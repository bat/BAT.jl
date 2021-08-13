# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ForwardDiffAD

*Experimental feature, not part of stable public API.*

Constructors:

* ```ForwardDiffAD()```
"""
struct ForwardDiffAD <: DifferentiationAlgorithm end
export ForwardDiffAD


function unshaped_gradient!(grad_f_x::AbstractVector{<:Real}, f::Function, x::AbstractVector{<:Real}, diffalg::ForwardDiffAD)
    R = promote_type(eltype(x), Float64) # ToDo: Don't use fixed type Float64

    result = DiffResults.MutableDiffResult(zero(R), (grad_f_x,))

    # chunk = ForwardDiff.Chunk(x)
    # config = ForwardDiff.GradientConfig(f, x, chunk)
    # ForwardDiff.gradient!(result, f, x, config)
    ForwardDiff.gradient!(result, f, x)

    @assert DiffResults.gradient(result) === grad_f_x
    convert(R, DiffResults.value(result))::R
end
