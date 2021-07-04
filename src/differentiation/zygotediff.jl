# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ZygoteAD

*Experimental feature, not part of stable public API.*

Constructors:

* ```ZygoteAD()```
"""
struct ZygoteAD <: DifferentiationAlgorithm end
export ZygoteAD


function unshaped_gradient!(grad_f_x::AbstractVector{<:Real}, ::Type{R}, f::Function, x::AbstractVector{<:Real}, diffalg::ZygoteAD) where {R<:Real}
    value, back = Zygote.pullback(f, x)
    grad_f_x[:] = first(back(Zygote.sensitivity(value)))
    return convert(R, value)::R
end
