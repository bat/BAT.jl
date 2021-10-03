# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ZygoteAD

*Experimental feature, not part of stable public API.*

Constructors:

* ```ZygoteAD()```
"""
struct ZygoteAD <: DifferentiationAlgorithm end
export ZygoteAD


function unshaped_gradient!(grad_f_x::AbstractVector{<:Real}, f::Function, x::AbstractVector{<:Real}, diffalg::ZygoteAD)
    value, back = Zygote.pullback(f, x)
    grad_f_x[:] = first(back(Zygote.sensitivity(value)))
    return value
end
