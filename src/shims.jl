# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Workaround for Distributions.jl issue #647
_iszero(x) = iszero(x)
_iszero(::Distributions.ZeroVector) = true
