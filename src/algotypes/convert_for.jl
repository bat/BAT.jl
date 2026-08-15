# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    convert_for(function, obj)

*Experimental feature, not part of stable public API.*

Convert `obj` into something that `function` can use.
"""
function convert_for end


"""
    batalgorithm(algorithm)

*Experimental feature, not part of stable public API.*

Map `algorithm` to its BAT equivalent.

Wraps backend configurations and third-party algorithms in the BAT
algorithm that uses them. Acts as the identity on BAT algorithms.
"""
batalgorithm(algorithm) = algorithm
export batalgorithm
