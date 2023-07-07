# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    abstract type WhiteningAlgorithm

*BAT-internal, not part of stable public API.*

Abstract type for sample whitening algorithms.
"""
abstract type WhiteningAlgorithm end


"""
    struct NoWhitening <: WhiteningAlgorithm

*BAT-internal, not part of stable public API.*

No-op whitening transformation, leaves samples unchanged.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct NoWhitening <: WhiteningAlgorithm end



"""
    struct CholeskyWhitening <: WhiteningAlgorithm

*BAT-internal, not part of stable public API.*

Whitening transformation based on a Cholesky transformation of the empirical
sample covariance matrix.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct CholeskyWhitening <: WhiteningAlgorithm end



"""
    struct CholeskyPartialWhitening <: WhiteningAlgorithm

*BAT-internal, not part of stable public API.*

Whitening transformation based on a Cholesky transformation of the empirical
sample covariance matrix.

Only transforms dimensions (degrees of freedom) for which the marginalized
distribution asymptotically approaches zero in the positive and negative
direction.

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct CholeskyPartialWhitening <: WhiteningAlgorithm end


"""
    struct StatisticalWhitening <: WhiteningAlgorithm

*BAT-internal, not part of stable public API.*

Whitening transformation based statistical whitening.
CholeskyPartialWhitening

Constructors:

* ```$(FUNCTIONNAME)()```
"""
struct StatisticalWhitening <: WhiteningAlgorithm end



# ToDo: Move whitening code from AHMI here, add new public API
# function `bat_whiten`.
