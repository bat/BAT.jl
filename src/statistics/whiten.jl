# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    WhiteningAlgorithm

Abstract type for sample whitening algorithms.
"""
abstract type WhiteningAlgorithm end
export WhiteningAlgorithm


"""
NoWhitening

No-op whitening transformation, leaves samples unchanged.
"""
struct NoWhitening <: WhiteningAlgorithm end
export NoWhitening



"""
    CholeskyWhitening

Whitening transformation based on a Cholesky transformation of the empirical
sample covariance matrix.
"""
struct CholeskyWhitening <: WhiteningAlgorithm end
export CholeskyWhitening



"""
    CholeskyPartialWhitening

Whitening transformation based on a Cholesky transformation of the empirical
sample covariance matrix.

Only transforms dimensions (degrees of freedom) for which the marginalized
distribution aymptotically approaches zero in the positive and negative
direction.
"""
struct CholeskyPartialWhitening <: WhiteningAlgorithm end
export CholeskyPartialWhitening


"""
    StatisticalWhitening

Whitening transformation based statistical whitening.
"""
struct StatisticalWhitening <: WhiteningAlgorithm end
export StatisticalWhitening



# ToDo: Move whitening code from AHMI here, add new public API
# function `bat_whiten`. Remove `_amhi_whitening_func` and
# change AHMI to use `bat_whiten` directly.

_amhi_whitening_func(algorithm::NoWhitening) = no_whitening!
_amhi_whitening_func(algorithm::CholeskyWhitening) = cholesky_whitening!
_amhi_whitening_func(algorithm::CholeskyPartialWhitening) = cholesky_partial_whitening!
_amhi_whitening_func(algorithm::StatisticalWhitening) = statistical_whitening!
