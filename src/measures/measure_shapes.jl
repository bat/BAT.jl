# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# ValueShapes support for MeasureBase measures. This is BAT's sample
# storage layer meeting MeasureBase, it is intended to move into a
# ValueShapes extension of MeasureBase - keep it self-contained.


# The variate shape of a measure follows from the structure of its variates:

ValueShapes.varshape(m::AbstractMeasure) = _varshape_of_value(testvalue(m))

ValueShapes.varshape(m::MeasureBase.Dirac) = ConstValueShape(m.x)
ValueShapes.varshape(m::MeasureBase.AbstractProductMeasure) = _product_varshape(marginals(m))

_product_varshape(mars::NamedTuple) = NamedTupleShape(map(varshape, mars))
_product_varshape(mars) = _varshape_of_value(testvalue(productmeasure(mars)))

_varshape_of_value(::Real) = ScalarShape{Real}()
_varshape_of_value(x::AbstractArray{<:Real}) = ArrayShape{Real}(size(x)...)
_varshape_of_value(x::NamedTuple) = NamedTupleShape(map(_varshape_of_value, x))
_varshape_of_value(x) = valshape(x)


"""
    ValueShapes.unshaped(m::MeasureBase.AbstractMeasure)

Returns the pushforward of `m` under `inverse(varshape(m))`, a measure
over flat real-valued vectors.
"""
ValueShapes.unshaped(m::AbstractMeasure) = _unshaped_measure(m, varshape(m))

function ValueShapes.unshaped(m::AbstractMeasure, vs::AbstractValueShape)
    varshape(m) <= vs || throw(ArgumentError("Shape of measure not compatible with given shape"))
    unshaped(m)
end

# Disambiguates against unshaped(x, ::ConstValueShape) of ValueShapes:
ValueShapes.unshaped(m::AbstractMeasure, vs::ConstValueShape) =
    invoke(unshaped, Tuple{AbstractMeasure,AbstractValueShape}, m, vs)

# Variates of such measures are flat real vectors already:
_unshaped_measure(m::AbstractMeasure, ::ArrayShape{<:Real,1}) = m
_unshaped_measure(m::AbstractMeasure, vs::AbstractValueShape) = pushfwd(inverse(vs), m)

# Shaping a measure over flat real-valued vectors:
(vs::AbstractValueShape)(m::AbstractMeasure) = pushfwd(vs, m)


# The ValueShapes distributions become their structural measure equivalents:

batmeasure(d::ConstValueDist) = MeasureBase.Dirac(d.value)
batmeasure(d::NamedTupleDist) = productmeasure(map(batmeasure, NamedTuple{keys(d)}(values(d))))


# Transports know their target measure, so the shape of their results
# doesn't have to be inferred from a result value (which would lose
# constant components):
ValueShapes.resultshape(f::TransportFunction, @nospecialize(vs::AbstractValueShape)) = varshape(f.ν)


# Shaping and unshaping change only the shape of the variates, not their
# values, so they don't decide where a pushforward lives:
_maps_to_uhc(::Base.Fix2{typeof(unshaped)}) = missing
_maps_to_uhc(::AbstractValueShape) = missing
