# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    struct ModeAsDefined <: AbstractModeEstimator

Get the mode as defined by the density, resp. the underlying distribution
(if available), via `StatsBase.mode`.

Constructors:

* ```$(FUNCTIONNAME)()```        
"""
struct ModeAsDefined <: AbstractModeEstimator end
export ModeAsDefined


function evalmeasure_impl(measure::BATMeasure, ::ModeAsDefined, ::BATContext)
    m_uneval = unevaluated(measure)
    # ToDo: Is this what we want in all cases?
    m_mode = StatsBase.mode(m_uneval)
    return EvalMeasureImplReturn(modes = [m_mode])
end


"""
    MaxDensitySearch <: AbstractModeEstimator

Constructors:

    MaxDensitySearch()

Estimate the mode as the variate with the highest posterior density value
within a given set of samples.
"""
struct MaxDensitySearch <: AbstractModeEstimator end
export MaxDensitySearch


function evalmeasure_impl(measure::BATMeasure, ::MaxDensitySearch, ::BATContext)
    smpls = samplesof(measure)
    v_mode, mode_idx = _get_mode(smpls)

    return EvalMeasureImplReturn(;
        modes = [v_mode],
        evalresult = (;mode_idx = mode_idx),
    )    
end
