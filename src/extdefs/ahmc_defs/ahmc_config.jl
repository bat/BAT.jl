# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Metric ==============================================

abstract type HMCMetric end

struct DiagEuclideanMetric <: HMCMetric end

struct UnitEuclideanMetric <: HMCMetric end

struct DenseEuclideanMetric <: HMCMetric end


# Tuning ==============================================

abstract type HMCTuning <: MCMCProposalTuning end

@with_kw struct MassMatrixAdaptor <: HMCTuning
    target_acceptance::Float64 = 0.8
end

@with_kw struct StepSizeAdaptor <: HMCTuning
    target_acceptance::Float64 = 0.8
end

@with_kw struct NaiveHMCTuning <: HMCTuning
    target_acceptance::Float64 = 0.8
end

# Uses Stan (also AdvancedHMC) defaults 
# (see https://mc-stan.org/docs/2_26/reference-manual/hmc-algorithm-parameters.html):
@with_kw struct StanLikeTuning <: MCMCTransformTuning
    "target acceptance rate"
    target_acceptance::Float64 = 0.8

    "width of initial fast adaptation interval"
    init_buffer::Int = 75

    "width of final fast adaptation interval"
    term_buffer::Int = 50

    "initial width of slow adaptation interval"
    window_size::Int = 25
end

export StanLikeTuning
