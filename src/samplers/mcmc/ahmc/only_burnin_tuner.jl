mutable struct JustBurninTuner{
    S<:MCMCBasicStats
} <: AbstractMCMCTuner
    stats::S
end

function JustBurninTuner(
    chain::MCMCIterator
)
    JustBurninTuner(MCMCBasicStats(chain))
end


struct JustBurninTunerConfig <: BAT.AbstractMCMCTuningStrategy end

(config::JustBurninTunerConfig)(chain::MCMCIterator; kwargs...) = JustBurninTuner(chain)
