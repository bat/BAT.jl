abstract type TransformedMCMCTempering end
struct TransformedNoMCMCTempering <: TransformedMCMCTempering end

"""
    temper_mcmc_target!!(tempering::MCMCTemperingInstance, μ::BATMeasure, stepno::Integer)
"""
function temper_mcmc_target!! end



abstract type TransformedMCMCTemperingInstance end

struct NoMCMCTemperingInstance <: TransformedMCMCTemperingInstance end

temper_mcmc_target!!(tempering::NoMCMCTemperingInstance, μ::BATMeasure, stepno::Integer) = tempering, μ

get_temperer(tempering::TransformedNoMCMCTempering, density::BATMeasure) = NoMCMCTemperingInstance()
get_temperer(tempering::TransformedNoMCMCTempering, chain::MCMCIterator) = get_temperer(tempering, chain.μ)
