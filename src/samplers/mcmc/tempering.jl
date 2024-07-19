abstract type TransformedMCMCTempering end
struct TransformedNoTransformedMCMCTempering <: TransformedMCMCTempering end

"""
    temper_mcmc_target!!(tempering::TransformedMCMCTemperingInstance, μ::BATMeasure, stepno::Integer)
"""
function temper_mcmc_target!! end



abstract type TransformedMCMCTemperingInstance end

struct NoTransformedMCMCTemperingInstance <: TransformedMCMCTemperingInstance end

temper_mcmc_target!!(tempering::NoTransformedMCMCTemperingInstance, μ::BATMeasure, stepno::Integer) = tempering, μ

get_temperer(tempering::TransformedNoTransformedMCMCTempering, density::BATMeasure) = NoTransformedMCMCTemperingInstance()
get_temperer(tempering::TransformedNoTransformedMCMCTempering, chain::MCMCIterator) = get_temperer(tempering, chain.μ)
