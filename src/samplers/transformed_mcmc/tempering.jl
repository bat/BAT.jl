abstract type TransformedMCMCTempering end
struct TransformedNoTransformedMCMCTempering <: TransformedMCMCTempering end

"""
    temper_mcmc_target!!(tempering::TransformedTransformedMCMCTemperingInstance, μ::BATMeasure, stepno::Integer)
"""
function temper_mcmc_target!! end



abstract type TransformedTransformedMCMCTemperingInstance end

struct NoTransformedTransformedMCMCTemperingInstance <: TransformedTransformedMCMCTemperingInstance end

temper_mcmc_target!!(tempering::NoTransformedTransformedMCMCTemperingInstance, μ::BATMeasure, stepno::Integer) = tempering, μ

get_temperer(tempering::TransformedNoTransformedMCMCTempering, density::BATMeasure) = NoTransformedTransformedMCMCTemperingInstance()
get_temperer(tempering::TransformedNoTransformedMCMCTempering, chain::MCMCIterator) = get_temperer(tempering, chain.μ)
