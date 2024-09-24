# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct NoMCMCTempering <: MCMCTempering end

"""
temper_mcmc_target!!(tempering::AbstractMCMCTemperingInstance, target::BATMeasure, stepno::Integer)
"""
function temper_mcmc_target!! end

struct NoAbstractMCMCTemperingInstance <: AbstractMCMCTemperingInstance end

temper_mcmc_target!!(tempering::NoAbstractMCMCTemperingInstance, target::BATMeasure, stepno::Integer) = tempering, target

create_temperering_state(tempering::NoMCMCTempering, target::BATMeasure) = NoAbstractMCMCTemperingInstance()
create_temperering_state(tempering::NoMCMCTempering, mc_state::MCMCState) = create_temperering_state(tempering, mc_state.target)
