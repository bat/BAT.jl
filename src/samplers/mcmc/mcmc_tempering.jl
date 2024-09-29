# This file is a part of BAT.jl, licensed under the MIT License (MIT).

struct NoMCMCTempering <: MCMCTempering end

"""
temper_mcmc_target!!(tempering::TemperingState, target::BATMeasure, stepno::Integer)
"""
function temper_mcmc_target!! end

struct MCMCNoTemperingState <: TemperingState end

temper_mcmc_target!!(tempering::MCMCNoTemperingState, target::BATMeasure, stepno::Integer) = tempering, target

create_temperering_state(tempering::NoMCMCTempering, target::BATMeasure) = MCMCNoTemperingState()
create_temperering_state(tempering::NoMCMCTempering, mc_state::MCMCChainState) = create_temperering_state(tempering, mc_state.target)
