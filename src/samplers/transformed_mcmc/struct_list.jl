



struct TransformedGelmanRubinConvergence <: ConvergenceTest
# Constructors:
# @with_kw struct TransformedGelmanRubinConvergence <: ConvergenceTest
    struct TransformedBrooksGelmanConvergence <: ConvergenceTest
# Constructors:
# @with_kw struct TransformedBrooksGelmanConvergence <: ConvergenceTest
struct TransformedMCMCNullStats <: AbstractMCMCStats end
struct TransformedMCMCBasicStats{L<:Real,P<:Real} <: AbstractMCMCStats
    struct TransformedRepetitionWeighting{T<:AbstractFloat} <: AbstractMCMCWeightingScheme{T}
# Constructors:
# struct TransformedRepetitionWeighting{T<:Real} <: AbstractMCMCWeightingScheme{T} end
# Constructors:
struct TransformedARPWeighting{T<:AbstractFloat} <: AbstractMCMCWeightingScheme{T} end
struct TransformedGenericProposalDist{D<:Distribution{Multivariate},SamplerF,S<:Sampleable} <: AbstractProposalDist
struct TransformedGenericUvProposalDist{D<:Distribution{Univariate},T<:Real,SamplerF,S<:Sampleable} <: AbstractProposalDist
struct TransformedMvTDistProposal <: ProposalDistSpec
struct TransformedUvTDistProposalSpec <: ProposalDistSpec
    struct TransformedMCMCMultiCycleBurnin <: MCMCBurninAlgorithm
# Constructors:
# @with_kw struct TransformedMCMCMultiCycleBurnin <: MCMCBurninAlgorithm
struct TransformedNoMCMCTempering <: MCMCTempering end
# struct NoMCMCTemperingInstance <: MCMCTemperingInstance end
    struct TransformedMCMCChainPoolInit <: MCMCInitAlgorithm
# Constructors:
# @with_kw struct TransformedMCMCChainPoolInit <: MCMCInitAlgorithm
# function _construct_chain(
# ) = [_construct_chain(rngpart, id, algorithm, density, initval_alg) for id in ids]
struct TransformedMCMCSampleID{
struct TransformedMCMCNoOpTuning <: MCMCTuningAlgorithm end
struct TransformedMCMCNoOpTuner <: AbstractMCMCTunerInstance end
@with_kw struct TransformedAdaptiveMHTuning <: MCMCTuningAlgorithm
mutable struct TransformedProposalCovTuner{
@with_kw struct TransformedRAMTuner <: MCMCTuningAlgorithm #TODO: rename to RAMTuning
@with_kw mutable struct TransformedRAMTunerInstance <: AbstractMCMCTunerInstance
@with_kw struct TransformedMCMCIteratorInfo
Constructors:
struct TransformedMCMCSampleGenerator{
mutable struct TransformedMCMCIterator{
struct TransformedMHProposal{
# TODO AC: find a better solution for this. Problem is that in the with_kw constructor below, we need to dispatch on this type.
struct TransformedMCMCDispatch end
@with_kw struct TransformedMCMCSampling{