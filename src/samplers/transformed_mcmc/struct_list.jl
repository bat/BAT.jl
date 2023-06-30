



struct TransformedGelmanRubinConvergence <: ConvergenceTest
# Constructors:
# @with_kw struct TransformedGelmanRubinConvergence <: ConvergenceTest
    struct TransformedBrooksGelmanConvergence <: ConvergenceTest
# Constructors:
# @with_kw struct TransformedBrooksGelmanConvergence <: ConvergenceTest
struct TransformedMCMCNullStats <: TransformedAbstractMCMCStats end
struct TransformedMCMCBasicStats{L<:Real,P<:Real} <: TransformedAbstractMCMCStats
    struct TransformedRepetitionWeighting{T<:AbstractFloat} <: TransformedAbstractMCMCWeightingScheme{T}
# Constructors:
# struct TransformedRepetitionWeighting{T<:Real} <: TransformedAbstractMCMCWeightingScheme{T} end
# Constructors:
struct TransformedARPWeighting{T<:AbstractFloat} <: TransformedAbstractMCMCWeightingScheme{T} end
struct TransformedGenericProposalDist{D<:Distribution{Multivariate},SamplerF,S<:Sampleable} <: TransformedAbstractProposalDist
struct TransformedGenericUvProposalDist{D<:Distribution{Univariate},T<:Real,SamplerF,S<:Sampleable} <: TransformedAbstractProposalDist
struct TransformedMvTDistProposal <: TransformedProposalDistSpec
struct TransformedUvTDistProposalSpec <: TransformedProposalDistSpec
    struct TransformedMCMCMultiCycleBurnin <: TransformedMCMCBurninAlgorithm
# Constructors:
# @with_kw struct TransformedMCMCMultiCycleBurnin <: TransformedMCMCBurninAlgorithm
struct TransformedNoTransformedMCMCTempering <: TransformedMCMCTempering end
# struct NoTransformedTransformedMCMCTemperingInstance <: TransformedTransformedMCMCTemperingInstance end
    struct TransformedMCMCChainPoolInit <: TransformedMCMCInitAlgorithm
# Constructors:
# @with_kw struct TransformedMCMCChainPoolInit <: TransformedMCMCInitAlgorithm
# function _construct_chain(
# ) = [_construct_chain(rngpart, id, algorithm, density, initval_alg) for id in ids]
struct TransformedMCMCTransformedSampleID{
struct TransformedMCMCNoOpTuning <: TransformedMCMCTuningAlgorithm end
struct TransformedMCMCNoOpTuner <: TransformedAbstractMCMCTunerInstance end
@with_kw struct TransformedAdaptiveMHTuning <: TransformedMCMCTuningAlgorithm
mutable struct TransformedProposalCovTuner{
@with_kw struct TransformedRAMTuner <: TransformedMCMCTuningAlgorithm #TODO: rename to RAMTuning
@with_kw mutable struct TransformedRAMTunerInstance <: TransformedAbstractMCMCTunerInstance
@with_kw struct TransformedMCMCIteratorInfo
Constructors:
struct TransformedMCMCSampleGenerator{
mutable struct TransformedMCMCIterator{
struct TransformedMHProposal{
# TODO AC: find a better solution for this. Problem is that in the with_kw constructor below, we need to dispatch on this type.
struct TransformedMCMCDispatch end
@with_kw struct TransformedMCMCSampling{