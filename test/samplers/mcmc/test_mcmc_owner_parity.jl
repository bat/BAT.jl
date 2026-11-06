# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMCMCOwnerParity
using BAT, Test, Random, Random123, Distributions, LinearAlgebra, Logging

# Frozen success path of the fe77d361 vector implementation. This deliberately
# calls the real scalar sampler without replacing any BAT method or selecting a
# production toggle. Its @sync/@spawn task topology is the compatibility oracle.
function legacy_group(outputs, states; update_visualizer=false, kwargs...)
    isempty(states) && return states
    outs = isnothing(outputs) ? fill(nothing, size(states)...) : outputs
    result = similar(states, BAT.MCMCState)
    cancelled = Base.Threads.Atomic{Bool}(false)
    @sync for i in eachindex(outs, states)
        Base.Threads.@spawn try
            result[i] = BAT.mcmc_iterate!!(outs[i], states[i];
                kwargs..., _cancelled=cancelled, update_visualizer=update_visualizer)
        catch
            cancelled[] = true
            rethrow()
        end
    end
    return result
end

# Preserve every numerical field, not object identity or compiler-generated
# closure names. All fixtures below use finite, acyclic numerical state.
function structural(x)
    x isa Union{Number,Symbol,AbstractString,Nothing} && return x
    x isa Type && return (:type, string(x))
    x isa AbstractArray && return (:array, size(x), map(structural, collect(vec(x))))
    x isa NamedTuple && return NamedTuple{keys(x)}(map(structural, values(x)))
    x isa Tuple && return map(structural, x)
    values = ntuple(i -> structural(getfield(x,i)), fieldcount(typeof(x)))
    return (x isa Function ? :function : nameof(typeof(x)), values)
end
structural(::Val{value}) where {value} = (:Val, value)

samples_snapshot(s) = (v=structural(s.v),weight=copy(s.weight),logd=copy(s.logd),
    info=structural(s.info),aux=structural(s.aux))
rng_snapshot(rng) = (state=structural(rng),future=rand(copy(rng),UInt64,16))

function chain_snapshot(chain)
    return (
        info=structural(chain.info), stepno=chain.stepno,
        nattempts=copy(chain.nattempts), nsamples=copy(chain.nsamples),
        accepted=copy(chain.accepted), walker_order=copy(chain.walker_order),
        current=map(samples_snapshot,chain.current), proposed=map(samples_snapshot,chain.proposed),
        output=samples_snapshot(chain.output), proposal=structural(chain.proposal),
        transform=structural(chain.f_transform), cycle_rng=structural(chain.rngpart_cycle),
        rng=rng_snapshot(BAT.get_rng(chain.context)),
        walkers=[rng_snapshot(BAT.get_rng(c)) for c in chain.walker_genctxs],
    )
end

state_snapshot(state) = (chain=chain_snapshot(state.chain_state),
    proposal_tuner=structural(state.proposal_tuner_state),
    transform_tuner=structural(state.trafo_tuner_state),temperer=structural(state.temperer_state))

function default_and_child_streams()
    parent = rand(copy(Random.default_rng()),UInt64,16)
    children = Vector{Vector{UInt64}}(undef,3)
    @sync for i in eachindex(children)
        Threads.@spawn children[i] = rand(UInt64,16)
    end
    return (parent=parent,children=children,parent_after=rand(UInt64,16))
end

const TARGET = MvNormal([0.5,-0.7],[1.3 0.2;0.2 0.8])

function algorithm(; nchains=2,nwalkers=1,nsteps=80,retained=false,arp=false,retry=false,ram=false,callback=BAT.nop_func)
    return TransformedMCMC(nchains=nchains,nwalkers=nwalkers,nsteps=nsteps,
        strict=false,store_burnin=retained,nonzero_weights=!arp,
        sample_weighting=arp ? ARPWeighting() : RepetitionWeighting(),
        pretransform=DoNotTransform(),
        adaptive_transform=ram ? BAT.TriangularAffineTransform() : BAT.NoAdaptiveTransform(),
        transform_tuning=ram ? RAMTuning() : NoMCMCTransformTuning(),
        init=retry ? MCMCRetryInit(nsteps_init=30) : MCMCChainPoolInit(nsteps_init=30),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=30,max_ncycles=1,nsteps_final=5),
        convergence=AssumeConvergence(),callback=callback)
end

# A real scalar MCMC step may replace a tuner with a new concrete state type.
# This test-local tuner has no numerical side effect and deterministically
# exercises that carrier-type change without relying on a random adaptation.
struct BeforeCommit <: BAT.MCMCTransformTunerState end
struct AfterCommit <: BAT.MCMCTransformTunerState end
function BAT.mcmc_tune_trafo_post_step!!(f::Function,::BeforeCommit,
    chain::BAT.MCMCChainState,::BAT.MCMCProposalState,::NamedTuple,::NamedTuple,::BAT.MCMCStepInfo)
    return f,AfterCommit(),chain
end

function initial_group(alg; widening=false)
    context=BATContext(rng=Philox4x((0x1502,0x7113)))
    target=BAT.batmeasure(TARGET)
    ids=1:alg.nchains
    partition=BAT.RNGPartition(BAT.get_rng(context),ids)
    states=BAT._gen_mcmc_states(alg,target,partition,ids,InitFromTarget(),context)
    if widening
        states=[BAT.MCMCState(s.chain_state,s.proposal_tuner_state,BeforeCommit(),s.temperer_state) for s in states]
    end
    BAT.next_cycle!.(states)
    BAT.mcmc_tuning_reinit!!.(states,alg.nsteps)
    return context,states
end

function group_result(run,alg; widening=false,seed=0x7ab1)
    context,states=initial_group(alg;widening)
    outputs=BAT._empty_chain_outputs.(states)
    initial_type=eltype(states)
    Random.seed!(seed)
    result=run(outputs,states;max_nsteps=alg.nsteps,nonzero_weights=alg.nonzero_weights)
    streams=default_and_child_streams()
    return (outputs=[[samples_snapshot(s) for s in walkers] for walkers in outputs],
        states=state_snapshot.(result),context=rng_snapshot(BAT.get_rng(context)),streams=streams,
        widened=any(s -> !(s isa initial_type),result))
end

# Also used by the separately archived baseline/candidate public-path probe.
const PUBLIC_CASES = (
    (name="one_chain_repetition",nchains=1,nwalkers=1,retained=false,arp=false,retry=true,ram=false),
    (name="pool_retained",nchains=2,nwalkers=1,retained=true,arp=false,retry=false,ram=false),
    (name="multiwalker_arp",nchains=2,nwalkers=2,retained=false,arp=true,retry=true,ram=false),
    (name="adaptive_retained",nchains=2,nwalkers=2,retained=true,arp=true,retry=true,ram=true),
)

function public_result(case; nsteps=80)
    events=NamedTuple[]
    owner=current_task()
    callback(event,states)=push!(events,(event=structural(event),owner=current_task()===owner,
        info=[structural(s.chain_state.info) for s in states],steps=[s.chain_state.stepno for s in states]))
    alg=algorithm(;nchains=case.nchains,nwalkers=case.nwalkers,nsteps=nsteps,
        retained=case.retained,arp=case.arp,retry=case.retry,ram=case.ram,callback=callback)
    context=BATContext(rng=Threefry4x((564,84,19,71)))
    Random.seed!(0x33ab)
    result=with_logger(NullLogger()) do
        BAT.evalmeasure(TARGET,alg,context)
    end
    streams=default_and_child_streams()
    return (samples=samples_snapshot(BAT.samplesof(result)),
        chains=chain_snapshot.(BAT.samplegenof(result).chain_states),
        context=rng_snapshot(BAT.get_rng(context)),events=events,streams=streams)
end

function run_tests()
    @testset "MCMC owner successful-run parity" begin
        for (label,kwargs) in (
            ("one chain",(nchains=1,nwalkers=1)),
            ("several chains and walkers",(nchains=3,nwalkers=2)),
            ("ARP rows including zero weights",(nchains=2,nwalkers=2,arp=true)),
            ("adaptive RAM",(nchains=2,nwalkers=2,ram=true)),
        )
            @testset "$label" begin
                alg=algorithm(;kwargs...)
                expected=group_result(legacy_group,alg)
                observed=group_result(BAT.mcmc_iterate!!,alg)
                @test isequal(observed.outputs,expected.outputs)
                @test isequal(observed.states,expected.states)
                @test isequal(observed.context,expected.context)
                @test isequal(observed.streams,expected.streams)
            end
        end
        @testset "result carrier can widen its tuner type" begin
            alg=algorithm(nchains=2,nwalkers=2)
            expected=group_result(legacy_group,alg;widening=true)
            observed=group_result(BAT.mcmc_iterate!!,alg;widening=true)
            @test expected.widened && observed.widened
            @test isequal(observed,expected)
        end
        @testset "public phase and callback repeatability" begin
            for case in PUBLIC_CASES
                first=public_result(case)
                second=public_result(case)
                @test isequal(first,second)
                @test length(first.events)==Int(case.retained)
                @test all(e -> e.event==(:Val,:mcmc_burnin) && e.owner &&
                    e.steps==fill(30,case.nchains),first.events)
                @test all(c -> c.stepno==80,first.chains)
            end
        end
    end
end

# Scratch comparison scripts include these helpers without rerunning the suite.
get(ENV,"BAT_MCMC_PARITY_HELPERS_ONLY","false")=="true" || run_tests()
end # module
