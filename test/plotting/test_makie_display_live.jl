# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieDisplayLive
using BAT, Makie, Test, Distributions, ValueShapes, Random, Logging
include("makie_rng_testutils.jl")
const EXT = Base.get_extension(BAT, :BATMakieExt)

mutable struct Capture <: AbstractDisplay
    figures::Vector{Makie.Figure}
end
Base.displayable(::Capture, ::MIME"image/png") = true
Base.display(c::Capture, fig::Makie.Figure) = (push!(c.figures, fig); nothing)
function captured(f)
    c=Capture(Makie.Figure[])
    old=Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(c)
    try
        return with_logger(() -> f(c), NullLogger())
    finally
        popdisplay(c)
        Makie.inline!(old)
    end
end

function configuration(incremental)
    c=BATMakieVisualization(adaptive_batching=false)
    n=incremental ? 4 : 100_000
    BATMakieVisualization((diagonal=Hist1D,upper=Hist2D,lower=Scatter2D),
        [1,2],2,n,2n,false,c.batch_growth_rate,.005,c.dark,
        merge(c.triagonal_config,(nbins=(8,8),)),merge(c.diagonal_config,(nbins=8,));
        observer_timeout=60.0)
end
function algorithm(intent,retained;callback=BAT.nop_func,nwalkers=2)
    TransformedMCMC(nchains=2,nwalkers=nwalkers,nsteps=20,strict=false,
        store_burnin=retained,pretransform=intent,callback=callback,
        init=MCMCRetryInit(max_init_tries=3,nsteps_init=20,strict=false),convergence=AssumeConvergence(),
        adaptive_transform=BAT.NoAdaptiveTransform(),transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=10,max_ncycles=1,nsteps_final=2))
end
fields(s)=(v=deepcopy(s.v),logd=copy(s.logd),weight=copy(s.weight),info=deepcopy(s.info),aux=deepcopy(s.aux))
function numerical(chain)
    (;info=chain.info,stepno=chain.stepno,nattempts=copy(chain.nattempts),
        nsamples=copy(chain.nsamples),accepted=copy(chain.accepted),
        current=map(fields,chain.current),proposed=map(fields,chain.proposed),output=fields(chain.output),
        context_rng=copy(BAT.get_rng(chain.context)),
        walker_rngs=[copy(BAT.get_rng(ctx)) for ctx in chain.walker_genctxs])
end
function default_streams()
    state=copy(Random.default_rng())
    parent=rand(copy(state),UInt64,8)
    tasks=[Threads.@spawn rand(UInt64,8) for _ in 1:2]
    children=fetch.(tasks)
    return (;state,parent,children,after=rand(UInt64,8))
end
function cleanup(vis)
    BAT.dispose_visualizer!(vis)
    @test timedwait(() -> !BAT.visualizer_status(vis).cleanup_pending,10.0;pollint=.005)===:ok
    @test BAT.visualizer_status(vis).history_released
end

function run_case(target,intent,retained,incremental;live=false)
    vis=live ? BATVisualizer(configuration(incremental)) : BATVisualizer()
    context=BATContext(rng=BAT.Philox4x((781,21)),visualizer=vis)
    events=Any[]
    callback(::Val{:mcmc_burnin},states)=(push!(events,[numerical(s.chain_state) for s in states]);nothing)
    alg=algorithm(intent,retained;callback)
    Random.seed!(71)
    result=BAT.evalmeasure(target,alg,context)
    streams=default_streams()
    chains=numerical.(BAT.samplegenof(result).chain_states)
    return (;vis,result,events,chains,streams,rng=copy(BAT.get_rng(context)))
end

@testset "Live original-coordinate schemas preserve fixed-work sampling" begin
    @test _prepare_makie_rng_test!() === nothing
    dirichlet=Dirichlet([2.,3.,4.])
    nested=NamedTupleDist(a=Normal(10,2),sub=NamedTupleDist(p=dirichlet,c=[4.2,3.7]))
    shaped=NamedTupleDist(a=Normal(10,2),p=dirichlet,c=4.2)
    source=BAT.StandardMvUniform(3)
    composed=BAT.MeasureBase.pushfwd(BAT.DistributionTransform(shaped,source),batmeasure(source))
    cases=NamedTuple[(name="Dirichlet retained=$r incremental=$i",target=dirichlet,intent=NormalBased(),retained=r,incremental=i,simplex=1:3)
        for r in (false,true) for i in (false,true)]
    append!(cases,[(name="Nested retained=$r",target=nested,intent=NormalBased(),retained=r,incremental=true,simplex=2:4) for r in (false,true)])
    push!(cases,(name="Shape-only constants",target=shaped,intent=DoNotTransform(),retained=false,incremental=true,simplex=2:4))
    push!(cases,(name="Composed constants",target=composed,intent=NormalBased(),retained=true,incremental=false,simplex=2:4))
    for case in cases
        @testset "$(case.name)" begin
            captured() do capture
                plain=run_case(case.target,case.intent,case.retained,case.incremental)
                repeat=run_case(case.target,case.intent,case.retained,case.incremental)
                @test isequal(fields(BAT.samplesof(repeat.result)),fields(BAT.samplesof(plain.result)))
                @test isequal(repeat.events,plain.events) && isequal(repeat.chains,plain.chains)
                @test repeat.rng==plain.rng && repeat.streams==plain.streams
                observed=nothing
                try
                    observed=run_case(case.target,case.intent,case.retained,case.incremental;live=true)
                    @test isequal(fields(BAT.samplesof(observed.result)),fields(BAT.samplesof(plain.result)))
                    @test isequal(observed.events,plain.events)
                    @test isempty(observed.events)==!case.retained
                    @test isequal(observed.chains,plain.chains)
                    @test observed.rng==plain.rng
                    @test rand(copy(observed.rng),UInt64,16)==rand(copy(plain.rng),UInt64,16)
                    @test observed.streams==plain.streams
                    status=BAT.visualizer_status(observed.vis)
                    @test status.outcome===:complete && status.data_complete && status.presentation_valid
                    @test status.quiesced && !status.cleanup_pending && isnothing(status.error)
                    @test status.last_publication >= (case.incremental ? 2 : 1)
                    @test length(capture.figures)==1
                    @test BAT.visualizer_figure(observed.vis)===only(capture.figures)
                    graph=EXT._live_graph(observed.vis)
                    d=totalndof(varshape(case.target))
                    @test observed.vis.content.n_dof[]==d
                    @test length(graph[:domain_lo][])==length(graph[:domain_hi][])==d
                    histories=graph[:samples][]
                    @test length(histories)==2 && all(c->length(c)==2,histories)
                    @test all(w->totalndof(varshape(w))==d,Iterators.flatten(histories))
                    for walker in Iterators.flatten(histories)
                        @test !isempty(walker)
                        for x in walker.v
                            @test sum(x[case.simplex])≈1
                            @test all(0 .< x[case.simplex] .< 1)
                        end
                        if case.target===dirichlet
                            @test walker.logd≈logpdf.(Ref(dirichlet),walker.v)
                        end
                    end
                    # Repeated real variable selections must expose the third
                    # component, not clamp it to the two sampler coordinates.
                    before=deepcopy([fields(w) for w in Iterators.flatten(histories)])
                    for selection in ([3,1],[2,3],[3,1])
                        EXT._apply_vsel!(observed.vis,selection)
                        @test graph[:idxs][]==selection
                        matrix=graph[:flat_samples][].v.data
                        @test size(matrix,1)==d
                        @test matrix[3,:]==reduce(vcat,[w.v.data[3,:] for w in Iterators.flatten(histories)])
                        @test vec(graph[EXT.marg_symbol((1,1))][])==matrix[selection[1],:]
                    end
                    @test isequal([fields(w) for w in Iterators.flatten(histories)],before)
                    timing=graph[:sample_timing][]
                    @test !isnothing(timing)
                    if !isnothing(timing)
                        actual_mass=Dict{Tuple,Int}()
                        for i in eachindex(histories),j in eachindex(histories[i])
                            w=histories[i][j]
                            @test sum(w.weight)==sum(s.last-s.first+1 for s in timing[i][j])
                            # Returned samples exclude burn-in when requested;
                            # explicit spans identify the corresponding main
                            # cycle even when rows merged across its boundary.
                            cycle=observed.chains[i].info.cycle
                            @test sum(s.last-s.first+1 for s in timing[i][j] if s.cycle==cycle-1) == 2
                            for span in timing[i][j]
                                (case.retained || span.cycle==cycle) || continue
                                info=w.info[span.row]
                                key=(info.chainid,info.walkerid,Tuple(w.v[span.row]))
                                actual_mass[key]=get(actual_mass,key,0)+span.last-span.first+1
                            end
                        end
                        expected_mass=Dict{Tuple,Int}()
                        for sample in BAT.samplesof(plain.result)
                            key=(sample.info.chainid,sample.info.walkerid,
                                Tuple(unshaped(sample.v,varshape(case.target))))
                            expected_mass[key]=get(expected_mass,key,0)+sample.weight
                        end
                        @test actual_mass==expected_mass
                    end
                finally
                    isnothing(observed) || cleanup(observed.vis)
                end
            end
        end
    end
end

# A direct empty initialization also exercises the documented targetless
# fallback and a late first publication from an initially idle chain/walker.
@testset "Empty targetless startup and idle slots" begin
    target=Dirichlet([2.,3.,4.])
    vis=BATVisualizer(configuration(false))
    ctx=BATContext(rng=BAT.Philox4x((431,12)),visualizer=vis)
    sampler,f=BAT.transform_and_unshape(NormalBased(),target,ctx)
    alg=algorithm(NormalBased(),false)
    states=BAT.MCMCState[BAT.MCMCState(alg,sampler,Int32(i),[zeros(2),ones(2)],ctx) for i in 1:2]
    outputs=BAT._empty_chain_outputs.(states)
    rng_before=copy(BAT.get_rng(ctx))
    captured() do capture
        try
            BAT.init_visualizer!(vis;rng=BAT.get_rng(ctx),mcmc_states=states,outputs,f_pretransform=f)
            @test BAT.get_rng(ctx)==rng_before
            graph=EXT._live_graph(vis)
            @test vis.content.n_dof[]==3
            @test all(isempty,Iterators.flatten(graph[:samples][]))
            @test all(w->totalndof(varshape(w))==3,Iterators.flatten(graph[:samples][]))
            @test all(w->totalndof(varshape(w))==2,Iterators.flatten(vis.content.output_buffer))
            @test all(0 .< graph[:domain_lo][] .< graph[:domain_hi][] .< 1)
            BAT.mcmc_iterate!!(outputs[1],states[1];max_nsteps=8,update_visualizer=true)
            BAT.finish_visualizer!(vis;success=true)
            @test BAT.visualizer_status(vis).outcome===:complete
            @test all(!isempty,graph[:samples][][1])
            @test all(isempty,graph[:samples][][2])
            @test all(w->totalndof(varshape(w))==3,graph[:samples][][2])
        finally
            cleanup(vis)
        end
    end
end

end
