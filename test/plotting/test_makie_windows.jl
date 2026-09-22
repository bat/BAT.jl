# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieWindows
using BAT, Test, Makie, Distributions, LinearAlgebra, Random
const EXT = Base.get_extension(BAT, :BATMakieExt)
const RECIPES = (upper=Hist2D, diagonal=Hist1D, lower=Hist2D)
function fixture(; steps=[1,101], weights=[100,1], cycles=ones(Int,length(steps)), chains=ones(Int,length(steps)), walkers=ones(Int,length(steps)))
    n = length(steps)
    samples = DensitySampleVector(v=[[Float64(i),Float64(i^2)] for i in 1:n], logd=zeros(n), weight=weights,
        info=[BAT.MCMCSampleID(Int32(chains[i]),Int32(walkers[i]),Int32(cycles[i]),Int64(steps[i]),Int32(1),true) for i in 1:n])
    intervals = [[(cycle=cycles[i], first=steps[i], last=steps[i]+weights[i]-1)] for i in 1:n]
    return samples, intervals
end
function graph(samples; timing=nothing)
    first(EXT._setup_static_graph(samples, RECIPES, [1,2], 2,
        BAT._default_makie_triagonal_config(), BAT._default_makie_diagonal_config(); step_intervals=timing))
end

struct FailingInit <: BAT.MCMCInitAlgorithm end
const INIT_FAILURE = ErrorException("Intentional initialization failure after timing capture")
function BAT.mcmc_init!(alg::TransformedMCMC, target::BAT.BATMeasure, ::FailingInit, callback::Function, ctx::BATContext)
    state = BAT.MCMCState(alg,target,Int32(1),[zeros(2)],ctx)
    BAT.next_cycle!(state)
    BAT.mcmc_iterate!!(BAT._empty_chain_outputs(state),state;max_nsteps=2,record_visualizer_init=true)
    @test !isempty(ctx.visualizer.content.init_segments)
    throw(INIT_FAILURE)
end

@testset "Cycle-aware sample windows" begin
    @testset "Review examples and ownership" begin
        for (samples, timing) in (fixture(), fixture(steps=[1,2,1,2],weights=ones(Int,4),cycles=[1,1,2,2]))
            before = deepcopy(samples)
            g = graph(samples; timing)
            for (lo,hi) in ((1,typemax(Int)),(2,2),(100,101),(3,4),(1000,1001))
                Makie.update!(g; window_steps=(lo,hi))
                actual = g[:flat_samples][]
                expected = Dict{Float64,Int}()
                for row in eachindex(samples)
                    s = only(timing[row])
                    mass = max(0,min(s.last,hi)-max(s.first,lo)+1)
                    mass > 0 && (expected[samples.v[row][1]] = mass)
                end
                @test Dict(actual.v[i][1]=>actual.weight[i] for i in eachindex(actual)) == expected
                @test samples == before
            end
            @test g[:max_step][] == maximum(only(t).last for t in timing)
            Makie.update!(g; window_steps=nothing)
            @test g[:flat_samples][] == before
        end
    end
    @testset "Merged rows, large endpoints, and validation" begin
        s,_ = fixture(steps=[0],weights=[6])
        intervals = [[(cycle=1,first=1,last=3),(cycle=2,first=1,last=3)]]
        g = graph(s; timing=intervals)
        Makie.update!(g; window_steps=(2,2))
        @test g[:flat_weights][] == [2]
        @test length(g[:window_segments][]) == 2
        @test s.weight == [6]
        limit = typemax(Int64)
        bigs,_ = fixture(steps=[1],weights=[3])
        bg = graph(bigs; timing=[[(cycle=1,first=limit-2,last=limit)]])
        Makie.update!(bg; window_steps=(limit-1,limit))
        @test bg[:flat_weights][] == [2]
        @test bg[:max_step][] == limit
        @test_throws ArgumentError graph(s; timing=[])
        @test_throws ArgumentError graph(s; timing=[[(cycle=1,first=0,last=5)]])
        @test_throws ArgumentError graph(s; timing=[[(cycle=1,first=1,last=5)]])
        @test_throws ArgumentError graph(s; timing=[[(cycle=1,first=1,last=3),(cycle=1,first=3,last=5)]])
        @test_throws ArgumentError graph(s; timing=[[(cycle=1,first=1.5,last=6)]])
        @test_throws ArgumentError graph(s; timing=[[(cycle=big(typemax(Int32))+1,first=1,last=6)]])
        @test_throws ArgumentError graph(s; timing=[[(cycle=1,first=1,last=big(typemax(Int64))+1)]])
        @test_throws ArgumentError graph(s; timing=[[(first=1,last=6)]])
        # Source sample IDs are never rewritten, even for clipped timing.
        @test only(s.info).stepno == 0
    end
    @testset "Row fallback preserves arbitrary statistical mass" begin
        s,_ = fixture(steps=[1,1,1], weights=[1,1,1])
        s = DensitySampleVector(v=s.v,logd=s.logd,weight=[0.25,1.75,0.0],info=s.info)
        g = graph(s)
        @test !g[:has_step_timing][]
        @test g[:max_step][] == 3
        Makie.update!(g; window_steps=(2,3), show_trace_upper=true)
        @test g[:flat_weights][] == [1.75,0.0]
        @test isempty(g[:trace_groups][])
        @test s.weight == [0.25,1.75,0.0]
    end
    @testset "Indexed query against an expanded reference" begin
        rng = MersenneTwister(822)
        for _ in 1:40
            weights = rand(rng,1:8,12)
            cycles = repeat(1:3; inner=4)
            steps = vcat(([1; 1 .+ cumsum(weights[i:i+2])] for i in (1,5,9))...)
            s,timing = fixture(; steps,weights,cycles)
            g = graph(s; timing)
            for _ in 1:12
                lo,hi = sort(rand(rng,1:40,2))
                Makie.update!(g; window_steps=(lo,hi))
                actual = g[:flat_samples][]
                expected = Dict{Float64,Int}()
                for row in eachindex(s)
                    count = sum(lo <= t <= hi for t in steps[row]:steps[row]+weights[row]-1)
                    count > 0 && (expected[s.v[row][1]] = count)
                end
                @test Dict(actual.v[i][1]=>actual.weight[i] for i in eachindex(actual)) == expected
            end
        end
    end
    @testset "Trace cycle, clipping and gap semantics" begin
        s,timing = fixture(steps=[1,101,1,5],weights=[100,1,2,2],cycles=[1,1,2,2])
        g = graph(s; timing)
        Makie.update!(g; show_trace_upper=true)
        groups = g[:trace_groups][]
        @test length(groups) == 2 # cycle 2 only, split at its gap
        @test vcat(getproperty.(groups,:rows)...) == [3,4]
        @test last(groups).recency[end] == 1.0
        @test g[:max_step][] == 101 # older cycle is longer
        Makie.update!(g; window_steps=(100,101))
        @test only(g[:trace_groups][]).rows == [1,2]
        @test g[:flat_weights][] == [1,1]
        Makie.update!(g; show_trace_upper=false)
        @test isempty(g[:trace_groups][])
    end
    @testset "Windowed chain colors agree with traces" begin
        s,timing = fixture(steps=[1,2],weights=[1,1],chains=[10,20])
        g = graph(s;timing)
        Makie.update!(g;upper_recipe=ChainScatter2D,show_trace_upper=true,window_steps=(2,2))
        scatter = g[EXT.primitive_symbol(ChainScatter2D(),(2,1))][]
        trace = g[EXT.primitive_symbol(Trace2D(),(2,1))][]
        @test scatter.all_chainids == trace.all_chainids == [10,20]
        @test scatter.chainids == trace.chainids == [20]
        spec = only(EXT.compose_plotspecs(scatter,ChainScatter2D(),BAT._default_makie_triagonal_config()))
        @test spec.kwargs[:color] == EXT._CHAIN_COLOR_PALETTE[2]
    end
    @testset "Real initialization cursors survive cycle merges and resets" begin
        alg = TransformedMCMC(nchains=1,nwalkers=2,store_burnin=true,
            adaptive_transform=BAT.NoAdaptiveTransform(),transform_tuning=NoMCMCTransformTuning())
        vis = BATVisualizer(BATMakieVisualization())
        BAT.prepare_visualizer_init!(vis,alg)
        target = BAT.BATDistMeasure(MvNormal(zeros(2),1e-24 .* Matrix{Float64}(I,2,2)))
        state = BAT.MCMCState(alg,target,Int32(7),[zeros(2),zeros(2)],BATContext(rng=BAT.Philox4x((1,71)),visualizer=vis))
        outputs = BAT._empty_chain_outputs(state)
        for cycle in 1:2
            BAT.next_cycle!(state)
            for n in (2,3)
                state = BAT.mcmc_iterate!!(outputs,state;max_nsteps=n,record_visualizer_init=true)
            end
        end
        @test length.(outputs) == [1,1]
        @test only(outputs[1].weight) == 10
        spans = EXT._initial_step_spans(vis,state,outputs)
        @test !isnothing(spans)
        @test [(s.cycle,s.first,s.last) for s in spans[1]] == [(1,1,5),(2,1,5)]
        @test spans[1][1].row == spans[1][2].row == 1
        BAT.reset_visualizer_init!(vis,state.chain_state,[1])
        empty!(outputs[1])
        state = BAT.mcmc_iterate!!(outputs,state;max_nsteps=2,record_visualizer_init=true)
        spans = EXT._initial_step_spans(vis,state,outputs)
        @test [(s.first,s.last) for s in spans[1]] == [(6,7)]
        @test sum(s.last-s.first+1 for s in spans[2]) == 12
        state.chain_state.current.x.weight .= 1
        @test isnothing(EXT._initial_step_spans(vis,state,BAT._empty_chain_outputs(state)))
        BAT.clear_visualizer_init!(vis)
        @test isempty(vis.content.init_segments)
        @test !BAT.capture_visualizer_init(vis)
        @test !BAT.capture_visualizer_init(BATVisualizer())
    end
    @testset "Apply/All controls do not recompute on draft changes" begin
        s,timing = fixture(steps=[1,11], weights=[10,10])
        g = graph(s; timing)
        built = EXT._build_fig(g,EXT._init_gridlayout(g,2),nothing;has_chain_info=true,has_trace_info=true)
        controls = built.range_controls[]
        computations = Ref(0)
        observer = on(g[:flat_samples]) do _
            computations[] += 1
        end
        try
            for i in 2:9
                controls.slider.interval[] = (i,15)
            end
            @test computations[] == 0
            @test isnothing(g[:window_steps][])
            Makie.notify(controls.apply.clicks)
            @test computations[] == 1
            @test g[:window_steps][] == (9,15)
            @test sum(g[:flat_weights][]) == 7
            @test !controls.dirty[]
            Makie.notify(controls.apply.clicks)
            @test computations[] == 1
            Makie.notify(controls.all.clicks)
            @test computations[] == 2
            @test isnothing(g[:window_steps][])
            @test sum(g[:flat_weights][]) == 20
            @test !EXT._apply_sample_window!(g,nothing)
            @test_throws ArgumentError EXT._apply_sample_window!(g,(2,1))
        finally
            off(observer)
        end
    end
    @testset "Initialization failure discards timing before UI startup" begin
        alg = TransformedMCMC(nchains=1,store_burnin=true,init=FailingInit(),convergence=AssumeConvergence(),
            adaptive_transform=BAT.NoAdaptiveTransform(),transform_tuning=NoMCMCTransformTuning())
        vis = BATVisualizer(BATMakieVisualization())
        caught = try
            BAT.evalmeasure(MvNormal(zeros(2),ones(2)),alg,BATContext(rng=BAT.Philox4x((1,71)),visualizer=vis))
            nothing
        catch err
            err
        end
        @test caught === INIT_FAILURE
        @test isempty(vis.content.init_segments)
        @test isempty(vis.content.invalid_init)
        @test !BAT.capture_visualizer_init(vis)
        @test isnothing(vis.content.listener_task[])
        @test isempty(vis.content.chain_ids)
    end
    @testset "Bounded slider work and exact large integer entry" begin
        limit = typemax(Int64)
        grid = EXT.SampleRangeGrid(limit)
        @test length(grid) == 10_001
        @test first(grid) == 1
        @test last(grid) == limit
        @test issorted(grid)
        @test all(diff(collect(grid)) .> 0)
        @test collect(EXT.SampleRangeGrid(20)) == 1:20
        @test EXT._parse_sample_window(" $(limit-1) : $limit ",limit) == (limit-1,limit)
        for text in ("0:1","2:1","1:1.5","1:$(big(limit)+1)","1:2:3",nothing)
            @test isnothing(EXT._parse_sample_window(text,limit))
        end
        s,_ = fixture(steps=[1],weights=[3])
        g = graph(s;timing=[[(cycle=1,first=limit-2,last=limit)]])
        built = EXT._build_fig(g,EXT._init_gridlayout(g,2),nothing;has_chain_info=true,has_trace_info=true)
        controls = built.range_controls[]
        @test length(controls.slider.range[]) == 10_001
        Makie.set!(controls.value_entry,"$(limit-1):$limit")
        @test controls.draft[] == (limit-1,limit)
        @test isnothing(g[:window_steps][])
        Makie.notify(controls.apply.clicks)
        @test g[:window_steps][] == (limit-1,limit)
        @test g[:flat_weights][] == [2]
        Makie.notify(controls.all.clicks)
        @test g[:flat_weights][] == [3]
    end
end
end
