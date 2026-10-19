# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieAPIBoundaries
using BAT, Makie, Test
const EXT=Base.get_extension(BAT,:BATMakieExt)
const RECIPES=(upper=Hist2D,diagonal=Hist1D,lower=Hist2D)
fixture(;weights=ones(Int,5))=DensitySampleVector(
    v=[[i+0.1d for d in 1:4] for i in eachindex(weights)],
    logd=zeros(length(weights)),weight=weights)
graph(s;timing=nothing)=first(EXT._setup_static_graph(s,RECIPES,[1,2],2,
    BAT._default_makie_triagonal_config(),BAT._default_makie_diagonal_config();step_intervals=timing))
cell(fig,i,j)=only(filter(x->x isa Axis,contents(fig[1,1][i,j])))

@testset "Makie public boundaries and static ownership" begin
    @testset "Applied variable selection seeds picker" begin
        fig=bat_makie_plot(fixture(),RECIPES,[1,2,3],2;display=false)
        try
            checkboxes=filter(x->x isa Checkbox,fig.content)
            diagonal(i)=checkboxes[i*(i+1)÷2]
            @test [diagonal(i).checked[] for i in 1:4]==[true,true,false,false]
            diagonal(1).checked[]=false
            diagonal(4).checked[]=true
            @test [diagonal(i).checked[] for i in 1:4]==[false,true,false,true]
            @test occursin("v_2",String(cell(fig,1,1).xlabel[]))
            @test occursin("v_4",String(cell(fig,2,2).xlabel[]))
        finally
            empty!(fig)
        end
    end

    @testset "Static control subscriptions follow figure lifetime" begin
        g=graph(fixture())
        built=EXT._build_fig(g,EXT._init_gridlayout(g,2))
        max_step=Makie.ComputePipeline.get_observable!(g[:max_step];use_deepcopy=false)
        window=Makie.ComputePipeline.get_observable!(g[:window_steps];use_deepcopy=false)
        controls=built.range_controls[]
        @test !isempty(Makie.Observables.listeners(max_step))
        @test !isempty(Makie.Observables.listeners(window))
        empty!(built.fig)
        @test isempty(Makie.Observables.listeners(max_step))
        @test isempty(Makie.Observables.listeners(window))
        controls.draft[]=(Int64(2),Int64(3))
        Makie.notify(controls.apply.clicks)
        @test isnothing(g[:window_steps][])
        @test isempty(built.fig.content)
    end

    @testset "Generic positive integer grid sizes" begin
        s=fixture()
        for n in (Int32(2),big(2))
            fig=bat_makie_plot(s,RECIPES,[1,2],n;display=false)
            try
                @test length(filter(x->x isa Axis,fig.content))==4
            finally
                empty!(fig)
            end
            spec=Makie.convert_arguments(Makie.Plot{Makie.plot},s;recipes=RECIPES,vsel=[1,2],N_max=n)
            @test spec isa Makie.GridLayoutSpec
        end
        for n in (0,-1)
            @test_throws ArgumentError bat_makie_plot(s,RECIPES,Int[],n;display=false)
            cfg=BATMakieVisualization()
            @test_throws ArgumentError BATMakieVisualization(cfg.recipes,Int[],n,cfg.n_batch,
                cfg.max_buffered,cfg.adaptive_batching,cfg.batch_growth_rate,cfg.poll_interval,
                cfg.dark,cfg.triagonal_config,cfg.diagonal_config)
        end
    end

    @testset "Optional heterogeneous metadata is not chain identity" begin
        original=fixture()
        metadata=Union{Nothing,Int}[nothing,1,nothing,1,nothing]
        s=DensitySampleVector(v=original.v,logd=original.logd,weight=original.weight,info=metadata)
        g=graph(s)
        @test !EXT._samples_have_chain_ids(s)
        @test isempty(g[:flat_chainids][])
        @test g[:flat_samples][].info==metadata
        fig=bat_makie_plot(s,RECIPES,[1,2],2;display=false)
        try
            @test length(filter(x->x isa Axis,fig.content))==4
        finally
            empty!(fig)
        end
    end

    @testset "Timed clipping rejects lossy repetition weights" begin
        for (T,power) in ((Float16,12),(Float32,25),(Float64,54))
            total=Int64(2)^power
            wanted=div(total,2)+1
            s=fixture(weights=T[total])
            g=graph(s;timing=[[(cycle=1,first=Int64(1),last=total)]])
            data=g[:window_data][]
            before=deepcopy(data.samples)
            @test_throws ArgumentError EXT._window_selection(data,(Int64(1),wanted))
            @test data.samples==before
            safe,_=EXT._window_selection(data,(Int64(1),wanted-1))
            @test only(safe.weight)==wanted-1
        end
    end
end
end
