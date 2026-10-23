# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# Native counterpart of the original-coordinate live regression. Run with a
# working display and an external process-group deadline, as other GL checks.
using BAT, GLMakie, Makie, Distributions, ValueShapes, Random, Test
const EXT=Base.get_extension(BAT,:BATMakieExt)
realpath(dirname(dirname(pathof(BAT))))==realpath(joinpath(@__DIR__,"../../..")) ||
    error("Native display-schema test loaded a different BAT checkout")
GLMakie.activate!(visible=false)
Makie.inline!(false)

function configured()
    c=BATMakieVisualization(adaptive_batching=false)
    BATMakieVisualization((diagonal=Hist1D,upper=Hist2D,lower=Scatter2D),[3,1],2,
        100_000,200_000,false,c.batch_growth_rate,.01,c.dark,
        merge(c.triagonal_config,(nbins=(8,8),)),merge(c.diagonal_config,(nbins=8,));observer_timeout=60.0)
end
function algorithm(retained)
    TransformedMCMC(nchains=2,nsteps=16,strict=false,store_burnin=retained,
        init=MCMCChainPoolInit(nsteps_init=24),convergence=AssumeConvergence(),
        adaptive_transform=BAT.NoAdaptiveTransform(),transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=10,max_ncycles=1,nsteps_final=2))
end
@testset "Native live original-coordinate rendering" begin
    for retained in (false,true)
        target=Dirichlet([2.,3.,4.])
        plain_context=BATContext(rng=BAT.Philox4x((91,19)))
        Random.seed!(312)
        plain=BAT.evalmeasure(target,algorithm(retained),plain_context)
        vis=BATVisualizer(configured())
        context=BATContext(rng=BAT.Philox4x((91,19)),visualizer=vis)
        try
            Random.seed!(312)
            observed=BAT.evalmeasure(target,algorithm(retained),context)
            @test BAT.samplesof(observed)==BAT.samplesof(plain)
            @test BAT.get_rng(context)==BAT.get_rng(plain_context)
            status=BAT.visualizer_status(vis)
            @test status.outcome===:complete && status.data_complete && status.presentation_valid
            @test isnothing(status.error)
            fig=BAT.visualizer_figure(vis)
            @test fig isa Figure
            graph=EXT._live_graph(vis)
            @test graph[:idxs][]==[3,1] && vis.content.n_dof[]==3
            for selection in ([3,1],[2,3],[3,1])
                EXT._apply_vsel!(vis,selection)
                @test graph[:idxs][]==selection
                data=graph[:flat_samples][].v.data
                @test vec(graph[EXT.marg_symbol((1,1))][])==data[selection[1],:]
                @test all(x->isapprox(x,1),vec(sum(data,dims=1)))
                pixels=Makie.colorbuffer(fig)
                @test !isempty(pixels)
                @test length(unique(pixels))>16
            end
        finally
            BAT.dispose_visualizer!(vis)
            @test timedwait(() -> !BAT.visualizer_status(vis).cleanup_pending,15.0;pollint=.01)===:ok
        end
        @test BAT.visualizer_status(vis).history_released
    end
end
