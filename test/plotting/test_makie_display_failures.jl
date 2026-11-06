# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieDisplayFailures
using BAT, Makie, Test, Distributions, ValueShapes, Random, Logging
const EXT=Base.get_extension(BAT,:BATMakieExt)
const FAILURE=ErrorException("intentional display schema transform failure")

mutable struct Capture <: AbstractDisplay
    count::Int
end
Base.displayable(::Capture,::MIME"image/png")=true
Base.display(c::Capture,::Makie.Figure)=(c.count+=1;nothing)
function captured(f)
    c=Capture(0); old=Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true); pushdisplay(c)
    try
        with_logger(() -> f(c),NullLogger())
    finally
        popdisplay(c); Makie.inline!(old)
    end
end
function config()
    c=BATMakieVisualization(adaptive_batching=false)
    BATMakieVisualization((diagonal=Hist1D,upper=Hist2D,lower=Scatter2D),[1,3],2,
        100_000,200_000,false,c.batch_growth_rate,.005,c.dark,
        merge(c.triagonal_config,(nbins=(8,8),)),merge(c.diagonal_config,(nbins=8,));observer_timeout=30.0)
end
function fixture()
    vis=BATVisualizer(config())
    ctx=BATContext(rng=BAT.Philox4x((273,94)),visualizer=vis)
    target=Dirichlet([2.,3.,4.])
    sampler,f=BAT.transform_and_unshape(NormalBased(),target,ctx)
    alg=TransformedMCMC(nchains=1,nwalkers=1,
        adaptive_transform=BAT.NoAdaptiveTransform(),transform_tuning=NoMCMCTransformTuning())
    state=BAT.MCMCState(alg,sampler,Int32(1),[zeros(2)],ctx)
    return (;vis,ctx,target=batmeasure(target),f,state,outputs=BAT._empty_chain_outputs(state))
end

mutable struct ControlledInverse{F} <: Function
    inner::F
    fail::Bool
    calls::Threads.Atomic{Int}
end
ControlledInverse(inner,fail,calls::Int)=ControlledInverse(inner,fail,Threads.Atomic{Int}(calls))
function (f::ControlledInverse)(x)::Vector{Float64}
    Threads.atomic_add!(f.calls,1)
    f.fail && throw(FAILURE)
    return Float64.(f.inner(x))
end
BAT.resultshape(f::ControlledInverse,shape::AbstractValueShape)=BAT.resultshape(f.inner,shape)
function BAT.with_logabsdet_jacobian(f::ControlledInverse,x)
    Threads.atomic_add!(f.calls,1)
    f.fail && throw(FAILURE)
    BAT.with_logabsdet_jacobian(f.inner,x)
end
struct ControlledForward{F} <: Function
    backward::F
end
BAT.inverse(f::ControlledForward)=f.backward

function caught(f)
    try f();nothing catch err;err end
end
function assert_released(vis)
    # Observe autonomous completion before any emergency cleanup or disposal.
    ctl=vis.content.lifecycle
    @test timedwait(() -> vis.content.graph[]===nothing && ctl.task===nothing &&
        ctl.timer===nothing && !ctl.initializing,10.0;pollint=.005)===:ok
    status=BAT.visualizer_status(vis)
    @test status.outcome===:failed && !status.data_complete && status.history_released
    @test status.quiesced && !status.cleanup_pending
end
function cleanup(vis)
    BAT.request_visualizer_stop!(vis;reason=:disabled)
    BAT.finish_visualizer!(vis;success=false)
    BAT.dispose_visualizer!(vis)
    @test timedwait(() -> !BAT.visualizer_status(vis).cleanup_pending,10.0;pollint=.005)===:ok
end

@testset "Empty prototype failure uses owned startup rollback" begin
    f=fixture()
    backward=ControlledInverse(BAT.inverse(f.f),true,0)
    transform=ControlledForward(backward)
    before=deepcopy(BAT.current_sample(f.state.chain_state))
    rng_before=copy(BAT.get_rng(f.ctx))
    captured() do capture
        try
            err=caught(() -> BAT.init_visualizer!(f.vis;rng=BAT.get_rng(f.ctx),
                mcmc_states=BAT.MCMCState[f.state],outputs=[f.outputs],target=f.target,f_pretransform=transform))
            @test err===FAILURE
            @test backward.calls[]==1
            @test capture.count==0
            @test BAT.current_sample(f.state.chain_state)==before
            @test BAT.get_rng(f.ctx)==rng_before
            assert_released(f.vis)
            @test BAT.visualizer_status(f.vis).error.phase===:startup
        finally
            cleanup(f.vis)
        end
    end
end

@testset "Original shape mismatch is explicit and releases startup" begin
    f=fixture()
    captured() do capture
        try
            err=caught(() -> BAT.init_visualizer!(f.vis;rng=BAT.get_rng(f.ctx),
                mcmc_states=BAT.MCMCState[f.state],outputs=[f.outputs],
                target=batmeasure(MvNormal(zeros(4),ones(4))),f_pretransform=f.f))
            @test err isa ArgumentError
            @test capture.count==0
            assert_released(f.vis)
        finally
            cleanup(f.vis)
        end
    end
end

@testset "A later inverse failure releases observation after sampling" begin
    f=fixture()
    backward=ControlledInverse(BAT.inverse(f.f),false,0)
    captured() do capture
        try
            BAT.init_visualizer!(f.vis;rng=BAT.get_rng(f.ctx),mcmc_states=BAT.MCMCState[f.state],
                outputs=[f.outputs],target=f.target,f_pretransform=ControlledForward(backward))
            @test capture.count==1
            @test all(isempty,Iterators.flatten(EXT._live_graph(f.vis)[:samples][]))
            @test f.vis.content.n_dof[]==3
            BAT.mcmc_iterate!!(f.outputs,f.state;max_nsteps=8,update_visualizer=true)
            @test !isempty(only(f.outputs))
            rng_before=copy(BAT.get_rng(f.ctx))
            samples_before=deepcopy(only(f.outputs))
            backward.fail=true
            calls_before=backward.calls[]
            BAT.finish_visualizer!(f.vis;success=true)
            assert_released(f.vis)
            @test BAT.get_rng(f.ctx)==rng_before
            @test only(f.outputs)==samples_before
            @test BAT.visualizer_status(f.vis).error.phase===:finalize
            # The threaded transform can wrap FAILURE in a task exception;
            # lifecycle diagnostics deliberately omit those exception details.
            # A call after arming this inverse necessarily throws FAILURE.
            @test backward.calls[]>calls_before
        finally
            cleanup(f.vis)
        end
    end
end

end
