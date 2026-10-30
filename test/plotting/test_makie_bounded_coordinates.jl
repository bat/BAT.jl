# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieBoundedCoordinates

using BAT, Test, Makie, CairoMakie, Statistics, Distributions, Random, Logging

const E = Base.get_extension(BAT, :BATMakieExt)
const PAIR = BAT._default_makie_triagonal_config()
const DIAG = BAT._default_makie_diagonal_config()

# Exact represented-value ranks and moments; BigFloat is used only for roots and
# powers after cancellation/interpolation have already been resolved rationally.
function exact_quantile(values, weights, p, weighted)
    pairs=sort(collect(zip(Rational{BigInt}.(values),Rational{BigInt}.(weights))))
    n=length(pairs)
    if !weighted
        rank=1+p*(n-1);i=min(floor(Int,rank),n-1)
        return pairs[i][1]+(rank-i)*(pairs[i+1][1]-pairs[i][1])
    end
    target=p*sum(last,pairs)+(1-p)*last(first(pairs))
    prefix=0//BigInt(1)
    for i in eachindex(pairs)
        previous=prefix;prefix+=pairs[i][2]
        if prefix>target
            i==1 && return pairs[1][1]
            return pairs[i-1][1]+(target-previous)/pairs[i][2]*(pairs[i][1]-pairs[i-1][1])
        end
    end
    first(last(pairs))
end
function bandwidth_reference(values, weights; weighted=!allequal(weights))
    n=length(values);n==1 && return 0.9
    z=Rational{BigInt}.(values);z .-= first(z)
    w=Rational{BigInt}.(weights);total=sum(w)
    mean=weighted ? sum(w.*z)/total : sum(z)/n
    variance=weighted ? sum(w.*(z.-mean).^2)/total : sum(abs2,z.-mean)/(n-1)
    iqr=exact_quantile(values,weights,3//4,weighted)-exact_quantile(values,weights,1//4,weighted)
    count=weighted ? total^2/sum(abs2,w) : n
    setprecision(BigFloat,256) do
        sd=sqrt(BigFloat(variance))
        width=iszero(iqr) ? (iszero(variance) ? big"1" : sd) : min(sd,BigFloat(iqr)/BigFloat(1.34))
        Float64(BigFloat(0.9)*width*BigFloat(count)^BigFloat(-0.2))
    end
end

const W = Float16[.25,1,1,1,.5,.5,.5,.5]
# This companion admits under the same Float16 weights, including Real storage.
const Y = Float16[-.7421875,.421875,-.9921875,-.9140625,-.4453125,.8515625,.6015625,.140625]
const D = BigFloat(0x1.00010b7c3fc74p-41)
const X = BigFloat[1-BigFloat(2.)^-35,1-D,1-D,1-D,1,1,1,1]
const B = (Float64(minimum(X))-4*Float64(E._kde_bandwidth(X,W)),1+4*Float64(E._kde_bandwidth(X,W)))

function admitted(x, w)
    (values,), relative = E._kde_population((x,), w, ((-Inf,Inf),))
    h = E._kde_bounded_bandwidth(values, relative)
    @test h == Float64(E._kde_bandwidth(values, relative))
    @test abs(h/bandwidth_reference(values,relative)-1) <= 2.0^-19
    return h
end

@testset "Bounded coordinate uncertainty and independently admitted controls" begin
    before = (copy(X),copy(W))
    selected = Float64(E._kde_bandwidth(X,W))
    @test abs(selected/bandwidth_reference(X,W)-1) > 2.0^-20
    @test_throws "bandwidth reliably" E._kde_bounded_bandwidth(X,W)
    for offset in (big"0",big"3",big"-3"), scale in (big"0.0009765625",big"1",big"1024"),
            perm in (collect(1:8),[8,7,6,5,4,3,2,1],[3,1,5,2,8,7,4,6])
        x = (offset .+ scale.*X)[perm]; w = W[perm]
        @test abs(E._kde_bandwidth(x,w)/bandwidth_reference(x,w)-1) > 2.0^-20
        @test_throws "bounded estimate" E._kde_bounded_bandwidth(x,w)
    end
    for bounds in (B,(B[1],Inf),(-Inf,B[2]))
        @test_throws "bandwidth reliably" E._weighted_kde1d(X,W,bounds)
    end
    @test isequal((X,W),before)
    @test admitted(Y,W) > 0
    @test all(isfinite,E._weighted_kde1d(Y,W,(-2.,2.)).density)
    for swapped in (false,true), mixed in (false,true)
        a,b = swapped ? (Y,X) : (X,Y)
        support = mixed ? ((-Inf,Inf),(-2.,2.)) : (B,(-2.,2.))
        swapped && (support=reverse(support))
        @test_throws "bandwidth reliably" E._weighted_kde2d(a,b,W,support)
    end
    # Recenter before conversion and widen weights; useful wider inputs remain.
    local_x = X .- big"1"; wide = Float64.(W); h = admitted(local_x,wide)
    safe = E._weighted_kde1d(local_x,wide,(Float64(minimum(local_x))-4h,4h))
    @test !isempty(safe.density) && all(isfinite,safe.density)
    for x in ([0.],fill(2.,5),[-0.,0.],[-0.,0.,0.,-0.],[-2.,-1.,0.,1.,2.],
            Float32[-2,-2,0,2,2],Float16[-2,-2,0,2,2],range(-2.,2.;length=5)),
            w in (ones(length(x)),collect(1.:length(x)))
        before = (copy(x),copy(w)); admitted(x,w)
        @test all(isfinite,E._weighted_kde1d(x,w,(-4.,4.)).density)
        @test isequal((x,w),before)
    end
    x = [-2.,-1.,0.,1.,2.]; w = [1.,2.,3.,4.,2.]
    wbig = [big"1",big"1.00000000000000000000001",big"1",big"1",big"1"]
    @test !allequal(wbig) && allequal(Float64.(wbig))
    for scale in (1e-90,1.,1e90), weights in (ones(5),w,wbig)
        z = scale.*x; admitted(z,weights)
        @test all(isfinite,E._weighted_kde1d(z,weights,(-4scale,4scale)).density)
        @test all(isfinite,E._weighted_kde2d(z,-z,weights,((-4scale,4scale),(-Inf,Inf))).density)
    end
    @test admitted(1e170.*x,ones(5)) > 0
    @test admitted([-1e308,1e308],ones(2)) > 0
    @test_throws "data span" E._weighted_kde1d([-1e308,1e308],ones(2),(-1.1e308,1.1e308))
    for (bad,weights) in ((fill(.1f0,10),ones(10)),
            (Float32[2^24,2^24,2^24,2^24,2^24+4],ones(5)),
            (vcat(zeros(6),nextfloat(0.),1.),ones(8)),
            ([0.,0.,0.,.1,1.],[1.,1.,1.,1e-20,2/3]))
        @test_throws "bounded estimate" E._kde_bounded_bandwidth(bad,weights)
    end
    @test E._weighted_kde1d(vcat(x,NaN),vcat(w,0.),(-4.,4.)) == E._weighted_kde1d(x,w,(-4.,4.))
    @test E._weighted_kde1d([0.,1.],[big"1e400",big"1e-400"],(-4.,4.)) == E._weighted_kde1d([0.],[1.],(-4.,4.))
end

@testset "Generated retained-source bandwidth controls" begin
    rng=MersenneTwister(0x11bada55);accepted=rejected=0
    for T in (Float16,Float32,Float64,BigFloat),Wtype in (Float16,Float64,BigFloat),
            offset in (0.,16.,1e5),repeat in 1:8
        x=T.(offset.+randn(rng,13));w=Wtype.(rand(rng,1:8,13));before=(copy(x),copy(w))
        result=try admitted(x,w) catch e;e end
        if result isa Exception
            @test result isa ArgumentError;rejected+=1
        else
            @test result>0;accepted+=1
        end
        @test isequal((x,w),before)
    end
    @test accepted>0 && rejected>0
    println("GENERATED seed=0x11bada55 accepted=",accepted," rejected=",rejected)
end

samples(data,w)=DensitySampleVector(v=E.VectorOfSimilarVectors(copy(data)),logd=zeros(size(data,2)),weight=copy(w))
function reject(f)
    err=try f();nothing catch e;e end
    @test !isnothing(err)
    msg=sprint(showerror,err)
    @test occursin("KDE: bounded estimate",msg)
    @test occursin("bandwidth reliably",msg)
end
function descendants(axis,T)
    found=Any[]
    function visit(p)
        p isa T && push!(found,p);foreach(visit,p.plots)
    end
    foreach(visit,axis.scene.plots);found
end
cell(fig,i,j)=only(filter(p->p isa Axis,contents(fig[1,1][i,j])))
function wait_done(task)
    @test task isa Task
    task isa Task || error("Missing owned task")
    done=timedwait(()->istaskdone(task),30.;pollint=.005)===:ok
    @test done;done || error("Owner did not terminate")
    @test !istaskfailed(task);wait(task)
end
@testset "MR11 actual bounded public admission and axis reachability" begin
    data=Matrix{Real}(undef,2,8);data[1,:]=X;data[2,:]=Y
    @test all(v->v isa BigFloat,data[1,:]) && all(v->v isa Float16,data[2,:])
    for y in (view(data,2,:),data[2,:])
        h=E._kde_bounded_bandwidth(y,W)
        @test abs(h/bandwidth_reference(y,W)-1)<=2.0^-20
        plan=E._kde_axis_plan(y,W,(-2.,2.),8,256,E._KDE_MAX_AXIS_2D)
        @test !isempty(plan.grid) && plan.h==h
    end
    for recipe in (KDE1D(),QuantileKDE1D(),KDE2D(),QuantileKDE2D())
        n=recipe isa Union{KDE1D,QuantileKDE1D} ? 1 : 2
        coords=n==1 ? reshape(X,1,:) : data
        support=n==1 ? B : (B,(-2.,2.));config=n==1 ? DIAG : PAIR
        for levels in (nothing,Float64[])
            c=merge(config,(;support));isnothing(levels) || (c=merge(c,(;levels)))
            reject(()->E.compute_plotting_primitives(view(coords,:,:),view(W,:),recipe,E.LiveRecipe(),E.LiveCell(),c))
        end
        for zero in (false,true)
            z=zero ? coords : coords[:,1:0];w=zero ? zeros(8) : Float64[]
            @test all(isempty,values(E.compute_plotting_primitives(view(z,:,:),view(w,:),recipe,E.LiveRecipe(),E.LiveCell(),merge(config,(;support)))))
        end
        for swapped in (n==1 ? (false,) : (false,true)),mixed in (n==1 ? (false,) : (false,true)),ordinary in (false,true)
            d=swapped ? reverse(coords;dims=1) : coords
            bounds=n==1 ? [B] : mixed ? [(-Inf,Inf),(-2.,2.)] : [B,(-2.,2.)]
            swapped && (bounds=reverse(bounds))
            source=samples(d,W);before=deepcopy(source)
            recipes=n==1 ? (upper=Scatter2D,diagonal=typeof(recipe),lower=Scatter2D) :
                (upper=typeof(recipe),diagonal=QuantileHist1D,lower=typeof(recipe))
            fig=Figure()
            try
                reject() do
                    ordinary ? Makie.plot(fig[1,1],source;recipes,vsel=collect(1:n),N_max=n,support=bounds) :
                        bat_makie_plot(source,recipes,collect(1:n),n;support=bounds,display=false)
                end
            finally
                empty!(fig);ordinary || empty!(Makie.current_figure())
            end
            @test source==before
        end
    end
    for recipe in (KDE1D,QuantileKDE1D,KDE2D,QuantileKDE2D),ordinary in (false,true)
        n=recipe in (KDE1D,QuantileKDE1D) ? 1 : 2
        x=[-2.,-1.,0.,1.,2.];w=[1.,2.,3.,4.,2.];d=n==1 ? reshape(x,1,:) : permutedims(hcat(x,-x))
        source=samples(d,w);bounds=fill((-3.,3.),n)
        recipes=n==1 ? (upper=Scatter2D,diagonal=recipe,lower=Scatter2D) : (upper=recipe,diagonal=QuantileHist1D,lower=recipe)
        wrapper=nothing;fig=ordinary ? Figure() : bat_makie_plot(source,recipes,collect(1:n),n;support=bounds,display=false)
        try
            ordinary && (wrapper=Makie.plot(fig[1,1],source;recipes,vsel=collect(1:n),N_max=n,support=bounds))
            @test !isempty(Makie.colorbuffer(fig))
            if recipe===KDE1D
                curves=filter(p->length(p[1][])>2 && all(>(0),diff(getindex.(p[1][],1))),descendants(cell(fig,1,1),Makie.Lines))
                line=only(curves)[1][];grid=Float64[p[1] for p in line];density=Float64[p[2] for p in line]
                h=bandwidth_reference(x,w);expected=zeros(length(grid));mass=0.
                for (xi,wi) in zip(x,w),center in (xi,-6-xi,6-xi)
                    center!=xi && abs(center-xi)>=8h && continue
                    expected .+= [wi*exp(-.5*((t-center)/h)^2)/(sqrt(2pi)*h) for t in grid]
                    mass+=wi*(cdf(Normal(),(last(grid)-center)/h)-cdf(Normal(),(first(grid)-center)/h))
                end
                @test maximum(abs.(density.-expected./mass))/maximum(expected./mass)<.005
            end
        finally
            empty!(fig);isnothing(wrapper)||Makie.free(wrapper)
        end
    end
end
@testset "MR11 actual rejected updates, control and owner cleanup" begin
    good=samples(reshape(Real[Y...],1,:),W);bad=samples(reshape(Real[X...],1,:),W)
    @test typeof(good)===typeof(bad)
    recipes=(upper=Scatter2D,diagonal=KDE1D,lower=Scatter2D);support=[(-2.,2.)]
    input=Observable(good);fig=Figure();wrapper=nothing
    try
        wrapper=Makie.plot(fig[1,1],input;recipes,vsel=[1],N_max=1,support)
        for value in (good[1:0],good)
            input[]=value;@test !isempty(Makie.colorbuffer(fig))
        end
        reject(()->(input[]=bad))
    finally
        empty!(fig);isnothing(wrapper)||Makie.free(wrapper)
    end
    @test isempty(Makie.Observables.listeners(input))
    fresh=Figure();wrapper=nothing
    try
        wrapper=Makie.plot(fresh[1,1],good;recipes,vsel=[1],N_max=1,support)
        @test !isempty(Makie.colorbuffer(fresh))
    finally
        empty!(fresh);isnothing(wrapper)||Makie.free(wrapper)
    end
    hist=(upper=Scatter2D,diagonal=QuantileHist1D,lower=Scatter2D)
    fig=bat_makie_plot(bad,hist,[1],1;support,display=false)
    menu=only(filter(p->p isa Menu && any(o->last(o)===KDE1D,p.options[]),fig.content))
    try
        reject(()->(menu.selection[]=KDE1D))
    finally
        empty!(fig)
    end
    @test isempty(Makie.Observables.listeners(menu.selection))
    graph,_=E._setup_static_graph(good,recipes,[1],1,PAIR,DIAG;support)
    fig=Figure();other=Figure();lines!(Axis(other[1,1]),[0.,1.],[1.,0.]);snapshot=copy(Makie.colorbuffer(other))
    vis=BATVisualizer(BATMakieVisualization());E._claim_live_visualizer!(vis)
    try
        E._bind_interactive_grid!(fig,E._init_gridlayout!(graph,1))
        @test E._with_live_view(()->E._attach_live_view!(vis,fig),vis)
        E._start_live_listener!(vis,()->nothing,()->nothing)
        listener=lock(()->vis.content.lifecycle.task,vis.content.buffer_lock)
        logger=Test.TestLogger(min_level=Logging.Error)
        @test !with_logger(logger) do
            E._with_live_view(vis) do
                Makie.update!(graph;samples=[[bad]],current_idxs=[[8]])
            end
        end
        try
            record=only(logger.logs);@test record.kwargs[:phase]===:view
            message=sprint(showerror,first(record.kwargs[:exception]))
            @test occursin("bounded estimate",message) && occursin("bandwidth reliably",message)
        finally
            empty!(logger.logs)
        end
        releaser=lock(()->vis.content.lifecycle.disposal_task,vis.content.buffer_lock)
        wait_done(listener);wait_done(releaser)
        status=BAT.visualizer_status(vis)
        @test status.outcome===:failed && !status.data_complete && !status.presentation_valid
        @test status.quiesced && status.view_disposed && !status.cleanup_pending
        @test isempty(fig.content)
        @test !E._with_live_view(()->error("revived"),vis)
        @test Makie.colorbuffer(other)==snapshot
    finally
        BAT.request_visualizer_stop!(vis);BAT.finish_visualizer!(vis;success=false);BAT.dispose_visualizer!(vis)
        empty!(fig);empty!(other)
    end
end

mutable struct NativeCapture <: AbstractDisplay
    figures::Vector{Figure}
end
Base.displayable(::NativeCapture, ::MIME"image/png") = true
Base.display(capture::NativeCapture, figure::Figure) = (push!(capture.figures, figure); nothing)

function native_with_capture(f)
    capture = NativeCapture(Figure[])
    old_inline = Makie.ALWAYS_INLINE_PLOTS[]
    Makie.inline!(true)
    pushdisplay(capture)
    try
        return with_logger(() -> f(capture), NullLogger())
    finally
        popdisplay(capture)
        Makie.inline!(old_inline)
    end
end

function native_chain_snapshot(chain)
    (current=deepcopy(chain.current), proposed=deepcopy(chain.proposed),
        output=deepcopy(chain.output), accepted=copy(chain.accepted),
        order=copy(chain.walker_order), info=chain.info, stepno=chain.stepno,
        nattempts=copy(chain.nattempts), nsamples=copy(chain.nsamples),
        rng=copy(BAT.get_rng(chain.context)),
        walker_rngs=[copy(BAT.get_rng(context)) for context in chain.walker_genctxs])
end

function native_configuration(recipes)
    c = BATMakieVisualization(adaptive_batching=false)
    BATMakieVisualization(recipes, [1, 2], 2, 20_000, 40_000, false,
        c.batch_growth_rate, 0.01, c.dark, PAIR, DIAG; observer_timeout=60.0)
end

function native_algorithm(callback)
    TransformedMCMC(nchains=2, nsteps=24, strict=false, store_burnin=true,
        sample_weighting=ARPWeighting(),
        init=MCMCChainPoolInit(nsteps_init=20), callback=callback,
        convergence=AssumeConvergence(), adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=3))
end

@testset "Bounded coordinate rejection preserves sampling and releases its owner" begin
    initial_recipes = (upper=Scatter2D, diagonal=QuantileHist1D, lower=Scatter2D)
    kde_recipes = (upper=KDE2D, diagonal=KDE1D, lower=KDE2D)
    vis = BATVisualizer(native_configuration(initial_recipes))
    plain_context = BATContext(rng=BAT.Philox4x((144, 72)))
    live_context = BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis)
    target = product_distribution([Uniform(-2., 4.), Uniform(-3., 5.)])
    plain_callbacks, live_callbacks = Any[], Any[]
    injected = Ref(false)
    listener = Ref{Union{Nothing,Task}}(nothing)
    releaser = Ref{Union{Nothing,Task}}(nothing)
    owned_view = Ref{Any}(nothing)
    owned_figure = Ref{Union{Nothing,Figure}}(nothing)
    owned_handles = Any[]
    callback_steps = Ref{Vector{Int64}}(Int64[])
    plain_callback = function(tag, states)
        push!(plain_callbacks, (tag=tag, states=native_chain_snapshot.(getproperty.(states, :chain_state))))
        return nothing
    end
    live_callback = function(tag, states)
        state_before = native_chain_snapshot.(getproperty.(states, :chain_state))
        push!(live_callbacks, (tag=tag, states=state_before))
        if !injected[]
            graph = E._live_graph(vis)
            @test !isempty(graph[:flat_weights][])
            @test graph[:support_lo][] == [-2., -3.]
            @test graph[:support_hi][] == [4., 5.]
            good = graph[E.primitive_symbol(QuantileHist1D(), (1, 1))][]
            @test any(v -> v isa AbstractArray && !isempty(v), values(good))
            listener[] = lock(() -> vis.content.lifecycle.task, vis.content.buffer_lock)
            owned_view[] = lock(() -> vis.content.lifecycle.view, vis.content.buffer_lock)
            owned_figure[] = owned_view[].figure
            @test owned_figure[] isa Figure
            @test !isempty(owned_figure[].content)
            append!(owned_handles, owned_view[].subscriptions)
            @test !isempty(owned_handles)
            @test all(h -> any(pair -> last(pair) === h.f,
                Makie.Observables.listeners(h.observable)), owned_handles)
            callback_steps[] = [state.chain_state.stepno for state in states]
            injected[] = true

            # Keep the native ARP publication schema, grouping, row counts and
            # metadata. Only copied coordinates and Float64 weights change.
            x = [0., 0., 1.]
            weights = [1., 1., 1e-30]
            original = graph[:samples][]
            before = deepcopy(original)
            payload = deepcopy(original)
            @test typeof(payload) === typeof(original)
            @test axes(payload) == axes(original)
            @test length.(payload) == length.(original)
            walkers = collect(Iterators.flatten(payload))
            for walker in walkers
                @test eltype(walker.weight) === Float64
                walker.v.data .= 0
                walker.weight .= 0
            end
            slot = findfirst(walker -> length(walker) >= 3, walkers)
            @test !isnothing(slot)
            isnothing(slot) && error("Display fixture needs three retained rows")
            walkers[slot].v.data[1, 1:3] .= x
            walkers[slot].weight[1:3] .= weights
            idxs = [length.(chain) for chain in payload]
            @test idxs == graph[:current_idxs][]
            window = E._sample_window_data(payload, idxs, nothing)
            @test typeof(window) === typeof(graph[:window_data][])
            selected, groups = E._window_selection(window, nothing)
            @test !window.timed && isempty(window.groups) && isempty(groups)
            positive = findall(>(0), selected.weight)
            @test selected.v.data[:, positive] == permutedims(hcat(x, zeros(3)))
            @test selected.weight[positive] == weights
            for (a, b) in zip(Iterators.flatten(payload), Iterators.flatten(before))
                @test typeof(a) === typeof(b)
                @test length(a) == length(b)
                @test a.info == b.info && a.aux == b.aux && a.logd == b.logd
            end
            @test original == before
            @test native_chain_snapshot.(getproperty.(states, :chain_state)) == state_before

            # This native-weight case selects an independently correct tiny h.
            # MR11 conservatively rejects its coordinate uncertainty before the
            # inherited grid-budget rejection. The old scalar acceptance is
            # checked separately against committed MR10 in supplemental evidence.
            expected = bandwidth_reference(x, weights)
            @test expected ≈ 5.540149860052124e-16 rtol=3e-15
            @test Float64(E._kde_bandwidth(x, weights)) ≈ expected rtol=3e-15
            cause = "KDE: bounded estimate cannot resolve its bandwidth reliably"
            @test_throws cause E._kde_bounded_bandwidth(x, weights)
            @test_throws cause E._weighted_kde1d(x, weights, (-2., 4.))
            companion = E._weighted_kde1d(zeros(3), weights, (-3., 5.))
            @test !isempty(companion.density) && all(isfinite, companion.density)
            @test any(>(0), companion.density)
            @test E._kde_bounded_bandwidth(zeros(3), weights) ≈
                bandwidth_reference(zeros(3), weights) rtol=3e-15

            logger = Test.TestLogger(min_level=Logging.Error)
            @test !with_logger(logger) do
                E._live_graph_update!(vis; publication=true, samples=payload, current_idxs=idxs,
                    sample_timing=nothing, window_steps=nothing, domain_lo=[0., 0.], domain_hi=[1., 0.],
                    upper_recipe=kde_recipes.upper, diagonal_recipe=kde_recipes.diagonal,
                    lower_recipe=kde_recipes.lower)
            end
            try
                record = only(logger.logs)
                @test record.kwargs[:phase] === :view
                message = sprint(showerror, first(record.kwargs[:exception]))
                @test occursin(cause, message)
            finally
                empty!(logger.logs)
            end
            releaser[] = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
            @test !vis.content.lifecycle.accepting[]
            @test original == before
            @test native_chain_snapshot.(getproperty.(states, :chain_state)) == state_before
        end
        return nothing
    end

    try
        Random.seed!(8921)
        plain = with_logger(() -> BAT.evalmeasure(target, native_algorithm(plain_callback), plain_context), NullLogger())
        default_after_plain = copy(Random.default_rng())
        Random.seed!(8921)
        observed = native_with_capture() do capture
            result = BAT.evalmeasure(target, native_algorithm(live_callback), live_context)
            @test length(capture.figures) == 1
            @test only(capture.figures) === owned_figure[]
            result
        end
        @test injected[]
        @test length(plain_callbacks) == length(live_callbacks) == 1
        @test live_callbacks == plain_callbacks
        live_samples, plain_samples = BAT.samplesof(observed), BAT.samplesof(plain)
        @test live_samples == plain_samples
        for field in (:v, :logd, :weight, :info, :aux)
            @test isequal(getproperty(live_samples, field), getproperty(plain_samples, field))
        end
        live_chains = BAT.samplegenof(observed).chain_states
        plain_chains = BAT.samplegenof(plain).chain_states
        @test native_chain_snapshot.(live_chains) == native_chain_snapshot.(plain_chains)
        @test all(chain -> chain.stepno == 24, live_chains)
        @test callback_steps[] == fill(12, 2)
        @test BAT.get_rng(live_context) == BAT.get_rng(plain_context)
        @test rand(UInt64, 16) == rand(default_after_plain, UInt64, 16)

        # Join the captured listener and disposal owners before status reads or
        # rescue cleanup can retire tasks or hide an autonomous-release failure.
        wait_done(listener[])
        wait_done(releaser[])
        @test isnothing(vis.content.graph[])
        @test isnothing(vis.content.lifecycle.task)
        @test isnothing(vis.content.lifecycle.timer)
        @test isnothing(vis.content.lifecycle.owner_task)
        @test isempty(vis.content.output_buffer) && isempty(vis.content.pending_samples)
        @test isempty(vis.content.output_timing) && isempty(vis.content.pending_timing)
        @test isempty(owned_view[].subscriptions)
        @test isnothing(owned_view[].figure)
        @test isempty(owned_figure[].content)
        @test isempty(owned_figure[].scene.deregister_callbacks)
        @test all(h -> all(pair -> last(pair) !== h.f,
            Makie.Observables.listeners(h.observable)), owned_handles)
        @test !E._live_graph_update!(vis; publication=true, samples=Any[])
        status = BAT.visualizer_status(vis)
        @test status.outcome === :failed
        @test !status.data_complete && !status.presentation_valid
        @test status.quiesced && status.history_released && status.view_disposed
        @test !status.cleanup_pending && isnothing(status.cleanup_error)
        @test status.error.type == "ResolveException" && status.error.phase === :view
        @test status.error.message == "Exception details are not retained"
        @test isnothing(BAT.visualizer_figure(vis))
    finally
        BAT.request_visualizer_stop!(vis; reason=:disabled)
        BAT.finish_visualizer!(vis; success=false)
        BAT.dispose_visualizer!(vis)
        coordinator = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
        isnothing(coordinator) || wait_done(coordinator)
    end
end

end # module
