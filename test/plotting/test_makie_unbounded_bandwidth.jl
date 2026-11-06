# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieUnboundedBandwidth

using BAT, Test, Makie, CairoMakie, Statistics, Distributions, Random, Logging

const EXT = Base.get_extension(BAT, :BATMakieExt)
const PAIR = BAT._default_makie_triagonal_config()
const DIAG = BAT._default_makie_diagonal_config()
const RECIPES = (upper=KDE2D, diagonal=KDE1D, lower=KDE2D)

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

# Preserve the inherited dependency call independently of the new admission
# checks. The exact-rational oracle separately checks the selected bandwidth.
function check_estimate(coords, weights)
    relative = weights ./ maximum(weights)
    hs = map(x -> allequal(relative) ? nothing : EXT._weighted_kde_bandwidth(x,relative), coords)
    data = length(coords) == 1 ? only(coords) : coords
    expected = any(isnothing,hs) ? EXT.KernelDensity.kde(data; weights=relative) :
        EXT.KernelDensity.kde(data; weights=relative,bandwidth=length(coords)==1 ? only(hs) : hs)
    actual = length(coords)==1 ? EXT._weighted_kde1d(only(coords),weights) :
        EXT._weighted_kde2d(coords...,weights)
    @test actual.x == expected.x
    @test actual.density == expected.density
    length(coords)==2 && (@test actual.y == expected.y)
end

function admission(E,x,weights)
    (v,),w=E._kde_population((x,),weights,((-Inf,Inf),))
    weighted=!allequal(w)
    h=weighted ? E._weighted_kde_bandwidth(v,w) : E.KernelDensity.default_bandwidth(v)
    isnothing(h) && throw(ArgumentError("unavailable selected bandwidth"))
    E._kde_check_unbounded_bandwidth(v,w,h,weighted)
    return h,v,w
end
function rejects(x,w)
    for support in (nothing,(-Inf,Inf))
        @test_throws "unbounded estimate" EXT._weighted_kde1d(x,w,support)
    end
    other=collect(range(-1.,1.;length=length(x)))
    for swapped in (false,true),support in (nothing,((-Inf,Inf),(-Inf,Inf)))
        a,b=swapped ? (other,x) : (x,other)
        @test_throws "unbounded estimate" EXT._weighted_kde2d(a,b,w,support)
    end
end
@testset "MR10 independent admission, counterexamples and exact preservation" begin
    for small in (1e-20,8e-16)
        x=[0.,0.,0.,0.1,1.];w=[1.,1.,1.,small,2/3]
        expected=bandwidth_reference(x,w);original=EXT._kde_bandwidth(x,w)
        @test abs(original/expected-1)>0.1
        for perm in ([1,2,3,4,5],[5,4,3,2,1],[4,1,5,2,3])
            rejects(x[perm],w[perm])
        end
        for offset in (0.,16.),scale in (2.0^-20,1.,2.0^20)
            rejects(offset .+ scale.*x,w)
        end
    end
    # Constant width-one fallback also has to validate source Kish arithmetic.
    x=zeros(3);w=Float16[1,0.5,0.25]
    @test abs(EXT._kde_bandwidth(x,w)/bandwidth_reference(x,w)-1)>2.0^-20
    rejects(x,w)
    @test admission(EXT,x,Float64.(w))[1] ≈ bandwidth_reference(x,w) rtol=3e-15
    check_estimate((x,),Float64.(w))
    # First-origin rounding can cancel a source effective-count discrepancy.
    delta=0x1.00010b7c3fc74p-41
    x=[-1.,-delta,-delta,-delta,0.,0.,0.,0.]
    w=Float16[0.25,1,1,1,0.5,0.5,0.5,0.5]
    @test abs(EXT._kde_bandwidth(x,w)/bandwidth_reference(x,w)-1)>2.0^-20
    rejects(x,w)
    # Widening alone still fails the preserved first-origin gate. Reordering
    # the same represented observations puts that anchor at the quartile.
    @test_throws "bandwidth reliably" admission(EXT,x,Float64.(w))
    perm=[2,1,3,4,5,6,7,8]
    @test admission(EXT,x[perm],Float64.(w[perm]))[1] ≈ bandwidth_reference(x,Float64.(w)) rtol=3e-15
    converted=BigFloat[0,1-BigFloat(delta),1-BigFloat(delta),1-BigFloat(delta),1,1,1,1]
    @test abs(EXT._kde_bandwidth(converted,w)/bandwidth_reference(converted,w)-1)>2.0^-20
    rejects(converted,w)

    # Uniform and weighted false-zero IQR and coalesced original plateaus.
    for (x,w) in ((vcat(zeros(6),nextfloat(0.),1.),ones(8)),
            ([0.,0.,0.,0.5,1.],[1.,1.,1.,nextfloat(1/3),1/3]))
        rejects(x,w)
    end
    small=big"1e-30"
    x=[big"0",big"1",big"1",big"1",1+small,1+small,1+small,big"2"]
    rejects(x,ones(8))

    examples=([-2.,-2.,0.,2.,2.],[-10.,-.1,0.,.1,10.],[0.,0.,0.,0.,4.],
        fill(2.,5),[0.],Float32[-2,-2,0,2,2],Float16[-2,-2,0,2,2],range(-2.,2.;length=5))
    for x in examples,w in (ones(length(x)),collect(1.:length(x)))
        before=(copy(x),copy(w));h,v,rw=admission(EXT,x,w)
        @test h == EXT._kde_bandwidth(v,rw)
        @test h ≈ bandwidth_reference(v,rw) rtol=2.0^-20
        check_estimate((x,),w)
        @test isequal((x,w),before)
    end
    x=[-2.,-1.,0.,1.,2.];w=[1.,2.,3.,4.,2.]
    for sx in (2.0^-40,1.,2.0^40),sy in (1.,inv(sx))
        check_estimate((sx.*x,),w)
        check_estimate((sx.*x,sy.*-x),w)
    end
    wbig=[big"1",big"1.00000000000000000000001",big"1",big"1",big"1"]
    @test allequal(Float64.(wbig)) && !allequal(wbig)
    for scale in (big"1",big"1e150")
        v=scale.*x;h,xx,ww=admission(EXT,v,wbig)
        @test h isa BigFloat
        @test h == EXT._kde_bandwidth(xx,ww)
        @test h ≈ bandwidth_reference(xx,ww) rtol=2.0^-20
        check_estimate((v,-v),wbig)
    end
    base=EXT._weighted_kde1d(x,w)
    @test EXT._weighted_kde1d(vcat(x,NaN),vcat(w,0.))==base
    @test EXT._weighted_kde1d([0.,1.],[big"1e400",big"1e-400"])==EXT._weighted_kde1d([0.],[big"1"])
    again=EXT._weighted_kde1d(x,w)
    @test base.density !== again.density
    # Keep existing constant/signed-zero and source/physical arithmetic limits.
    rejects([-0.,0.],ones(2))
    for factor in (1e-170,1e170)
        rejects(factor.*x,ones(5))
    end
end
@testset "Generated admission is conservative and independently accurate" begin
    rng=MersenneTwister(0x10c0ffee);accepted=rejected=0
    for T in (Float16,Float32,Float64), W in (Float16,Float32,Float64),
            offset in (0.,16.),repeat in 1:8
        x=T.(offset .+ randn(rng,17));w=W.(rand(rng,1:8,17))
        before=(copy(x),copy(w))
        result=try admission(EXT,x,w) catch e;e end
        if result isa Exception
            @test result isa ArgumentError
            rejected+=1
        else
            h,v,rw=result;old=EXT._kde_bandwidth(v,rw)
            @test h==old
            @test abs(h/bandwidth_reference(v,rw)-1)<=2.0^-19
            accepted+=1
        end
        @test isequal((x,w),before)
    end
    @test accepted>0 && rejected>0
    println("GENERATED seed=0x10c0ffee accepted=",accepted," rejected=",rejected)
end

function check_enclosure(x,w)
    weighted=!allequal(w);rw=Float64.(w);z=Float64.(x)
    origin=weighted ? only(EXT.quantile(z,EXT.Weights(rw),[.25])) : quantile(z,.25)
    z .-= origin;span=maximum(abs,z);span>0 || return
    lo=similar(z);hi=similar(z);error=0.
    for i in eachindex(z)
        v=Float64(x[i]);a=v>x[i] ? prevfloat(v) : v;b=v<x[i] ? nextfloat(v) : v
        lo[i]=prevfloat(prevfloat(a-origin)/span);hi[i]=nextfloat(nextfloat(b-origin)/span)
        z[i]/=span
        error=max(error,nextfloat(z[i]-lo[i]),nextfloat(hi[i]-z[i]))
        exact=(Rational{BigInt}(x[i])-Rational{BigInt}(origin))/Rational{BigInt}(span)
        @test Rational{BigInt}(lo[i])<=exact<=Rational{BigInt}(hi[i])
    end
    low,high=EXT._kde_iqr_bounds(x,w,z,weighted,(lo,hi))
    exact=(exact_quantile(x,w,3//4,weighted)-exact_quantile(x,w,1//4,weighted))/Rational{BigInt}(span)
    @test low <= high
    @test low == -Inf || (isfinite(low) && Rational{BigInt}(low)<=exact)
    @test high == Inf || (isfinite(high) && exact<=Rational{BigInt}(high))
    @test (low,high)!=(0.,0.) || iszero(exact)
    # Verify the coordinate-error SD theorem using exact rational moments and
    # high-precision roots, separately from floating reference-SD rounding.
    original=(Rational{BigInt}.(x).-Rational{BigInt}(origin))./Rational{BigInt}(span)
    point=Rational{BigInt}.(z);weights=Rational{BigInt}.(rw)
    function variance(v)
        mean=weighted ? sum(weights.*v)/sum(weights) : sum(v)/length(v)
        weighted ? sum(weights.*(v.-mean).^2)/sum(weights) : sum(abs2,v.-mean)/(length(v)-1)
    end
    setprecision(BigFloat,256) do
        difference=abs(sqrt(BigFloat(variance(original)))-sqrt(BigFloat(variance(point))))
        @test difference<=BigFloat(nextfloat(2error))
    end
end
@testset "Exact original-coordinate/rank enclosure and SD bound" begin
    rng=MersenneTwister(0x10b0a1d)
    for weighted in (false,true),repeat in 1:200
        x=Float64.(rand(rng,-16:16,12))./16
        w=weighted ? Float64.(rand(rng,1:8,12))./8 : ones(12)
        check_enclosure(x,w)
    end
    for shift in (big"0",big"1",big"1e150"),scale in (big"1e-14",big"1",big"1e140")
        x=shift .+ scale.*BigFloat[-2,-1,0,1,2]
        length(unique(Float64.(x)))>1 || continue
        check_enclosure(x,[1.,.5,1.,.5,1.])
    end
    d=BigFloat(0x1.00010b7c3fc74p-41)
    check_enclosure(BigFloat[0,1-d,1-d,1-d,1,1,1,1],Float16[.25,1,1,1,.5,.5,.5,.5])
    for w in ([1.,1.,1.,1e-20,2/3],[1.,1.,1.,8e-16,2/3],
            [big"1",big"1.00000000000000000000001",big"1",big"0.3",big"0.5"])
        check_enclosure([0.,0.,0.,.1,1.],w)
    end
    check_enclosure([0.,0.,0.,0.,4.],[2.,3.,4.,5.,1.])
    println("INTERVAL seed=0x10b0a1d")
end

samples(data,w)=DensitySampleVector(v=EXT.VectorOfSimilarVectors(copy(data)),logd=zeros(size(data,2)),weight=copy(w))
function reject(f)
    error=try f();nothing catch e;e end
    @test !isnothing(error)
    @test occursin("KDE: unbounded estimate",sprint(showerror,error))
end
function descendants(axis,T)
    found=Any[]
    function visit(p)
        p isa T && push!(found,p)
        foreach(visit,p.plots)
    end
    foreach(visit,axis.scene.plots);found
end
cell(fig,i,j)=only(filter(p->p isa Axis,contents(fig[1,1][i,j])))
function wait_done(task)
    @test task isa Task
    task isa Task || error("Missing owned task in unbounded bandwidth fixture")
    done=timedwait(()->istaskdone(task),30.;pollint=.005)===:ok
    @test done
    done || error("Owned task did not terminate")
    @test !istaskfailed(task)
    wait(task)
end
@testset "MR10 actual public demand and ownership" begin
    fixtures=(([0.,0.,0.,.1,1.],[1.,1.,1.,1e-20,2/3]),
        ([0.,0.,0.,.1,1.],[1.,1.,1.,8e-16,2/3]),(zeros(3),Float16[1,.5,.25]))
    for (x,w) in fixtures,recipe in (KDE1D(),QuantileKDE1D(),KDE2D(),QuantileKDE2D())
        n=recipe isa Union{KDE1D,QuantileKDE1D} ? 1 : 2
        data=n==1 ? reshape(x,1,:) : permutedims(hcat(x,collect(range(-1.,1.;length=length(x)))))
        config=n==1 ? DIAG : PAIR
        for levels in (nothing,Float64[])
            c=isnothing(levels) ? config : merge(config,(;levels))
            @test_throws "unbounded estimate" EXT.compute_plotting_primitives(view(data,:,:),view(w,:),recipe,EXT.LiveRecipe(),EXT.LiveCell(),c)
        end
        for zero in (false,true)
            z=zero ? data : data[:,1:0];weights=zero ? zeros(length(w)) : Float64[]
            @test all(isempty,values(EXT.compute_plotting_primitives(view(z,:,:),view(weights,:),recipe,EXT.LiveRecipe(),EXT.LiveCell(),config)))
        end
        for swapped in (n==1 ? (false,) : (false,true)),ordinary in (false,true)
            d=swapped ? reverse(data;dims=1) : data
            source=samples(d,w);before=deepcopy(source)
            recipes=n==1 ? (upper=Scatter2D,diagonal=typeof(recipe),lower=Scatter2D) :
                (upper=typeof(recipe),diagonal=QuantileHist1D,lower=typeof(recipe))
            fig=Figure()
            try
                reject() do
                    ordinary ? Makie.plot(fig[1,1],source;recipes,vsel=collect(1:n),N_max=n) :
                        bat_makie_plot(source,recipes,collect(1:n),n;display=false)
                end
            finally
                empty!(fig);ordinary || empty!(Makie.current_figure())
            end
            @test source==before
        end
    end
    for (x,w) in (([-2.,-1.,0.,1.,2.],[1.,2.,3.,4.,2.]),(zeros(3),[1.,.5,.25])),ordinary in (false,true)
        source=samples(reshape(x,1,:),w);recipes=(upper=Scatter2D,diagonal=KDE1D,lower=Scatter2D)
        wrapper=nothing;fig=ordinary ? Figure() : bat_makie_plot(source,recipes,[1],1;display=false)
        try
            ordinary && (wrapper=Makie.plot(fig[1,1],source;recipes,vsel=[1],N_max=1))
            curves=filter(p->length(p[1][])>2 && all(>(0),diff(getindex.(p[1][],1))),descendants(cell(fig,1,1),Makie.Lines))
            line=only(curves)[1][];grid=Float64[p[1] for p in line];ys=Float64[p[2] for p in line]
            h=bandwidth_reference(x,w./maximum(w))
            @test (minimum(x)-first(grid))/4 ≈ h rtol=3e-15
            expected=[sum(wi*exp(-.5*((t-xi)/h)^2)/(sqrt(2pi)*h) for (xi,wi) in zip(x,w))/sum(w) for t in grid]
            @test maximum(abs.(ys.-expected))/maximum(expected)<.003
            @test !isempty(Makie.colorbuffer(fig))
        finally
            empty!(fig);isnothing(wrapper)||Makie.free(wrapper)
        end
    end
    good=samples(reshape([-2.,-1.,0.,1.,2.],1,:),[1.,2.,3.,4.,2.])
    bad=samples(reshape(fixtures[1][1],1,:),fixtures[1][2])
    recipes=(upper=Scatter2D,diagonal=KDE1D,lower=Scatter2D)
    input=Observable(good);fig=Figure();wrapper=nothing
    try
        wrapper=Makie.plot(fig[1,1],input;recipes,vsel=[1],N_max=1)
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
        wrapper=Makie.plot(fresh[1,1],good;recipes,vsel=[1],N_max=1)
        @test !isempty(Makie.colorbuffer(fresh))
    finally
        empty!(fresh);isnothing(wrapper)||Makie.free(wrapper)
    end
    hist=(upper=Scatter2D,diagonal=QuantileHist1D,lower=Scatter2D)
    fig=bat_makie_plot(bad,hist,[1],1;display=false)
    menu=only(filter(p->p isa Menu && any(o->last(o)===KDE1D,p.options[]),fig.content))
    try
        reject(()->(menu.selection[]=KDE1D))
    finally
        empty!(fig)
    end
    @test isempty(Makie.Observables.listeners(menu.selection))

    graph,_=EXT._setup_static_graph(good,recipes,[1],1,PAIR,DIAG)
    fig=Figure();other=Figure();lines!(Axis(other[1,1]),[0.,1.],[1.,0.]);snapshot=copy(Makie.colorbuffer(other))
    vis=BATVisualizer(BATMakieVisualization());EXT._claim_live_visualizer!(vis)
    try
        EXT._bind_interactive_grid!(fig,EXT._init_gridlayout!(graph,1))
        @test EXT._with_live_view(()->EXT._attach_live_view!(vis,fig),vis)
        EXT._start_live_listener!(vis,()->nothing,()->nothing)
        listener=lock(()->vis.content.lifecycle.task,vis.content.buffer_lock)
        logger=Test.TestLogger(min_level=Logging.Error)
        @test !with_logger(logger) do
            EXT._with_live_view(vis) do
                Makie.update!(graph;samples=[[bad]],current_idxs=[[length(bad)]])
            end
        end
        try
            record=only(logger.logs)
            @test record.kwargs[:phase]===:view
            @test occursin("interquartile width",sprint(showerror,first(record.kwargs[:exception])))
        finally
            empty!(logger.logs)
        end
        releaser=lock(()->vis.content.lifecycle.disposal_task,vis.content.buffer_lock)
        wait_done(listener);wait_done(releaser)
        status=BAT.visualizer_status(vis)
        @test status.outcome===:failed && !status.data_complete && !status.presentation_valid
        @test status.quiesced && status.view_disposed && !status.cleanup_pending
        @test isempty(fig.content)
        @test !EXT._with_live_view(()->error("revived"),vis)
        @test Makie.colorbuffer(other)==snapshot
    finally
        BAT.request_visualizer_stop!(vis);BAT.finish_visualizer!(vis;success=false);BAT.dispose_visualizer!(vis)
        empty!(fig);empty!(other)
    end
end

mutable struct Capture <: AbstractDisplay
    figures::Vector{Figure}
end
Base.displayable(::Capture, ::MIME"image/png") = true
Base.display(capture::Capture, figure::Figure) = (push!(capture.figures, figure); nothing)

function with_capture(f)
    capture = Capture(Figure[])
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

function chain_snapshot(chain)
    (current=deepcopy(chain.current), proposed=deepcopy(chain.proposed),
        output=deepcopy(chain.output), accepted=copy(chain.accepted),
        order=copy(chain.walker_order), info=chain.info, stepno=chain.stepno,
        nattempts=copy(chain.nattempts), nsamples=copy(chain.nsamples),
        rng=copy(BAT.get_rng(chain.context)),
        walker_rngs=[copy(BAT.get_rng(context)) for context in chain.walker_genctxs])
end

function configuration(recipes)
    c = BATMakieVisualization(adaptive_batching=false)
    BATMakieVisualization(recipes, [1, 2], 2, 20_000, 40_000, false,
        c.batch_growth_rate, 0.01, c.dark, PAIR, DIAG; observer_timeout=60.0)
end

function algorithm(callback)
    TransformedMCMC(nchains=2, nsteps=24, strict=false, store_burnin=true,
        sample_weighting=ARPWeighting(),
        init=MCMCChainPoolInit(nsteps_init=20), callback=callback,
        convergence=AssumeConvergence(), adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=3))
end

@testset "Unbounded rank rejection preserves sampling and releases its owner" begin
    initial_recipes = (upper=Scatter2D, diagonal=QuantileHist1D, lower=Scatter2D)
    vis = BATVisualizer(configuration(initial_recipes))
    plain_context = BATContext(rng=BAT.Philox4x((144, 72)))
    live_context = BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis)
    target = MvNormal(zeros(2), ones(2))
    plain_callbacks, live_callbacks = Any[], Any[]
    injected = Ref(false)
    listener = Ref{Union{Nothing,Task}}(nothing)
    releaser = Ref{Union{Nothing,Task}}(nothing)
    callback_steps = Ref{Vector{Int64}}(Int64[])
    plain_callback = function(tag, states)
        push!(plain_callbacks, (tag=tag, states=chain_snapshot.(getproperty.(states, :chain_state))))
        return nothing
    end
    live_callback = function(tag, states)
        push!(live_callbacks, (tag=tag, states=chain_snapshot.(getproperty.(states, :chain_state))))
        if !injected[]
            graph = EXT._live_graph(vis)
            @test !isempty(graph[:flat_weights][])
            @test graph[:support_lo][] == [-Inf, -Inf]
            @test graph[:support_hi][] == [Inf, Inf]
            good = graph[EXT.primitive_symbol(QuantileHist1D(), (1, 1))][]
            @test any(v -> v isa AbstractArray && !isempty(v), values(good))
            listener[] = lock(() -> vis.content.lifecycle.task, vis.content.buffer_lock)
            callback_steps[] = [state.chain_state.stepno for state in states]
            injected[] = true
            # Mutate only copied display coordinates/weights, keeping the
            # live graph's native storage and metadata. ARP supplies Float64
            # weights; this publication declares no repetition-step timing.
            x = [0.,0.,0.,0.1,1.]
            weights = [1.,1.,1.,1e-20,2/3]
            original = graph[:samples][]
            before = deepcopy(original)
            payload = deepcopy(original)
            walkers = collect(Iterators.flatten(payload))
            for walker in walkers
                @test eltype(walker.weight) === Float64
                walker.v.data .= 0
                walker.weight .= 0
            end
            slot = findfirst(walker -> length(walker) >= 5, walkers)
            @test !isnothing(slot)
            isnothing(slot) && error("Display fixture needs five retained rows")
            walkers[slot].v.data[1,1:5] .= x
            walkers[slot].weight[1:5] .= weights
            idxs = [length.(chain) for chain in payload]
            @test idxs == graph[:current_idxs][]
            window = EXT._sample_window_data(payload,idxs,nothing)
            @test typeof(window) === typeof(graph[:window_data][])
            selected,groups = EXT._window_selection(window,nothing)
            @test !window.timed && isempty(window.groups) && isempty(groups)
            positive = findall(>(0),selected.weight)
            @test selected.v.data[:,positive] == permutedims(hcat(x,zeros(5)))
            @test selected.weight[positive] == weights
            @test original == before
            for (a,b) in zip(Iterators.flatten(payload),Iterators.flatten(before))
                @test a.info == b.info && a.aux == b.aux && a.logd == b.logd
            end
            expected = bandwidth_reference(x,weights)
            @test abs(EXT._kde_bandwidth(x,weights)/expected-1) > 0.1
            @test_throws "interquartile width" EXT._weighted_kde1d(x,weights)
            @test all(isfinite,EXT._weighted_kde1d(zeros(5),weights).density)
            logger = Test.TestLogger(min_level=Logging.Error)
            @test !with_logger(logger) do
                EXT._live_graph_update!(vis; publication=true, samples=payload, current_idxs=idxs,
                    sample_timing=nothing, window_steps=nothing, domain_lo=[0.,0.], domain_hi=[1.,0.],
                    upper_recipe=RECIPES.upper, diagonal_recipe=RECIPES.diagonal, lower_recipe=RECIPES.lower)
            end
            try
                record = only(logger.logs)
                @test record.kwargs[:phase] === :view
                message = sprint(showerror, first(record.kwargs[:exception]))
                @test occursin("KDE: unbounded estimate", message)
                @test occursin("interquartile width", message)
            finally
                empty!(logger.logs)
            end
            releaser[] = lock(() -> vis.content.lifecycle.disposal_task, vis.content.buffer_lock)
            @test !vis.content.lifecycle.accepting[]
        end
        return nothing
    end

    try
        Random.seed!(8921)
        plain = with_logger(() -> BAT.evalmeasure(target, algorithm(plain_callback), plain_context), NullLogger())
        default_after_plain = copy(Random.default_rng())
        Random.seed!(8921)
        observed = with_capture() do capture
            result = BAT.evalmeasure(target, algorithm(live_callback), live_context)
            @test length(capture.figures) == 1
            result
        end
        @test injected[]
        @test length(plain_callbacks) == length(live_callbacks) == 1
        @test live_callbacks == plain_callbacks
        @test BAT.samplesof(observed) == BAT.samplesof(plain)
        live_chains = BAT.samplegenof(observed).chain_states
        plain_chains = BAT.samplegenof(plain).chain_states
        @test chain_snapshot.(live_chains) == chain_snapshot.(plain_chains)
        @test all(chain -> chain.stepno == 24, live_chains)
        @test callback_steps[] == fill(12, 2)
        @test BAT.get_rng(live_context) == BAT.get_rng(plain_context)
        @test rand(UInt64, 16) == rand(default_after_plain, UInt64, 16)

        # Wait for the actual owners before any status poll or rescue cleanup.
        wait_done(listener[])
        wait_done(releaser[])
        @test isnothing(vis.content.graph[])
        @test !EXT._live_graph_update!(vis; publication=true, samples=Any[])
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
