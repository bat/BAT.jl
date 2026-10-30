# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieBoundedBandwidth

using BAT, Test, Makie, CairoMakie, Statistics, Distributions, Random, Logging
using ValueShapes: NamedTupleDist

const EXT = Base.get_extension(BAT, :BATMakieExt)
const PAIR = BAT._default_makie_triagonal_config()
const DIAG = BAT._default_makie_diagonal_config()
const RECIPES = (upper=KDE2D, diagonal=KDE1D, lower=KDE2D)

# Independent 256-bit formula on exactly represented observations. Quantiles
# use their documented rank/interpolation definition, not production helpers.
function bandwidth_reference(values, weights)
    setprecision(BigFloat, 256) do
        n = length(values)
        n == 1 && return 0.9
        z = BigFloat.(values) .- BigFloat(first(values))
        w = BigFloat.(weights)
        equal = allequal(weights)
        total = sum(w)
        mu = equal ? sum(z)/n : sum(w .* z)/total
        variance = equal ? sum(abs2, z .- mu)/(n-1) : sum(w .* (z .- mu).^2)/total
        sd = sqrt(variance)
        order = sortperm(eachindex(z); by=i -> (z[i], w[i]))
        z, w = z[order], w[order]
        function q(p)
            if equal
                rank = 1 + p*(n-1)
                i = min(floor(Int, rank), n-1)
                return z[i] + (rank-i)*(z[i+1]-z[i])
            end
            target = w[1] + p*(total-w[1])
            cumulative = cumsum(w)
            j = searchsortedlast(cumulative, target)
            j >= n && return z[end]
            return z[j] + (target-cumulative[j])/w[j+1]*(z[j+1]-z[j])
        end
        width = min(sd, (q(big"0.75")-q(big"0.25"))/BigFloat(1.34))
        iszero(width) && (width = iszero(sd) ? big"1" : sd)
        count = equal ? BigFloat(n) : total^2/sum(abs2,w)
        Float64(BigFloat(0.9)*width*count^BigFloat(-0.2))
    end
end

@testset "MR09 independent bandwidth and preservation" begin
    bad = (fill(0.1f0,10), Float32[2^24,2^24,2^24,2^24,2^24+4])
    @test bandwidth_reference(bad[1],ones(10)) ≈ 0.9*10^(-0.2) rtol=3e-15
    @test bandwidth_reference(bad[2],ones(5)) ≈ 0.9sqrt(16/5)*5^(-0.2) rtol=3e-15
    for x in bad
        w = ones(length(x)); expected = bandwidth_reference(x,w)
        @test abs(EXT._kde_bandwidth(x,w)/expected - 1) > 0.1
        @test_throws "bandwidth reliably" EXT._kde_bounded_bandwidth(x,w)
        for bounds in ((Float64(minimum(x))-100,Float64(maximum(x))+100),
                (Float64(minimum(x))-100,Inf),(-Inf,Float64(maximum(x))+100))
            @test_throws "bounded estimate" EXT._weighted_kde1d(x,w,bounds)
            wide = Float64.(x)
            @test EXT._kde_bounded_bandwidth(wide,w) ≈ expected rtol=3e-15
        end
    end

    # Explicit accepted branches: SD, IQR, zero IQR -> SD, constants, singleton,
    # small floating types, range storage, and unequal BigFloat weights which
    # become equal when narrowed but must retain the population-variance rule.
    examples = ([-2.,-2.,0.,2.,2.], [-10.,-0.1,0.,0.1,10.], [0.,0.,0.,0.,4.],
        fill(2.,5), [0.], Float32[-2,-2,0,2,2], Float16[-2,-2,0,2,2],
        range(-2.,2.;length=5))
    for x in examples, w in (ones(length(x)), collect(1.:length(x)))
        before = (copy(x),copy(w)); h = EXT._kde_bounded_bandwidth(x,w)
        @test h == Float64(EXT._kde_bandwidth(x,w))
        @test h ≈ bandwidth_reference(x,w) rtol=2.0^-20
        @test isequal((x,w),before)
    end
    x = [-2.,-1.,0.,1.,2.]
    wbig = [big"1",big"1.00000000000000000000001",big"1",big"1",big"1"]
    @test allequal(Float64.(wbig)) && !allequal(wbig)
    @test EXT._kde_bounded_bandwidth(x,wbig) ≈ bandwidth_reference(x,wbig) rtol=3e-15
    @test EXT._kde_bounded_bandwidth(x,wbig) != EXT._kde_bounded_bandwidth(x,ones(5))

    for w in ([1.,2.,3.,4.,2.], ones(5)), scale in (1e-90,1.,1e90)
        v = scale .* x
        @test EXT._kde_bounded_bandwidth(v,w) ≈ bandwidth_reference(v,w) rtol=3e-15
    end
    for w in ([1.,1e-100,1e-100,1e-100,1e-100],Float32[1,1e-9,1e-9,1e-9,1e-9])
        @test isnothing(EXT._weighted_kde_bandwidth(x,w))
        @test_throws "effective sample count or bandwidth" EXT._kde_bounded_bandwidth(x,w)
        @test_throws "bounded estimate" EXT._weighted_kde1d(x,w,(-3.,3.))
    end
    @test_throws "bounded estimate" EXT._kde_bounded_bandwidth(1e-170.*x,ones(5))
    # Overflow of source SD alone need not corrupt the selected finite IQR.
    @test EXT._kde_bounded_bandwidth(1e170.*x,ones(5)) ≈ bandwidth_reference(1e170.*x,ones(5)) rtol=3e-15
    # A centered witness can share variance underflow with the source rule.
    @test_throws "bounded estimate" EXT._kde_bounded_bandwidth([0.,0.,1.], [1.,1.,nextfloat(0.)])
    plateau = [0.,1.,1.,1.,1.,1.,nextfloat(1.),2.]
    @test bandwidth_reference(plateau,ones(8)) < 1e-16
    @test_throws "bounded estimate" EXT._kde_bounded_bandwidth(plateau,ones(8))
    collapsed = [big"1",big"1.000000000000000000000001"]
    @test_throws "centered span" EXT._kde_bounded_bandwidth(collapsed,ones(2))
    @test EXT._kde_bounded_bandwidth([-1e308,1e308],ones(2)) ≈ bandwidth_reference([-1e308,1e308],ones(2)) rtol=3e-15
    @test_throws "data span" EXT._weighted_kde1d([-1e308,1e308],ones(2),(-1.1e308,1.1e308))
    t=nextfloat(0.0)
    @test_throws "reference scaling" EXT._kde_bounded_bandwidth([0.,t,t,t,t,t,nextfloat(t),1e150],ones(8))

    # A finite boundary on either axis selects the bounded planner for both.
    xbad = bad[2]; y = [-2.,-1.,0.,1.,2.]; w=ones(5)
    for swap in (false,true), mixed in (false,true)
        bounds=((Float64(first(xbad))-100,Float64(first(xbad))+100),(-5.,5.))
        mixed && (bounds=((-Inf,Inf),bounds[2]))
        xx,yy,bb=swap ? (y,xbad,reverse(bounds)) : (xbad,y,bounds)
        @test_throws "bounded estimate" EXT._weighted_kde2d(xx,yy,w,bb)
    end
    # Existing population filtering, true-singleton fallback and output ownership.
    base=EXT._weighted_kde1d(x,w,(-3.,3.))
    @test EXT._weighted_kde1d(vcat(x,NaN),vcat(w,0.),(-3.,3.)) == base
    tiny=[big"1e400",big"1e-400"]
    @test EXT._weighted_kde1d([0.,1.],tiny,(-3.,3.)) == EXT._weighted_kde1d([0.],[1.],(-3.,3.))
    second=EXT._weighted_kde1d(x,w,(-3.,3.))
    @test base.x !== second.x && base.density !== second.density
end

function exact_quantile(x,w,p,weighted)
    pairs=sort(collect(zip(Rational{BigInt}.(x),Rational{BigInt}.(w))))
    if !weighted
        rank=1+p*(length(x)-1);i=min(floor(Int,rank),length(x)-1)
        return pairs[i][1]+(rank-i)*(pairs[i+1][1]-pairs[i][1])
    end
    target=p*sum(last,pairs)+(1-p)*last(first(pairs))
    cumulative=0//BigInt(1)
    for i in eachindex(pairs)
        previous=cumulative;cumulative+=pairs[i][2]
        if cumulative>target
            i==1 && return pairs[1][1]
            return pairs[i-1][1]+(target-previous)/pairs[i][2]*(pairs[i][1]-pairs[i-1][1])
        end
    end
    first(last(pairs))
end
function enclosure(x,w;weighted=!allequal(w))
    lo,hi=EXT._kde_iqr_bounds(x,w,Float64.(x),weighted)
    exact=exact_quantile(x,w,3//4,weighted)-exact_quantile(x,w,1//4,weighted)
    @test lo<=hi
    @test !isfinite(lo) || Rational{BigInt}(lo)<=exact
    @test !isfinite(hi) || exact<=Rational{BigInt}(hi)
    if iszero(lo) && iszero(hi)
        @test iszero(exact)
    end
    return lo,hi
end
@testset "Independent rational quartile enclosure" begin
    for n in 3:16, left in 0:n-2, right in left+1:n
        x=vcat(fill(-1.,left),zeros(right-left),fill(1.,n-right))
        enclosure(x,ones(n))
    end
    rng=MersenneTwister(0xc3a71f1)
    for _ in 1:200
        x=Float64.(rand(rng,-16:16,12))./16
        w=Float64.(rand(rng,1:8,12))./8
        enclosure(x,w)
    end
    for w in ([1.,1.,1.,1e-20,2/3],[1.,1.,1.,8e-16,2/3],
            [1.,1.,1.,nextfloat(1/3),1/3],
            [big"1",big"1.0000000000000000000001",big"1",big"0.33333333333333333333333",big"0.33333333333333333333333"])
        enclosure([0.,0.,0.,0.1,1.],w)
        @test_throws "bounded estimate" EXT._kde_bounded_bandwidth([0.,0.,0.,0.1,1.],w)
    end
    # Include weight conversion whose lower enclosure reaches zero.
    enclosure([0.,0.,0.5,1.],[big"1",big"1",BigFloat(nextfloat(0.))/2,big"1"])
    enclosure(vcat(zeros(6),nextfloat(0.),1.),ones(8))
    x=[0.,0.,0.,0.,4.];w=[2.,3.,4.,5.,1.]
    @test enclosure(x,w)==(0.,0.)
    for perm in ([1,2,3,4,5],[4,3,2,1,5],[5,2,4,1,3])
        xx,ww=x[perm],w[perm]
        @test EXT._kde_bounded_bandwidth(xx,ww) ≈ bandwidth_reference(xx,ww) rtol=3e-15
        @test bandwidth_reference(xx,ww)==bandwidth_reference(x,w)
    end
    # Two distinct original plateaus can share the same normalized Float64
    # coordinate; equal numerical bounds alone must not certify zero IQR.
    small=big"1e-30"
    original=[big"0",big"1",big"1",big"1",1+small,1+small,1+small,big"2"]
    lo,hi=EXT._kde_iqr_bounds(original,ones(8),Float64.(original),false)
    @test (lo,hi)!= (0.,0.)
    @test !isfinite(lo) || lo<=0
    @test !isfinite(hi) || hi>0
    for values in ([-0.,0.],[-0.,0.,0.,-0.]), weights in (ones(length(values)),collect(1.:length(values)))
        @test EXT._kde_bounded_bandwidth(values,weights) ≈ bandwidth_reference(values,weights) rtol=3e-15
    end
    println("RANK seed=0xc3a71f1")
end

@testset "Zero IQR requires a genuine plateau" begin
    a=vcat(zeros(6),nextfloat(0.),1.); wa=ones(8)
    b=[0.,0.,0.,0.5,1.]; wb=[1.,1.,1.,nextfloat(1/3),1/3]
    @test Rational{BigInt}(sum(wb)) != sum(Rational{BigInt}.(wb))
    target=3//4*sum(Rational{BigInt}.(wb))+1//4
    @test target-3 == 1//BigInt(72057594037927936)
    for (x,w) in ((a,wa),(b,wb))
        @test_throws "bounded estimate" EXT._kde_bounded_bandwidth(x,w)
        @test_throws "bounded estimate" EXT._weighted_kde1d(x,w,(0.,1.))
    end
end

@testset "Positive IQR rank uncertainty" begin
    x=[0.,0.,0.,0.1,1.]
    for small in (1e-20,8e-16)
        w=[1.,1.,1.,small,2/3]
        expected=bandwidth_reference(x,w)
        actual=EXT._kde_bandwidth(x,w)
        @test abs(actual/expected-1)>0.1
        @test_throws "bounded estimate" EXT._kde_bounded_bandwidth(x,w)
        @test_throws "bounded estimate" EXT._weighted_kde1d(x,w,(0.,1.))
    end
end

@testset "Shifted maintained-control arithmetic" begin
    for offset in (1e100,1e95)
        x=offset .+ 1e90.*[0.1,0.2,0.4,0.8];w=ones(4)
        actual=EXT._kde_bandwidth(x,w); expected=bandwidth_reference(x,w)
        println("SHIFTED offset=",offset," h=",actual," exact_reference=",expected," relative_error=",actual/expected-1)
        if offset==1e100
            @test abs(actual/expected-1)>2.0^-20
            @test_throws "bounded estimate" EXT._weighted_kde1d(x,w,(offset,offset+1e90))
            local_x=x.-first(x)
            @test EXT._kde_bounded_bandwidth(local_x,w) ≈ expected rtol=3e-15
        else
            @test EXT._kde_bounded_bandwidth(x,w) ≈ expected rtol=2.0^-20
            @test all(isfinite,EXT._weighted_kde1d(x,w,(offset,offset+1e90)).density)
        end
    end
end

@testset "Generated represented-value bandwidth controls" begin
    rng=MersenneTwister(0x09bada55)
    accepted=0;rejected=0
    for T in (Float32,Float64), offset in (0.,1e4,1e7), scale in (0.25,1.,20.), repeat in 1:4
        x=T.(offset .+ scale.*randn(rng,17))
        w=repeat%2==0 ? ones(17) : Float64.(rand(rng,1:8,17))
        before=(copy(x),copy(w)); expected=bandwidth_reference(x,w)
        result=try EXT._kde_bounded_bandwidth(x,w) catch e; e end
        if result isa Exception
            @test result isa ArgumentError
            @test occursin("bounded estimate",sprint(showerror,result))
            rejected+=1
        else
            @test result == Float64(EXT._kde_bandwidth(x,w))
            @test abs(result/expected-1)<=2.0^-19
            accepted+=1
        end
        @test isequal((x,w),before)
    end
    @test accepted>0 && rejected>0
    println("GENERATED accepted=",accepted," rejected=",rejected," seed=0x09bada55")
end

samples(data; weights=ones(size(data,2))) = DensitySampleVector(
    v=EXT.VectorOfSimilarVectors(copy(data)),logd=zeros(size(data,2)),weight=copy(weights))
function reject(f)
    err=try f(); nothing catch e; e end
    @test !isnothing(err)
    @test occursin("KDE: bounded estimate",sprint(showerror,err))
end
function descendants(axis,T)
    found=Any[]
    function visit(p)
        p isa T && push!(found,p)
        foreach(visit,p.plots)
    end
    foreach(visit,axis.scene.plots)
    found
end
cell(fig,i,j)=only(filter(p->p isa Axis,contents(fig[1,1][i,j])))
function reference_curve(grid,x,w,h,bounds)
    out=zeros(length(grid));mass=0.
    for (v,weight) in zip(Float64.(x),w)
        images=[v]
        isfinite(bounds[1]) && v-bounds[1]<4h && push!(images,2bounds[1]-v)
        isfinite(bounds[2]) && bounds[2]-v<4h && push!(images,2bounds[2]-v)
        for center in images
            mass+=weight*(cdf(Normal(),(last(grid)-center)/h)-cdf(Normal(),(first(grid)-center)/h))
            out .+= [weight*exp(-0.5*((t-center)/h)^2)/(sqrt(2pi)*h) for t in grid]
        end
    end
    out ./ mass
end
@testset "MR09 public paths, graph and ownership" begin
    for x in (fill(0.1f0,10),Float32[2^24,2^24,2^24,2^24,2^24+4]), recipe in (KDE1D,QuantileKDE1D)
        bounds=(Float64(minimum(x))-2,Float64(maximum(x))+2)
        recipes=(upper=Scatter2D,diagonal=recipe,lower=Scatter2D)
        source=samples(reshape(x,1,:)); before=deepcopy(source)
        for ordinary in (false,true)
            fig=Figure()
            try
                reject() do
                    ordinary ? Makie.plot(fig[1,1],source;recipes,vsel=[1],N_max=1,support=[bounds]) :
                        bat_makie_plot(source,recipes,[1],1;support=[bounds],display=false)
                end
            finally
                empty!(fig)
                ordinary || empty!(Makie.current_figure())
            end
            wide=samples(reshape(Float64.(x),1,:))
            fig=ordinary ? Figure() : bat_makie_plot(wide,recipes,[1],1;support=[bounds],display=false)
            wrapper=nothing
            try
                ordinary && (wrapper=Makie.plot(fig[1,1],wide;recipes,vsel=[1],N_max=1,support=[bounds]))
                @test !isempty(Makie.colorbuffer(fig))
                if recipe===KDE1D
                    curves=filter(p -> length(p[1][])>2 && all(>(0),diff(getindex.(p[1][],1))),
                        descendants(cell(fig,1,1),Makie.Lines))
                    line=only(curves)[1][]
                    grid=Float64[p[1] for p in line]; ys=Float64[p[2] for p in line]
                    # Exact represented constant or IQR-zero corrected variance.
                    h=allequal(x) ? 0.9length(x)^(-0.2) : 0.9sqrt(16/5)*5^(-0.2)
                    expected=reference_curve(grid,x,ones(length(x)),h,bounds)
                    @test maximum(abs.(ys.-expected))/maximum(expected) < 0.005
                end
            finally
                empty!(fig); isnothing(wrapper) || Makie.free(wrapper)
            end
        end
        @test source==before
    end

    bad=Float32[2^24,2^24,2^24,2^24,2^24+4]
    other=Float32[-2,-1,0,1,2]
    for recipe in (KDE2D,QuantileKDE2D), swapped in (false,true), ordinary in (false,true)
        data=permutedims(hcat(bad,other)); support=[(-Inf,Inf),(-3.,3.)]
        swapped && (data=reverse(data;dims=1);support=reverse(support))
        recipes=(upper=recipe,diagonal=QuantileHist1D,lower=recipe)
        source=samples(data); fig=Figure()
        try
            reject() do
                ordinary ? Makie.plot(fig[1,1],source;recipes,vsel=[1,2],N_max=2,support) :
                    bat_makie_plot(source,recipes,[1,2],2;support,display=false)
            end
        finally
            empty!(fig); ordinary || empty!(Makie.current_figure())
        end
    end
    # Safe Float32 pair is still admitted. Pair transpose and native cells retain
    # their established values; this is a rendering control, not a mass oracle.
    gooddata=Float32[-2 -2 0 2 2;2 0 -2 2 -2]
    for recipe in (KDE2D,QuantileKDE2D), ordinary in (false,true)
        recipes=(upper=recipe,diagonal=QuantileHist1D,lower=recipe); source=samples(gooddata)
        support=[(-3.,3.),(-3.,3.)]; wrapper=nothing
        fig=ordinary ? Figure() : bat_makie_plot(source,recipes,[1,2],2;support,display=false)
        try
            ordinary && (wrapper=Makie.plot(fig[1,1],source;recipes,vsel=[1,2],N_max=2,support))
            up=only(descendants(cell(fig,1,2),Makie.Heatmap)); down=only(descendants(cell(fig,2,1),Makie.Heatmap))
            @test up[1][]==down[2][] && up[2][]==down[1][]
            @test isequal(up[3][],permutedims(down[3][]))
            @test !isempty(Makie.colorbuffer(fig))
        finally
            empty!(fig); isnothing(wrapper)||Makie.free(wrapper)
        end
    end

    good=samples(reshape(Float32[-2,-2,0,2,2],1,:)); badsample=samples(reshape(fill(0.1f0,10),1,:))
    recipes=(upper=Scatter2D,diagonal=KDE1D,lower=Scatter2D)
    graph,_=EXT._setup_static_graph(good,recipes,[1],1,BAT._default_makie_triagonal_config(),
        BAT._default_makie_diagonal_config();support=[(-3.,3.)])
    fig=Figure(); unrelated=Figure();lines!(Axis(unrelated[1,1]),[0.,1.],[1.,0.])
    snapshot=copy(Makie.colorbuffer(unrelated)); vis=BATVisualizer(BATMakieVisualization())
    EXT._claim_live_visualizer!(vis)
    try
        EXT._bind_interactive_grid!(fig,EXT._init_gridlayout!(graph,1))
        for data in (good[1:0],good)
            Makie.update!(graph;samples=[[data]],current_idxs=[[length(data)]])
            @test !isempty(Makie.colorbuffer(fig))
        end
        @test EXT._with_live_view(vis) do
            EXT._attach_live_view!(vis,fig)
        end
        EXT._start_live_listener!(vis,()->nothing,()->nothing)
        @test !EXT._with_live_view(vis) do
            Makie.update!(graph;samples=[[badsample]],current_idxs=[[length(badsample)]])
        end
        BAT.finish_visualizer!(vis;success=true)
        status=BAT.visualizer_status(vis)
        @test status.outcome===:failed
        @test !status.data_complete && !status.presentation_valid
        @test status.quiesced && status.view_disposed && !status.cleanup_pending
        @test status.error.phase===:view
        @test isempty(fig.content)
        called=Ref(false)
        @test !EXT._with_live_view(vis) do
            called[]=true
        end
        @test !called[]
        @test Makie.colorbuffer(unrelated)==snapshot
    finally
        BAT.request_visualizer_stop!(vis);BAT.finish_visualizer!(vis;success=false)
        empty!(fig);empty!(unrelated)
    end
end

function primitives(data, recipe; weights=ones(size(data, 2)), levels=nothing)
    one_dimensional = recipe isa Union{KDE1D,QuantileKDE1D}
    config = one_dimensional ? DIAG : PAIR
    support = one_dimensional ? (-3., 3.) : ((-3., 3.), (-3., 3.))
    config = merge(config, (; support))
    isnothing(levels) || (config = merge(config, (; levels)))
    return EXT.compute_plotting_primitives(view(data, :, :), view(weights, :), recipe,
        EXT.LiveRecipe(), EXT.LiveCell(), config)
end

@testset "Bounded bandwidth demand and caller ownership" begin
    @testset "Admission precedes masks; empty and inactive sentinels survive" begin
        for recipe in (KDE1D(), QuantileKDE1D(), KDE2D(), QuantileKDE2D())
            n = recipe isa Union{KDE1D,QuantileKDE1D} ? 1 : 2
            bad = fill(0.1f0, n, 10)
            for levels in (nothing, Float64[])
                @test_throws "bounded estimate" primitives(bad, recipe; levels)
            end
            @test all(isempty, values(primitives(bad, recipe; weights=zeros(10))))
            @test all(isempty, values(primitives(bad[:, 1:0], recipe)))
            config = n == 1 ? DIAG : PAIR
            for (rs, cs) in ((EXT.DeadRecipe(), EXT.LiveCell()),
                    (EXT.LiveRecipe(), EXT.DeadCell()))
                result = EXT.compute_plotting_primitives(view(bad, :, :), view(ones(10), :),
                    recipe, rs, cs, config)
                @test all(isempty, values(result))
            end
        end
        good = Float32[-2 -2 0 2 2; 2 0 -2 2 -2]
        q1 = primitives(good[1:1, :], QuantileKDE1D(); levels=Float64[])
        q2 = primitives(good, QuantileKDE2D(); levels=Float64[])
        @test isempty(q1.polys) && !isempty(q1.full_line)
        @test !isempty(q2.color_grid) && all(c -> iszero(c.alpha), q2.color_grid)
    end

    @testset "Ordinary rejected update disposes and recreates" begin
        good = samples(reshape(Float32[-2, -2, 0, 2, 2], 1, :))
        bad = samples(reshape(fill(0.1f0, 10), 1, :))
        input = Observable(good)
        recipes = (upper=Scatter2D, diagonal=KDE1D, lower=Scatter2D)
        fig = Figure()
        wrapper = nothing
        try
            wrapper = Makie.plot(fig[1, 1], input; recipes, vsel=[1], N_max=1, support=[(-3., 3.)])
            @test !isempty(Makie.colorbuffer(fig))
            reject() do
                input[] = bad
                Makie.colorbuffer(fig)
            end
        finally
            empty!(fig)
            isnothing(wrapper) || Makie.free(wrapper)
        end
        @test isempty(fig.content)
        @test isempty(Makie.Observables.listeners(input))
        fresh = Figure()
        wrapper = nothing
        try
            wrapper = Makie.plot(fresh[1, 1], good; recipes, vsel=[1], N_max=1, support=[(-3., 3.)])
            @test !isempty(Makie.colorbuffer(fresh))
        finally
            empty!(fresh)
            isnothing(wrapper) || Makie.free(wrapper)
        end
    end

    @testset "Construction rollback and public recipe control" begin
        other = Figure()
        lines!(Axis(other[1, 1]), [0., 1.], [1., 0.])
        before = copy(Makie.colorbuffer(other))
        font = Makie.to_value(Makie.theme(:fontsize))
        background = Makie.to_value(Makie.theme(:backgroundcolor))
        bad = samples(reshape(fill(0.1f0, 10), 1, :))
        kde = (upper=Scatter2D, diagonal=KDE1D, lower=Scatter2D)
        hist = (upper=Scatter2D, diagonal=QuantileHist1D, lower=Scatter2D)
        try
            reject(() -> bat_makie_plot(bad, kde, [1], 1; support=[(-3., 3.)], display=false))
            failed = Makie.current_figure()
            @test failed !== other && isempty(failed.content)
            @test isempty(failed.scene.deregister_callbacks)
            @test Makie.colorbuffer(other) == before
            @test Makie.to_value(Makie.theme(:fontsize)) == font
            @test Makie.to_value(Makie.theme(:backgroundcolor)) == background
            empty!(failed)
            fig = bat_makie_plot(bad, hist, [1], 1; support=[(-3., 3.)], display=false)
            menu = only(filter(x -> x isa Menu && any(o -> last(o) === KDE1D, x.options[]), fig.content))
            try
                @test !isempty(Makie.colorbuffer(fig))
                reject(() -> (menu.selection[] = KDE1D))
            finally
                empty!(fig)
            end
            @test isempty(fig.content)
            @test isempty(Makie.Observables.listeners(menu.selection))
            @test Makie.colorbuffer(other) == before
            @test Makie.to_value(Makie.theme(:fontsize)) == font
            @test Makie.to_value(Makie.theme(:backgroundcolor)) == background
            fresh = bat_makie_plot(bad, hist, [1], 1; support=[(-3., 3.)], display=false)
            try
                @test !isempty(Makie.colorbuffer(fresh))
            finally
                empty!(fresh)
            end
        finally
            empty!(other)
        end
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

function wait_done(task; seconds=30.0)
    @test task isa Task
    task isa Task || error("Missing owned task in bounded KDE failure fixture")
    done = timedwait(() -> istaskdone(task), seconds; pollint=0.005) === :ok
    @test done
    done || error("Bounded KDE failure fixture did not terminate its owned task")
    @test !istaskfailed(task)
    wait(task)
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
        init=MCMCChainPoolInit(nsteps_init=20), callback=callback,
        convergence=AssumeConvergence(), adaptive_transform=BAT.NoAdaptiveTransform(),
        transform_tuning=NoMCMCTransformTuning(),
        burnin=MCMCMultiCycleBurnin(nsteps_per_cycle=12, max_ncycles=1, nsteps_final=3))
end

@testset "Bounded bandwidth live rejection preserves sampling and releases its owner" begin
    vis = BATVisualizer(configuration(RECIPES))
    plain_context = BATContext(rng=BAT.Philox4x((144, 72)))
    live_context = BATContext(rng=BAT.Philox4x((144, 72)), visualizer=vis)
    target = NamedTupleDist(a=Uniform(-2., 4.), b=Uniform(10., 18.))
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
            @test graph[:support_lo][] == [-2., 10.]
            @test graph[:support_hi][] == [4., 18.]
            good = graph[EXT.primitive_symbol(KDE2D(), (2, 1))][]
            @test !isempty(good.x)
            listener[] = lock(() -> vis.content.lifecycle.task, vis.content.buffer_lock)
            callback_steps[] = [state.chain_state.stepno for state in states]
            injected[] = true
            # Only a private display copy changes. Metadata, timing, weights,
            # lengths and storage types remain the sampler's valid values.
            payload = deepcopy(graph[:samples][])
            for walkers in payload, walker in walkers
                walker.v.data[1, :] .= 1e-170 .* (mod.(1:length(walker), 5) .- 2)
            end
            x = reduce(vcat, [collect(w.v.data[1, :]) for ws in payload for w in ws])
            weights = reduce(vcat, [collect(w.weight) for ws in payload for w in ws])
            expected = bandwidth_reference(x, weights)
            @test 0 < expected < 1e-169
            @test abs(EXT._kde_bandwidth(x, weights) / expected - 1) > 0.1
            @test_throws "bandwidth reliably" EXT._weighted_kde1d(x, weights, (-2., 4.))
            logger = Test.TestLogger(min_level=Logging.Error)
            @test !with_logger(logger) do
                EXT._live_graph_update!(vis; publication=true, samples=payload)
            end
            try
                record = only(logger.logs)
                @test record.kwargs[:phase] === :view
                message = sprint(showerror, first(record.kwargs[:exception]))
                @test occursin("KDE: bounded estimate", message)
                @test occursin("bandwidth reliably", message)
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
