# This file is a part of BAT.jl, licensed under the MIT License (MIT).
module TestMakieViewState
using BAT, Test, Makie, CairoMakie
const EXT=Base.get_extension(BAT,:BATMakieExt)
const RECIPES=(upper=Hist2D,diagonal=Hist1D,lower=Hist2D)

cell(fig,i,j)=only(filter(x -> x isa Axis,contents(fig[1,1][i,j])))
cells(fig)=[cell(fig,i,j) for i in 1:2,j in 1:2]
function descendants(axis,T)
    result=Any[]
    function visit(plot)
        plot isa T && push!(result,plot)
        foreach(visit,plot.plots)
    end
    foreach(visit,axis.scene.plots)
    return result
end
function stable_cells(fig,original)
    @test length(filter(x -> x isa Axis,fig.content))==4
    @test all(cells(fig)[i]===original[i] for i in eachindex(original))
    @test all(axis -> axis in fig.content,original)
end
function select!(menu,recipe)
    index=findfirst(option -> option[2]===recipe,menu.options[])
    @test !isnothing(index)
    menu.i_selected[]=index
end
function pan!(axis)
    r=axis.targetlimits[]
    target=typeof(r)(r.origin .+ 0.2 .* r.widths,0.5 .* r.widths)
    axis.targetlimits[]=target
    return target
end
function manual_limits!(axis)
    r=axis.targetlimits[]
    lo=r.origin .+ 0.2 .* r.widths
    hi=r.origin .+ 0.7 .* r.widths
    limits!(axis,lo[1],hi[1],lo[2],hi[2])
    return (limits=deepcopy(axis.limits[]),target=axis.targetlimits[])
end
function histogram(axis,values,weights)
    @test length(axis.scene.plots)==2 # no stale KDE plots after a round trip
    bars=only(descendants(axis,Makie.BarPlot))[1][]
    edges=Float64[p[1] for p in only(descendants(axis,Makie.Stairs))[1][]]
    mass=[sum((weights[k] for k in eachindex(values) if edges[i]<=values[k]<edges[i+1]);init=0.0)
        for i in 1:length(edges)-1]
    @test sum(mass)==sum(weights)
    @test Float64[p[2] for p in bars] ≈ mass ./ (sum(weights).*diff(edges)) rtol=3e-5 atol=1e-6
    @test axis.targetlimits[].widths[1]>0 && axis.targetlimits[].widths[2]>0
    return copy(bars)
end
function heat(axis)
    plot=only(descendants(axis,Makie.Heatmap))
    return (x=copy(plot[1][]),y=copy(plot[2][]),z=copy(plot[3][]))
end

@testset "Interactive grid cell state" begin
    v2=[0.137,0.273,0.419,0.637,0.781,0.913]
    v3=[0.273,0.419,0.137,0.781,0.913,0.637]
    weights=[1.0,1.0,1.0,2.0,2.0,2.0]
    samples=DensitySampleVector(v=[[Float64(i),v2[i],v3[i]] for i in eachindex(v2)],
        logd=zeros(6),weight=weights)
    before=deepcopy(samples)
    fig=bat_makie_plot(samples,RECIPES,[2,3],2;display=false)
    unrelated=Figure()
    other=Axis(unrelated[1,1],title="unrelated")
    scatter!(other,[1.0,2.0],[3.0,4.0])
    try
        original=cells(fig)
        diagonal1,diagonal2=original[1,1],original[2,2]
        upper,lower=original[1,2],original[2,1]
        # Menus are intentionally constructed bottom-first by _build_fig.
        lower_menu,diag_menu,upper_menu=filter(x -> x isa Menu,fig.content)
        toggles=filter(x -> x isa Toggle,fig.content)
        slider=only(filter(x -> x isa IntervalSlider,fig.content))
        apply=only(filter(x -> x isa Button && x.label[]=="Apply",fig.content))
        all_button=only(filter(x -> x isa Button && x.label[]=="All",fig.content))
        initial_hist=histogram(diagonal1,v2,weights)
        initial_upper,initial_lower=heat(upper),heat(lower)
        @test isequal(initial_upper.z, permutedims(initial_lower.z))
        @test any(x -> isfinite(x) && x>0, initial_upper.z)
        stable_cells(fig,original)

        for _ in 1:2
            select!(diag_menu,KDE1D)
            @test isempty(descendants(diagonal1,Makie.BarPlot))
            curve=only(filter(p -> p isa Makie.Lines,diagonal1.scene.plots))[1][]
            @test !isempty(curve) && all(p -> all(isfinite,p),curve)
            stable_cells(fig,original)
            select!(diag_menu,Hist1D)
            @test histogram(diagonal1,v2,weights)==initial_hist
            stable_cells(fig,original)
        end

        # Finite pair domains do not change on a recipe switch. Both ordinary
        # pan state and explicit caller limits survive the corresponding update.
        target=pan!(upper)
        select!(upper_menu,KDE2D)
        @test upper.targetlimits[]==target
        @test all(x -> isfinite(x) || isnan(x),heat(upper).z)
        @test any(x -> isfinite(x) && x>0,heat(upper).z)
        select!(upper_menu,Hist2D)
        @test upper.targetlimits[]==target
        @test isequal(heat(upper),initial_upper)
        manual=manual_limits!(lower)
        select!(lower_menu,Scatter2D)
        @test only(descendants(lower,Makie.Scatter))[1][] ≈ Point2f.(v2,v3)
        @test lower.limits[]==manual.limits && lower.targetlimits[]==manual.target
        select!(lower_menu,Hist2D)
        @test isequal(heat(lower),initial_lower)
        @test lower.limits[]==manual.limits && lower.targetlimits[]==manual.target
        select!(upper_menu,Scatter2D)
        @test only(descendants(upper,Makie.Scatter))[1][] ≈ Point2f.(v3,v2)
        select!(upper_menu,Hist2D)
        stable_cells(fig,original)

        # Caller labels cannot masquerade as a semantic selection change.
        label=upper.xlabel[]
        upper.xlabel="caller label"
        upper.title="slot upper"
        upper.backgroundcolor=:lavender
        target=pan!(upper)
        toggles[1].active[]=true
        @test upper.xlabel[]==label
        @test upper.title[]=="slot upper"
        @test upper.backgroundcolor[]==Makie.to_color(:lavender)
        @test upper.targetlimits[]==target
        @test !isempty(descendants(upper,Makie.LineSegments))

        # Draft state survives recipe changes; Apply changes data, not pair
        # domains. A changed diagonal height resets the entire diagonal view.
        slider.interval[]=(2,4)
        select!(diag_menu,KDE1D)
        select!(diag_menu,Hist1D)
        @test slider.interval[]==(2,4)
        @test toggles[1].active[]
        @test histogram(diagonal1,v2,weights)==initial_hist
        old_diagonal=manual_limits!(diagonal1)
        pair_target=pan!(upper)
        Makie.notify(apply.clicks)
        histogram(diagonal1,v2[2:4],weights[2:4])
        @test diagonal1.targetlimits[] != old_diagonal.target
        @test upper.targetlimits[]==pair_target
        @test apply.label[]=="Applied"
        select!(upper_menu,KDE2D)
        @test slider.interval[]==(2,4) && apply.label[]=="Applied"
        @test toggles[1].active[]
        select!(upper_menu,Hist2D)
        Makie.notify(all_button.clicks)
        @test histogram(diagonal1,v2,weights)==initial_hist
        toggles[1].active[]=false

        # Vars 2/3 have identical weighted histograms and finite domains. The
        # first slot stays active when var 2 is removed, but its semantic identity
        # changes; equal numeric limits must not preserve the previous zoom.
        computed=deepcopy(diagonal1.limits[])
        diagonal1.title="slot one"
        diagonal2.title="slot two"
        old=manual_limits!(diagonal1)
        removed_zoom=manual_limits!(diagonal2)
        checkboxes=filter(x -> x isa Checkbox,fig.content)
        # Triangular picker construction: (1,1),(2,1),(2,2),(3,1),(3,2),(3,3).
        checkboxes[3].checked[]=false
        @test occursin("3",string(diagonal1.xlabel[]))
        @test diagonal1.limits[]==computed
        @test diagonal1.targetlimits[] != old.target
        histogram(diagonal1,v3,weights)
        @test !diagonal2.xlabelvisible[] && !upper.xlabelvisible[] && !lower.ylabelvisible[]
        @test isempty(descendants(diagonal2,Makie.BarPlot))
        @test isempty(descendants(upper,Makie.Heatmap))
        @test diagonal1.title[]=="slot one" && diagonal2.title[]=="slot two"
        stable_cells(fig,original)
        checkboxes[3].checked[]=true
        @test occursin("2",string(diagonal1.xlabel[]))
        @test diagonal2.xlabelvisible[] && upper.xlabelvisible[] && lower.ylabelvisible[]
        @test diagonal2.targetlimits[] != removed_zoom.target
        histogram(diagonal1,v2,weights)
        histogram(diagonal2,v3,weights)
        stable_cells(fig,original)

        # Automatic density-height recipes must undo stale explicit overrides
        # after publishing their new plots, even if the generated limits match.
        select!(diag_menu,Errorbars1D)
        @test diagonal1.limits[][2]===nothing
        limits!(diagonal1,100,101,100,101)
        toggles[1].active[]=true
        @test diagonal1.limits[][2]===nothing
        point=only(only(filter(p -> p isa Makie.Scatter,diagonal1.scene.plots))[1][])
        r=diagonal1.targetlimits[]
        @test r.origin[2] <= point[2] <= r.origin[2]+r.widths[2]
        @test r.origin[1]<1 && r.origin[2]<100
        select!(diag_menu,Hist1D)
        toggles[1].active[]=false
        @test samples==before
        @test other.title[]=="unrelated" && length(other.scene.plots)==1
        stable_cells(fig,original)
        # Navigation assertions above intentionally precede render/export, whose
        # own Makie reset behavior is outside the BAT graph-update contract.
        @test !isempty(Makie.colorbuffer(fig))
        histogram(diagonal1,v2,weights)
    finally
        empty!(fig)
        empty!(unrelated)
    end

    # Supplementary ownership probe: observe the actual source subscription,
    # rather than only proving that a stale guarded callback declines admission.
    graph,_=EXT._setup_static_graph(samples,RECIPES,[2,3],2,
        BAT._default_makie_triagonal_config(),BAT._default_makie_diagonal_config())
    grid=EXT._init_gridlayout(graph,2)
    source=Makie.ComputePipeline.get_observable!(grid;use_deepcopy=false)
    owned=Figure()
    before_listeners=length(Makie.Observables.listeners(source))
    try
        initial=source[]
        initial_rows,initial_cols=copy(initial.rowsizes),copy(initial.colsizes)
        EXT._bind_interactive_grid!(owned,grid)
        @test length(Makie.Observables.listeners(source))==before_listeners+1
        handles=filter(h -> h.observable===source,owned.scene.deregister_callbacks)
        @test length(handles)==1
        held=cells(owned)
        diagonal_plots=copy(held[1,1].scene.plots)
        Makie.update!(graph;show_stats_upper=true)
        @test length(held[1,1].scene.plots)==length(diagonal_plots)
        @test all(a === b for (a,b) in zip(held[1,1].scene.plots,diagonal_plots))
        EXT._apply_vsel_to_graph!(graph,3,2,[3])
        @test initial.rowsizes==initial_rows && initial.colsizes==initial_cols
        @test source[] === graph[:gridlayout][] # no geometry deepcopy in the bridge
        valid=source[]
        limits_before=map(axis -> axis.targetlimits[],held)
        @test_throws ArgumentError source[]=Makie.SpecApi.GridLayout([Makie.SpecApi.Axis()])
        @test all(cells(owned)[i]===held[i] for i in eachindex(held))
        @test map(axis -> axis.targetlimits[],held)==limits_before
        source[]=valid
        empty!(owned)
        @test length(Makie.Observables.listeners(source))==before_listeners
        @test all(h -> all(pair -> last(pair)!==h.f,Makie.Observables.listeners(source)),handles)
        Makie.update!(graph;diagonal_recipe=KDE1D)
        @test isempty(owned.content)
        @test all(axis -> isempty(axis.scene.plots),held)
    finally
        empty!(owned)
    end
end
end
