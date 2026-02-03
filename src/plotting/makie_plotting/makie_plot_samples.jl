using Makie, BAT, Statistics

@recipe(BATPlot, samples) do scene
    Attributes(
        vsel = collect(1:5),
        diagonal = QuantileHist1D,
        lower = QuantileHist2D,
        upper = nothing,
        labels = nothing,
        link_axes = true,
        bins = 200,
        closed = :left,
        filter = false
    )
end

function Makie.plot!(p::BATPlot)
    vs = varshape(p.samples[])
    indices = lift(p.vsel) do vsel_inp
        vsel = collect(vsel_inp)
        if !(vsel isa Vector{<:Integer})
            vsel = asindex.(Ref(vs), vsel_inp)
            vsel = reduce(vcat, vsel)
        end
        return vsel[vsel .<= totalndof(vs)]
    end

    n_params = lift(idxs -> length(idxs), indices)

    ax_grid = lift(n_params, indices) do n, idxs
        empty!(p.layout)

        axs = Matrix{Axis}(undef, n, n)

        # TODO
        # Catch the case that the user provided lables, but the vsel change during interactive plot
        # Write method for getstring() that does xlabel = ["v$i" for i in vsel] for unshaped samples
        xlbls = isnothing(p.labels[]) ? getstring.(Ref(p.samples[]), idxs) : p.labels[]
        ylbls = ["p($l)" for l in xlbls]

        for i in 1:n, j in 1:n
            if i == j || (i > j && !isnothing(p.lower[])) || (j > i && !isnothing(p.upper[]))
                ax = Axis(p.layout[i, j])
                axs[i, j] = ax
                apply_decorations!(ax, i, j, n, xlbls[j], ylbls[i])
            end
        end
        return axs
    end

    # TODO
    # Global bin calculation
    # Reactive Matrix of number of bins for each sub-plot. Pass to MarginalDist below

    on(ax_grid, indices, p.bins, p.closed, p.filter) do axs, idxs, n_bins, clsd, fltr
        n = n_params[]

        for i in 1:n, j in 1:n
            isassigned(axs, i, j) || continue
            ax = axs[i, j]

            # TODO
            # More complex recipe logic for dynamic recipies
            recipe = if i == j
                p.diagonal[]
            elseif i > j
                p.lower[]
            else
                p.upper[]
            end

            plot_idxs = (i == j) ? idxs[i] : (idxs[i], idxs[j])
            recipe(
                ax,
                smpls,
                plot_idxs;
                nbins = n_bins,
                closed = clsd,
                filter = fltr
            )
        end

        if p.link_axes[]
            link_axes!(axs)
        end
    end

    return p
end

function apply_decorations!(
    ax::Axis,
    i::Integer,
    j::Integer,
    n::Integer,
    xlabel::String,
    ylabel::String
)
    if i < n
        hidexdecorations!(
            ax,
            grid = false,
            ticks = true,
            ticklabels = true
        )
    else
        ax.xlabel = xlabel
    end

    if j > 1
        hideydecorations!(
            ax,
            grid = false,
            ticks = true,
            ticklabels = true
        )
    else
        ax.ylabel = ylabel
    end

    ax.xticklabelrotation = π/4
    ax.xtickalign = 1
    ax.ytickalign = 1

    if i < j
        ax.xaxisposition = :top
    end
end


const BATMakieRecipe = Union{
    Hist1D, Hist2D,
    QuantileHist1D, QuantileHist2D,
    Hexbin2D,
    KDE1D, KDE2D,
    QuantileKDE1D, QuantileKDE2D,
    Scatter2D,
    Cov2D,
    Std1D, Std2D,
    Mean1D, Mean2D,
    ErrorBars1D, ErrorBars2D,
    PDF1D
}

function Makie.convert_arguments(
    ::Type{<:BATMakieRecipe},
    samples::DensitySampleVector,
    idxs::Integer,
    args...
)
    return (samples, idxs, args...)
end

