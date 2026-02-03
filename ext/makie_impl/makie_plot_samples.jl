
function bat_makie_plot(samples; kwargs...)
    fig = Figure()
    bat_makie_plot!(fig, samples; kwargs...)
    return fig
end

function bat_makie_plot!(
    fig::Figure,
    samples;
    vsel = Observable(collect(1:5)),
    diagonal = Observable(QuantileHist1D),
    lower = Observable(QuantileHist2D),
    upper = Observable(nothing),
    labels = Observable(nothing),
    link_axes = Observable(true),
    nbins = Observable(200),
    closed = Observable(:left),
    filter = Observable(false)
)
    vs = varshape(samples[])
    indices = lift(vsel) do vsel_inp
        vsel = collect(vsel_inp)
        if !(vsel isa Vector{<:Integer})
            vsel = asindex.(Ref(vs), vsel_inp)
            vsel = reduce(vcat, vsel)
        end
        return vsel[vsel .<= totalndof(vs)]
    end

    n_params = lift(idxs -> length(idxs), indices)

    ax_grid = lift(n_params, indices) do n, idxs
        # empty!(fig.layout)

        axs = Matrix{Axis}(undef, n, n)

        # TODO
        # Catch the case that the user provided lables, but the vsel change during interactive plot
        # Write method for getstring() that does xlabel = ["v$i" for i in vsel] for unshaped samples
        xlbls = isnothing(labels[]) ? getstring.(Ref(samples[]), idxs) : labels[]
        ylbls = ["p($l)" for l in xlbls]

        # TODO
        # Global bin calculation
        # Reactive Matrix of number of bins for each sub-plot. Pass to MarginalDist below
        for i in 1:n, j in 1:n
            if i == j || (i > j && !isnothing(lower[])) || (j > i && !isnothing(upper[]))
                ax = Axis(fig.layout[i, j])
                axs[i, j] = ax
                apply_decorations!(ax, i, j, n, xlbls[j], ylbls[i])

                # TODO
                # More complex recipe logic for dynamic recipies
                recipe = if i == j
                    diagonal[]
                elseif i > j
                    lower[]
                else
                    upper[]
                end

                plot_idxs = lift(indices) do idxs
                    if (i == j)
                        idxs[i]
                    else
                        (idxs[i], idxs[j])
                    end
                end
                
                # global gs = (ax, recipe, samples_loc, plot_idxs, nbins, closed, filter)
                # BREAK
                plot!(
                    ax,
                    recipe,
                    samples,
                    plot_idxs;
                    nbins = nbins,
                    closed = closed,
                    filter = filter
                )
            end
        end

        if link_axes[]
            link_axes!(axs)
        end
        return axs
    end

    return fig
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
    Errorbars1D, Errorbars2D,
    PDF1D
}

function Makie.convert_arguments(
    ::Type{<:BATMakieExt.BATMakieRecipe},
    samples::Observable,
    idxs::Observable;
    args...
)
    return (samples, idxs)
end

function Makie.convert_arguments(
    ::Type{<:BATMakieExt.BATMakieRecipe},
    samples,
    idxs;
    args...
)
    return (samples, idxs)
end

