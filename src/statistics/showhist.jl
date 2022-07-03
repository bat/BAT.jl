# This file is a part of BAT.jl, licensed under the MIT License (MIT).

default_unicode_bars = ['\u2800', '▁', '▂', '▃', '▄', '▅', '▆', '▇', '█']

function showhist_unicode(io::IO, h::Histogram{<:Real,1}; bar_symbols::AbstractVector{Char} = default_unicode_bars)
    @argcheck first(h.edges) isa AbstractRange

    edge = first(h.edges)
    W = h.weights
    X = (edge[1:(end - 1)] + edge[2:end]) / 2

    symidxs = eachindex(bar_symbols)
    norm_factor = length(symidxs) / maximum(W)
    get_sym_idx(x) = isnan(x) ? 1 : clamp(first(symidxs) + floor(Int, norm_factor * x), first(symidxs), last(symidxs))

    mean_idx = StatsBase.binindex(h, mean(X, FrequencyWeights(W)))
    median_idx = StatsBase.binindex(h, median(X, FrequencyWeights(W)))
    
    print(io, replace(lpad(@sprintf("%8.3g", minimum(edge)), 8, '\u2800'), " " => "\u2800"))
    print(io, h.closed == :left ? "[" : "]")
    for (i) in eachindex(W)
        color = if i == mean_idx == median_idx
            :cyan
        elseif i == mean_idx
            :green
        elseif i == median_idx
            :blue
        else
            :default
        end
        sym = bar_symbols[get_sym_idx(W[i])]
        printstyled(io, sym; color=color)
    end
    print(io, h.closed == :right ? "]" : "[")
    print(io, replace(lpad(@sprintf("%-8.3g", maximum(edge)), 8, '\u2800'), " " => "\u2800"))
end
