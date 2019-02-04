# This file is a part of BAT.jl, licensed under the MIT License (MIT).

#doc
"""
    calculate_localmode(hist)

Calculates the modes of a 1d statsbase histogram.
A vector of the bin-center of the heighest bin(s) is(are) returned.
"""
function calculate_localmode(hist)
    maxima = maximum(hist.weights)
    maxima_idx = findall(x->x==maxima, hist.weights)
    edges = collect(hist.edges[1])

    modes = Vector{Real}(undef, length(maxima_idx))
    for (i, ind) in enumerate(maxima_idx)
        modes[i] = edges[ind] + 0.5 *abs(edges[ind+1] - edges[ind])
    end

    return modes
end


#todo
function calculate_localmode_2d(hist)
    maxima = maximum(hist.weights)
    maxima_idx = findall(x->x==maxima, hist.weights)
    edges = [collect(hist.edges[1]), collect(hist.edges[2])]

    modes = Vector{Vector{Real}}(undef, length(maxima_idx))
    for (i, ind) in enumerate(maxima_idx)

        x = edges[1][ind[1]] + 0.5 *abs(edges[1][ind[1]+1] - edges[1][ind[1]])
        y = edges[2][ind[2]] + 0.5 *abs(edges[2][ind[2]+1] - edges[2][ind[2]])
       modes[i] = [x , y]
    end

    return modes
end


