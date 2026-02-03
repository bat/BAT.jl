@recipe(Scatter2D, x, y) do scene
    Attributes(
        weights = nothing,
        color = Makie.wong_colors()[1],
        alpha = 1.0,
        markersize = 2.0
    )
end

function Makie.plot!(p::Scatter2D)
    real_markersize = lift(p.weights, p.markersize) do w, base_size
        if isnothing(w) || w isa Number
            return base_size
        else
            w_vals = w isa AbstractWeights ? w.values : w

            mean_w = mean(w_vals)
            if mean_w <= 0
                return base_size
            end

            return sqrt.(w_vals ./ mean_w) .* base_size
        end
    end

    scatter!(p, p[1], p[2];
        color = p.color,
        alpha = p.alpha,
        markersize = real_markersize
    )

    return p
end

