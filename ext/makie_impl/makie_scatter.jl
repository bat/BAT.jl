@recipe(Scatter2D, x, y) do scene
    Attributes(
        weights = nothing,
        filter = false,
        color = Makie.wong_colors()[1],
        alpha = 1.0,
        markersize = 2.0
    )
end

function Makie.plot!(p::Scatter2D)
    data = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = drop_low_weight_samples(marg_res)
        end

        flat_vals = flatview(unshaped.(marg_res).v)
        x = flat_vals[1, :]
        y = flat_vals[2, :]
        w = marg_res.weight

        return (x, y, w)
    end

    x_vals = lift(d -> d[1], data)
    y_vals = lift(d -> d[2], data)
    w_vals = lift(d -> d[3], data)

    real_markersize = lift(w_vals, p.markersize) do w, base_size
        if isempty(w)
            return base_size
        end

        if all(x -> x ≈ w[1], w)
            return base_size
        end

        mean_w = mean(w)
        if mean_w <= 0
            return base_size
        end

        return sqrt.(w ./ mean_w) .* base_size
    end

    scatter!(p, x_vals, y_vals;
        color = p.color,
        alpha = p.alpha,
        markersize = real_markersize
    )

    return p
end

