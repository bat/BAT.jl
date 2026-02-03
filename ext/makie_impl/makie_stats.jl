@recipe(Cov2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        nsigma = 1.0,
        color = :red,
        linestyle = :solid,
        linewidth = 2.0,
    )
end

function Makie.plot!(p::Cov2D)
    stats_data = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        flat_vals = flatview(unshaped.(marg_res).v)
        x_data = flat_vals[1, :]
        y_data = flat_vals[2, :]
        w_data = marg_res.weight

        return (x_data, y_data, w_data)
    end

    geometry = lift(stats_data, p.nsigma) do d, nsigma
        x, y, w = d
        data_matrix = hcat(x, y)
        w_obj = ProbabilityWeights(w)

        μ = mean(data_matrix, w_obj, dims=1) |> vec
        Σ = cov(data_matrix, w_obj)

        vals, vecs = eigen(Σ)
        stds = sqrt.(clamp.(vals, 0, Inf))

        θ = range(0, 2π, length=200)
        circle = [cos.(θ)'; sin.(θ)']

        ellipse_mat = vecs * nsigma * Diagonal(stds) * circle .+ μ
        ellipse_points = Point2f.(ellipse_mat[1, :], ellipse_mat[2, :])

        axes_segments = Point2f[]
        for i in 1:2
            direction = vecs[:, i] * stds[i] * 2 * nsigma
            p1 = Point2f(μ .- direction)
            p2 = Point2f(μ .+ direction)
            push!(axes_segments, p1, p2)
        end

        return (ellipse_points, axes_segments)
    end

    ellipse_pts = lift(g -> g[1], geometry)
    axes_pts    = lift(g -> g[2], geometry)

    lines!(p, ellipse_pts;
        color = p.color,
        linestyle = p.linestyle,
        linewidth = p.linewidth
    )

    linesegments!(p, axes_pts;
        color = p.color,
        linestyle = p.linestyle,
        linewidth = p.linewidth
    )

    return p
end


@recipe(Std1D, x) do scene
    Attributes(
        weights = nothing,
        filter = false,
        nsigma = 1.0,
        color = :red,
        linestyle = :solid,
        linewidth = 2.0
    )
end

function Makie.plot!(p::Std1D)
    positions = lift(p.samples, p.vsel, p.filter, p.nsigma) do smpls, vsel, f, nsigma
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        vals = flatview(unshaped.(marg_res).v)
        x_data = vec(vals)
        w_data = marg_res.weight

        w_obj = ProbabilityWeights(w_data)
        μ = mean(x_data, w_obj)
        σ = std(x_data, w_obj)

        return [μ - nsigma * σ, μ + nsigma * σ]
    end

    vlines!(p, positions;
        color = p.color,
        linestyle = p.linestyle,
        linewidth = p.linewidth
    )

    return p
end


@recipe(Std2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        nsigma = 1.0,
        color = :red,
        linestyle = :solid,
        linewidth = 2.0
    )
end

function Makie.plot!(p::Std2D)
    lines_data = lift(p.samples, p.vsel, p.filter, p.nsigma) do smpls, vsel, f, nsigma
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        flat_vals = flatview(unshaped.(marg_res).v)
        x_data = flat_vals[1, :]
        y_data = flat_vals[2, :]
        w_data = marg_res.weight

        w_obj = ProbabilityWeights(w_data)

        μ_x = mean(x_data, w_obj)
        μ_y = mean(y_data, w_obj)
        σ_x = std(x_data, w_obj)
        σ_y = std(y_data, w_obj)

        x_lines = [μ_x - nsigma * σ_x, μ_x + nsigma * σ_x]
        y_lines = [μ_y - nsigma * σ_y, μ_y + nsigma * σ_y]

        return (x_lines, y_lines)
    end

    v_pos = lift(d -> d[1], lines_data)
    h_pos = lift(d -> d[2], lines_data)

    vlines!(p, v_pos;
        color = p.color,
        linestyle = p.linestyle,
        linewidth = p.linewidth
    )

    hlines!(p, h_pos;
        color = p.color,
        linestyle = p.linestyle,
        linewidth = p.linewidth
    )

    return p
end


@recipe(Mean1D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        color = :black,
        linestyle = :dot,
        linewidth = 2.0
    )
end

function Makie.plot!(p::Mean1D)
    position = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        vals = flatview(unshaped.(marg_res).v)
        x_data = vec(vals)
        w_data = marg_res.weight

        w_obj = ProbabilityWeights(w_data)
        return [mean(x_data, w_obj)]
    end

    vlines!(p, position;
        color = p.color,
        linestyle = p.linestyle,
        linewidth = p.linewidth
    )

    return p
end


@recipe(Mean2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        color = :black,
        linestyle = :dot,
        linewidth = 2.0
    )
end

function Makie.plot!(p::Mean2D)
    positions = lift(p.samples, p.vsel, p.filter) do smpls, vsel, f
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        flat_vals = flatview(unshaped.(marg_res).v)
        x_data = flat_vals[1, :]
        y_data = flat_vals[2, :]
        w_data = marg_res.weight

        w_obj = ProbabilityWeights(w_data)
        μ_x = mean(x_data, w_obj)
        μ_y = mean(y_data, w_obj)

        return ([μ_x], [μ_y])
    end

    v_pos = lift(pos -> pos[1], positions)
    h_pos = lift(pos -> pos[2], positions)

    vlines!(p, v_pos;
        color = p.color,
        linestyle = p.linestyle,
        linewidth = p.linewidth
    )

    hlines!(p, h_pos;
        color = p.color,
        linestyle = p.linestyle,
        linewidth = p.linewidth
    )

    return p
end


@recipe(Errorbars1D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        nsigma = 1.0,
        y = 0.0,
        color = :red,
        linewidth = 2.0,
        whiskerwidth = 10
    )
end

function Makie.plot!(p::Errorbars1D)
    stats = lift(p.samples, p.vsel, p.filter, p.nsigma) do smpls, vsel, f, nsigma
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        vals = flatview(unshaped.(marg_res).v)
        x_data = vec(vals)
        w_data = marg_res.weight

        w_obj = ProbabilityWeights(w_data)
        μ = mean(x_data, w_obj)
        σ = std(x_data, w_obj)

        return (μ, σ * nsigma)
    end

    μ   = lift(s -> s[1], stats)
    err = lift(s -> s[2], stats)

    errorbars!(p,
        lift(m -> [m], μ),
        lift(y -> [y], p.y),
        lift(e -> [e], err);
        color = p.color,
        linewidth = p.linewidth,
        whiskerwidth = p.whiskerwidth,
        direction = :x
    )

    scatter!(p,
        lift(m -> [m], μ),
        lift(y -> [y], p.y);
        color = p.color,
        markersize = p.whiskerwidth
    )

    return p
end


@recipe(Errorbars2D, samples, vsel) do scene
    Attributes(
        weights = nothing,
        filter = false,
        nsigma = 1.0,
        color = :red,
        linewidth = 2.0,
        whiskerwidth = 10
    )
end

function Makie.plot!(p::Errorbars2D)
    stats = lift(p.samples, p.vsel, p.filter, p.nsigma) do smpls, vsel, f, nsigma
        marg = bat_marginalize(smpls, vsel)
        marg_res = marg.result

        if f
            marg_res = BAT.drop_low_weight_samples(marg_res)
        end

        flat_vals = flatview(unshaped.(marg_res).v)
        x_data = flat_vals[1, :]
        y_data = flat_vals[2, :]
        w_data = marg_res.weight

        w_obj = ProbabilityWeights(w_data)

        μ_x = mean(x_data, w_obj)
        μ_y = mean(y_data, w_obj)
        σ_x = std(x_data, w_obj)
        σ_y = std(y_data, w_obj)

        return (Point2f(μ_x, μ_y), σ_x * nsigma, σ_y * nsigma)
    end

    center_point = lift(s -> s[1], stats)
    err_x        = lift(s -> s[2], stats)
    err_y        = lift(s -> s[3], stats)

    errorbars!(p,
        lift(c -> [c[1]], center_point),
        lift(c -> [c[2]], center_point),
        lift(e -> [e], err_y);
        color = p.color,
        linewidth = p.linewidth,
        whiskerwidth = p.whiskerwidth,
        direction = :y
    )

    errorbars!(p,
        lift(c -> [c[1]], center_point),
        lift(c -> [c[2]], center_point),
        lift(e -> [e], err_x);
        color = p.color,
        linewidth = p.linewidth,
        whiskerwidth = p.whiskerwidth,
        direction = :x
    )

    scatter!(p, center_point;
        color = p.color,
        markersize = p.whiskerwidth
    )

    return p
end


# TODO REFACTOR
@recipe(PDF1D, distribution, vsel) do scene
    Attributes(
        color = Makie.wong_colors()[1],
        alpha = 1.0,
        filled = true,
        edge = false,
        strokecolor = Makie.wong_colors()[1],
        strokewidth = 1,
        npoints = 300
    )
end

# TODO REFACTOR
function Makie.plot!(p::PDF1D)
    curve_data = lift(p[1], p.npoints) do dist, n
        μ = mean(dist)
        σ = std(dist)

        if σ == 0
            x_min = μ - 1
            x_max = μ + 1
        else
            x_min = μ - 4σ
            x_max = μ + 4σ
        end

        x_grid = LinRange(x_min, x_max, n)
        y_grid = pdf.(dist, x_grid)

        return (x_grid, y_grid)
    end

    x = lift(d -> d[1], curve_data)
    y = lift(d -> d[2], curve_data)

    poly_points = lift(x, y) do x_val, y_val
        vcat(
            Point2f.(x_val, y_val),
            Point2f.(reverse(x_val), 0.0)
        )
    end

    poly!(p, poly_points;
        color = p.color,
        alpha = p.alpha,
        visible = p.filled
    )

    lines!(p, x, y;
        color = p.strokecolor,
        linewidth = p.strokewidth,
        visible = p.edge
    )

    return p
end

