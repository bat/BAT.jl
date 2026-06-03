
function compute_plotting_primitives(
        ::Nothing,
        ::Nothing,
        ::Scatter2D,
        ::NamedTuple
)
        return (x=SubArray(), y=SubArray(), weights=SubArray())
end

function compute_plotting_primitives(
        marg_coords::SubArray,
        weights::SubArray,
        recipe::Scatter2D,
        config::NamedTuple
)
        x = marg_coords[1, :]
        y = marg_coords[2, :]

        return (x=x, y=y, weights=weights)
end

function compose_plotspecs(
        primitives::NamedTuple,
        recipe::Scatter2D,
        config::NamedTuple
)
        (; x, y, weights) = primitives
        (; color, alpha, markersize) = config

        real_markersize = if isempty(weights) || (all(x -> x ≈ weights[1], weights)) || (mean(weights) <= 0)
                markersize
        else
                sqrt.(weights ./ mean(weights)) .* markersize
        end

        scatter = S.Scatter(x, y;
                color=color,
                alpha=alpha,
                markersize=real_markersize
        )

        return [scatter]
end

