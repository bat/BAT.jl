# This file is a part of BAT.jl, licensed under the MIT License (MIT).

"""
    partition_space(
		samples::DensitySampleVector,
		n_partitions::Integer,
		algorithm::KDTreePartitioning
	)

*BAT-internal, not part of stable public API.*

The function generates a space partition tree with the number of partitions
given by `n_partitions`, using `KDTreePartitioning` algorithm and `samples`.
The output contains `SpacePartTree` and the values of the cost function.
"""
function partition_space(samples::DensitySampleVector, n_partitions::Integer, algorithm::KDTreePartitioning)

	n_params = size(flatview(unshaped.(samples.v)))[1] # Change with smth smarter

	if algorithm.partition_dims == false #check whether the user specified manually dimensions for partition. Use all if not.
		partition_dims = collect(Base.OneTo(n_params))
	else
		partition_dims = sort(intersect(algorithm.partition_dims, collect(Base.OneTo(n_params)))) # to be safe
	end

	flat_scaled_data, μ, δ = scale_samples(samples)
	bounds = repeat([0.0 1.0],size(flat_scaled_data.samples)[1])
	partition_tree = def_init_node(flat_scaled_data, bounds)
	cost_values = Float64[]
	for i in 1:n_partitions-1
		@info "KDTreePartitioning: Increasing tree depth (depth = $i)"
		initialize_partitioning!(partition_tree, flat_scaled_data, partition_dims)
		ind, sum_cost = det_part_node(partition_tree)
		append!(cost_values, sum_cost)
		generate_node!(partition_tree, flat_scaled_data, ind)
	end

	rescale_tree!(partition_tree, μ, δ)

	return partition_tree, cost_values
end

function scale_samples(samples::DensitySampleVector)
	flat_samples = collect(flatview(unshaped.(samples.v)))
    μ = minimum(flat_samples, dims=2)
    δ = maximum(flat_samples, dims=2).-minimum(flat_samples, dims=2)
    return (samples = (flat_samples .- μ) ./ δ, weights = samples.weight, loglik = samples.logd), μ, δ
end

function def_init_node(data::T, bounds::Array{F}) where {T<:NamedTuple, F<:AbstractFloat}
    cost = evaluate_total_cost(data)
    return SpacePartTree(true, bounds, missing, missing, missing, missing, cost, missing)
end

function initialize_partitioning!(tree::SpacePartTree, data::T, axes::Array{I,1}) where {T<:NamedTuple, I<:Integer}
    if tree.terminated_leaf == true
        if ismissing(tree.cost_part)
            cost, (div_axis, cut_pos) = find_min_along_axes(mask_data(data, tree.bounds), axes)
            tree.cost_part = cost
            tree.cut_coordinate = cut_pos
            tree.cut_axis = div_axis
            return tree
        else
            return tree
        end
    else
        initialize_partitioning!(tree.left_child, data, axes)
        initialize_partitioning!(tree.right_child, data, axes)
    end
end

function find_min_along_axes(data::T, axes::Array{I,1}) where {T<:NamedTuple, I<:Integer}

    min_function = Float64[]
    min_positions = Float64[]

    for i in axes
        _, _, min_x, min_y = find_min_along_axis(data, i)
        push!(min_function, min_y)
        push!(min_positions, min_x)
    end
    min_glob_axis = argmin(min_function)
    divide_along = axes[min_glob_axis]
    cut_position = min_positions[min_glob_axis]
    return minimum(min_function), (divide_along, cut_position)
end

function find_min_along_axis(data::T, axis::Integer; n_δ=50) where {T<:NamedTuple}

    grid = range(minimum(data.samples[axis,:]),stop=maximum(data.samples[axis,:]), length=n_δ)
    cost_values = Float64[]

    for cut in collect(grid)
        cost_value = evaluate_param_cost(data, cut, axis)
        push!(cost_values, cost_value)
    end
    nan_mask = .!isnan.(cost_values)
    min_value = minimum(cost_values[nan_mask])

	min_indicies = findall(x->x==min_value, cost_values[nan_mask])
	mid_index = 1 + (length(min_indicies)-1)÷2
	mid_index = min_indicies[mid_index]

    # min_bin = argmin(cost_values[nan_mask])
    min_bin_value = grid[nan_mask][mid_index]

    return (grid, cost_values, min_bin_value, min_value)
end

function evaluate_param_cost(data::T, cut_position::F, axis::I) where {T<:NamedTuple, I<:Integer, F<:AbstractFloat}
    size_s = size(data.samples)
	n_samples_critical = 3 #if less than 3 samples then return Inf
    if size_s[2] > n_samples_critical
        mask = data.samples[axis, :] .< cut_position
		a1 = evaluate_total_cost((samples = data.samples[:,mask], weights = data.weights[mask], loglik = data.loglik[mask]))
		a2 = evaluate_total_cost((samples = data.samples[:,.!mask], weights = data.weights[.!mask], loglik = data.loglik[.!mask]))
        return  a1+a2
    else
        return Inf
    end
end

function mask_data(data::T, bounds::Array{F}) where {T<:NamedTuple, F<:AbstractFloat}
    vect_mask = [prod(bounds[:,2] .> data.samples .> bounds[:,1], dims=1)...]
    return (samples = data.samples[:,vect_mask], weights = data.weights[vect_mask], loglik = data.loglik[vect_mask] )
end

function det_part_node(tree::SpacePartTree)
    costs_arrays::Array{Float64} = []
    cost_delta::Array{Float64} = []
    collect_part_costs!(tree, costs_arrays, cost_delta)
    ind = argmax(cost_delta)
    return costs_arrays[ind], sum(costs_arrays)
end

function collect_part_costs!(tree::SpacePartTree, cost_values::Array{Float64,1}, cost_delta::Array{Float64,1})
    if tree.terminated_leaf == true
        append!(cost_values, tree.cost)
        append!(cost_delta, tree.cost - tree.cost_part)
    else
        collect_part_costs!(tree.left_child,cost_values, cost_delta)
        collect_part_costs!(tree.right_child,cost_values, cost_delta)
    end
end

function generate_node!(tree::SpacePartTree, data::T, cut_val::F) where {T<:NamedTuple, F<:AbstractFloat}
    if tree.terminated_leaf == true
        if tree.cost == cut_val

            new_bounds_left = copy(tree.bounds)
            new_bounds_right = copy(tree.bounds)

            new_bounds_left[tree.cut_axis, 2] = tree.cut_coordinate
            new_bounds_right[tree.cut_axis, 1] = tree.cut_coordinate

            tree.left_child = def_init_node(mask_data(data, new_bounds_left), new_bounds_left)
            tree.right_child = def_init_node(mask_data(data, new_bounds_right), new_bounds_right)
            tree.terminated_leaf = false

            return tree
        else
            return tree
        end

    else
        generate_node!(tree.left_child, data, cut_val)
        generate_node!(tree.right_child, data, cut_val)
    end
end

function evaluate_total_cost(data::T) where {T<:NamedTuple}
    sise_s = size(data.samples)
    if sise_s[2] > 3
        μ = mean(data.samples, weights(data.weights), dims=2)
        return sum(data.weights.*sum((data.samples .- μ).^2, dims=1))
    else
        return Inf
    end
end

function cost_f_2(data::T) where {T<:NamedTuple}
    sise_s = size(data.samples)
    if sise_s[2] > 3
		μ = mean(exp.(data.loglik), weights(data.weights))
        return sum(data.weights.*(exp.(data.loglik) .- μ).^2)
    else
        return Inf
    end
end

function cost_f_3(data::T) where {T<:NamedTuple}
    sise_s = size(data.samples)
    if sise_s[2] > 3
        μ = data.samples[:,argmax(data.weights)]
        return   sum( data.weights.*sum((data.samples .- μ).^2, dims=1))
    else
        return Inf
    end
end

function rescale_tree!(tree::SpacePartTree, μ::Array{F}, δ::Array{F}) where {F<:AbstractFloat}
    if tree.terminated_leaf == true
        tree.bounds = tree.bounds.*δ .+ μ
    else
        tree.bounds = tree.bounds.*δ .+ μ
        rescale_tree!(tree.left_child, μ, δ)
        rescale_tree!(tree.right_child, μ, δ)
    end
end

function get_tree_par_bounds!(tree::SpacePartTree, bounds_part::Array{Array{F},1}) where {F<:AbstractFloat}
    if tree.terminated_leaf == true
        push!(bounds_part, tree.bounds)
    else
        get_tree_par_bounds!(tree.left_child, bounds_part)
        get_tree_par_bounds!(tree.right_child, bounds_part)
    end
end

function get_tree_par_bounds(tree::SpacePartTree)
    param_bounds::Array{Array{Float64},1} = []
	return get_tree_par_bounds!(tree, param_bounds)
end

function extend_tree_bounds!(tree::SpacePartTree, lo::Array{F,1}, hi::Array{F,1}) where {F<:AbstractFloat}

	subspaces_rect_bounds = get_tree_par_bounds(tree)

	lo_tree_bounds = [minimum(hcat([tree_bound[:,1] for tree_bound in subspaces_rect_bounds]...), dims=2)...]
    hi_tree_bounds = [maximum(hcat([tree_bound[:,2] for tree_bound in subspaces_rect_bounds]...), dims=2)...]

	to_be_extended = [Pair.(lo_tree_bounds, lo); Pair.(hi_tree_bounds, hi)]

	extend_tree_bounds!(tree, to_be_extended)
end

function extend_tree_bounds!(tree::SpacePartTree, to_be_extended::Array{P}) where {P<:Pair}
    if tree.terminated_leaf == true
		tree.bounds = replace(tree.bounds, to_be_extended...)
    else
        extend_tree_bounds!(tree.left_child, to_be_extended)
        extend_tree_bounds!(tree.right_child, to_be_extended)
    end
end
