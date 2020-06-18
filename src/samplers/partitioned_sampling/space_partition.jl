abstract type SpacePartitioningAlgorithm end
export SpacePartitioningAlgorithm

@with_kw struct KDTreePartitioning <: SpacePartitioningAlgorithm
	partition_dims::Union{Array{Int64,1}, Bool} = false
	cost_function::Function = x -> x
end

mutable struct SpacePartTree
   terminate::Bool
   bounds::Array{AbstractFloat}
   left_child::Union{SpacePartTree, Bool}
   right_child::Union{SpacePartTree, Bool}
   cut_axis::Union{Integer, Bool}
   cut_coordinate::Union{AbstractFloat, Bool}
   cost::AbstractFloat
   cost_part::Union{AbstractFloat, Bool}
end

function partition_space(samples::DensitySampleVector, n_partitions::Integer, algorithm::A) where {A<:SpacePartitioningAlgorithm}

	n_params = size(flatview(unshaped.(samples.v)))[1] # Change with smth smarter

	if algorithm.partition_dims == false
		partition_dims = collect(Base.OneTo(n_params))
	else
		partition_dims = intersect(algorithm.partition_dims, collect(Base.OneTo(n_params))) # to be safe
	end

	if length(samples) > 10^4; @warn "KDTreePartitioning: Too many exploration samples"; end

	flat_scaled_data, μ, δ = scale_data(samples)
	bounds = repeat([0.0 1.0],size(flat_scaled_data.samples)[1])
	partition_tree = def_init_node(flat_scaled_data, bounds)
	cost_values = Float64[]
	for i in 1:n_partitions
		@info "KDTree: Increasing tree depth: depth = $i"
		initialize_partitioning!(partition_tree, flat_scaled_data, partition_dims)
		ind, sum_cost = det_part_node(partition_tree)
		append!(cost_values, sum_cost)
		generate_node!(partition_tree, flat_scaled_data, ind)
	end

	return partition_tree

end

function scale_data(samples::DensitySampleVector)
	flat_samples = collect(flatview(unshaped.(samples.v)))
    μ = minimum(flat_samples, dims=2)
    δ = maximum(flat_samples, dims=2).-minimum(flat_samples, dims=2)
    return (samples = (flat_samples .- μ) ./ δ, weights = samples.weight, loglik = samples.logd), μ, δ
end

function def_init_node(data::T, bounds::Array{F}) where {T<:NamedTuple, F<:AbstractFloat}
    cost = evaluate_total_cost(data)
    return SpacePartTree(true, bounds, false, false, false, false, cost, false)
end

function evaluate_total_cost(data::T) where {T<:NamedTuple}
    sise_s = size(data.samples)
    if sise_s[2] > 3
		μ = mean(exp.(data.loglik), weights(data.weights))
        return sum(data.weights.*(exp.(data.loglik) .- μ).^2)
    else
        return Inf
    end
end

function initialize_partitioning!(tree::SpacePartTree, data::T, axes::Array{I,1}) where {T<:NamedTuple, I<:Integer}
    if tree.terminate == true
        if typeof(tree.cost_part) == Bool
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
    if size_s[2] > 3
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
    if tree.terminate == true
        append!(cost_values, tree.cost)
        append!(cost_delta, tree.cost - tree.cost_part)
    else
        collect_part_costs!(tree.left_child,cost_values, cost_delta)
        collect_part_costs!(tree.right_child,cost_values, cost_delta)
    end
end

function generate_node!(tree::SpacePartTree, data::T, cut_val::F) where {T<:NamedTuple, F<:AbstractFloat}
    if tree.terminate == true
        if tree.cost == cut_val

            new_bounds_left = copy(tree.bounds)
            new_bounds_right = copy(tree.bounds)

            new_bounds_left[tree.cut_axis, 2] = tree.cut_coordinate
            new_bounds_right[tree.cut_axis, 1] = tree.cut_coordinate

            tree.left_child = def_init_node(mask_data(data, new_bounds_left), new_bounds_left)
            tree.right_child = def_init_node(mask_data(data, new_bounds_right), new_bounds_right)
            tree.terminate = false

            return tree
        else
            return tree
        end

    else
        generate_node!(tree.left_child, data, cut_val)
        generate_node!(tree.right_child, data, cut_val)
    end
end
