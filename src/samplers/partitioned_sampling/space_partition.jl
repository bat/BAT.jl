abstract type SpacePartitioningAlgorithm end
export SpacePartitioningAlgorithm

@with_kw struct KDTreePartitioning <: SpacePartitioningAlgorithm
	partition_dims::Union{Array{Int64,1}, Bool} = false
	cost_function::Function = x -> x
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
	bounds = repeat([0.0 1.0],size(flat_scaled_data.samples)[1])' #
	# tree = def_init_node(data, bounds)


	# try
	#
	# 	# tree = def_init_node(data, bounds)
	# 	# cost_values = []
	# 	# for i in 1:kd_size
	# 	# 	@info "KDTree: Increasing tree depth: depth = $i"
	# 	# 	initialize_partitioning!(tree, data, axes)
	# 	# 	ind, sum_cost = det_part_node(tree)
	# 	# 	append!(cost_values, sum_cost)
	# 	# 	generate_node!(tree, data, ind)
	# 	# end
	# 	# rescale_tree!(tree, μ, δ)
	# 	# return tree, cost_values
	# catch
		# @error "KDTreePartitioning: Error"
	# end

end


function scale_data(samples::DensitySampleVector)
	flat_samples = collect(flatview(unshaped.(samples.v)))
    μ = minimum(flat_samples, dims=2)
    δ = maximum(flat_samples, dims=2).-minimum(flat_samples, dims=2)
    return (samples = (flat_samples .- μ) ./ δ, weights = samples.weight, loglik = samples.logd), μ, δ
end
