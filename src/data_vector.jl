# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type BATDataVector{T} <: DenseVector{T} end


function Base.merge!(X::BATDataVector, Xs::BATDataVector...)
    for Y in Xs
        append!(X, Y)
    end
    X
end

Base.merge(X::BATDataVector, Xs::BATDataVector...) = merge!(deepcopy(X), Xs...)


mcmc_callback(x::BATDataVector) = MCMCPushCallback(x)
mcmc_callback(max_level::Integer, x::BATDataVector) = MCMCPushCallback(max_level, x)
