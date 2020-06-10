function repetition_weights(v::AbstractVector)
    n = length(eachindex(v))
    weights = ones(Int, n)

    i=2
    k = 0
    while i <= n-k
        if v[i] == v[i-1]
            weights[i-1] += 1
            deleteat!(v, i)
            deleteat!(weights, i)
            k += 1
        else
            i = i+1
        end
    end
    return v, weights
end
