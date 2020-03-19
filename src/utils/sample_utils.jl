export convert_to_bat_samples

function convert_to_bat_samples(samples, posterior)
    samples, weights = get_weighted_samples(samples)
    n = length(samples)

    logval = density_logval.(Ref(posterior), samples[:])
    info = Vector{Nothing}(undef, n)
    aux = Vector{Nothing}(undef, n)

    varshape(posterior).(DensitySampleVector((ArrayOfSimilarArrays(samples), logval, weights, info, aux)))
end


function get_weighted_samples(samples)
    n = length(samples)
    weights = ones(Int, n)

    i=2
    k = 0
    while i <= n-k
        if samples[i] == samples[i-1]
            weights[i-1] += 1
            deleteat!(samples, i)
            deleteat!(weights, i)
            k += 1
        else
            i = i+1
        end
    end
    return samples, weights
end
