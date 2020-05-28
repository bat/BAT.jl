export convert_to_bat_samples

function convert_to_bat_samples(samples::Any, posterior::BAT.AnyPosterior)
    logval = density_logval.(Ref(posterior), samples)

    n = length(logval)
    weights = exp.(logval)

    info = Vector{Nothing}(undef, n)
    aux = Vector{Nothing}(undef, n)

    shape = varshape(posterior)
    return shape.(DensitySampleVector((ArrayOfSimilarArrays(samples), logval, weights, info, aux)))
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
