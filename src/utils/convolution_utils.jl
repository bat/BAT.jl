# This file is a part of BAT.jl, licensed under the MIT License (MIT).

# simple 2d convolution with padding
function convolution(input, filter; padding=:same)
    input_r, input_c = size(input)
    filter_r, filter_c = size(filter)

    if padding == :same
        pad_r = (filter_r - 1) ÷ 2 
        pad_c = (filter_c - 1) ÷ 2 
        
        input_padded = zeros(input_r+(2*pad_r), input_c+(2*pad_c))
        for i in 1:input_r, j in 1:input_c
            input_padded[i+pad_r, j+pad_c] = input[i, j]
        end
        input = input_padded
        input_r, input_c = size(input)
    end

    result = zeros(input_r-filter_r+1, input_c-filter_c+1)
    result_r, result_c = size(result)

    for i in 1:result_r
        for j in 1:result_c
            for k in 1:filter_r 
                for l in 1:filter_c 
                    result[i,j] += input[i+k-1,j+l-1]*filter[k,l]
                end
            end
        end
    end

    return result
end


# gaussian kernel with same σ in both dimensions
function gaussian_kernel(σ::Real; l::Int = 4*ceil(Int,σ)+1)
    isodd(l) || throw(ArgumentError("length must be odd"))
    w = l>>1
    g = σ == 0 ? [exp(0/(2*oftype(σ, 1)^2))] : [exp(-x^2/(2*σ^2)) for x=-w:w]
    k = g/sum(g)
    return (k * k')
end

# gaussian kernel with different σs in both dimensions
function gaussian_kernel(
    σs::Tuple{Real, Real}; 
    l::Tuple{Int, Int} = (4*ceil(Int,σs[1])+1, 4*ceil(Int,σs[2])+1)
)
    all(isodd.(l)) || throw(ArgumentError("length must be odd"))
    w1 = l[1]>>1
    g1 = σs[1] == 0 ? [exp(0/(2*oftype(σs[1], 1)^2))] : [exp(-x^2/(2*σs[1]^2)) for x=-w1:w1]
    k1 = g1/sum(g1)

    w2 = l[2]>>1
    g2 = σs[2] == 0 ? [exp(0/(2*oftype(σs[2], 1)^2))] : [exp(-x^2/(2*σs[2]^2)) for x=-w2:w2]
    k2 = g2/sum(g2)

    return (k1 * k2')  
end
