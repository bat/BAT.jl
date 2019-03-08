# for 1d and 2d histograms
function split_smallest(histogram, intervals)
    intervals = sort(intervals)

    hist = deepcopy(histogram) 
    hists_weights = Array{Array{Real}}(undef, length(intervals))
    dims = size(hist.weights)

    weights = vec(hist.weights)
    totalweights = sum(weights)
    weight_ids = sortperm(weights, rev=true) # starting with highest weight

    for i in 1:length(intervals)
        hists_weights[i] = zeros(length(weights))
    end

    rel_weights = weights/totalweights

    for (i, intv) in enumerate(intervals)
        sum_weights = 0.
        for w in weight_ids
            if sum_weights < intv
                hists_weights[i][w] = hist.weights[w]
                sum_weights += rel_weights[w]
            else
                break
            end
        end
    end

    hists = Array{StatsBase.Histogram}(undef, length(intervals))

    for i in 1:length(intervals)
        hists[i] = deepcopy(hist)
        hists[i].weights = reshape(hists_weights[i], dims)
    end

    realintervals = calculate_hist_percentage(hists, hist)

    return reverse(hists), histogram, reverse(realintervals)
end


# for 1d histograms
function split_central(histogram, intervals)
    intervals = sort(intervals)
    intervals = (1 .-intervals)/2

    hist = deepcopy(histogram) 
    hists = Array{StatsBase.Histogram}(undef, length(intervals))

    for i in 1:length(intervals)
        hists[i] = deepcopy(hist)
    end

    totalweights = sum(hist.weights)
   
    rel_weights = hist.weights/totalweights

    for (i, intv) in enumerate(intervals)
        sum_left = 0.
        sum_right = 0.

        for l in 1:length(hist.weights)
           if sum_left + rel_weights[l] < intv
                sum_left = sum_left + rel_weights[l]
                hists[i].weights[l] = 0
           else
                break
           end
        end 

        for r in length(hist.weights):-1:1
           if sum_right + rel_weights[r] < intv
                sum_right = sum_right + rel_weights[r]
                hists[i].weights[r] = 0
           else
                break
           end
        end 
    end

    realintervals = calculate_hist_percentage(hists, hist)

    return reverse(hists), histogram, reverse(realintervals)
end


function calculate_hist_percentage(hists, hist)
# calculate percentage really enclosed inside the intervals (possible differences for to large bins)
    totalweights = sum(hist.weights)
    realintervals = zeros(length(hists))

    for (i, h) in enumerate(hists)
        realintervals[i] = sum(h.weights)/totalweights
    end

    return realintervals
end


function bin_centers(h)
    m = length(h.edges[1])-1
    n = length(h.edges[2])-1
    
    x = Vector{Real}(undef, m)
    y = Vector{Real}(undef, n)

    for i in 1:m
        x[i] = h.edges[1][i]+0.5*(h.edges[1][i+1]-h.edges[1][i])
    end
        
    for j in 1:n
        y[j] =  h.edges[2][j]+0.5*(h.edges[2][j+1]-h.edges[2][j])
    end
    
    return x, y, h.weights
end


function calculate_levels(weights, intervals)
    intervals = sort(intervals)
    levels = Vector{Real}(undef, length(intervals)+1)
    
    w = vec(weights)
    w = sort(w, rev=true)
    
    weight_ids = sortperm(w, rev=true);
    summedw = sum(w)

    sumw = 0.0
    
    for w_id in weight_ids
        if(sumw <= intervals[end])
            i = findfirst(x -> sumw <= x, intervals)
            levels[i] = w[w_id]
            sumw += w[w_id]/summedw
        end 
    end
    
    levels[end] = 1.1*summedw
    
    return sort(levels)
end