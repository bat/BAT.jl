function min_u(mat::Matrix)
    u = typemax(Float64)
    for i = 1:size(mat)[1]
        summa = sum(mat[i,:])
        u_test::Float64 = summa/(summa - mat[i,i])
        if u_test < u
            u = u_test
        end
    end
    println("the minimal is ", u)
    return u
end

function update_submat(mat::Matrix, u::Float64)
    for i = 1:size(mat)[1]
        mat[i,i] = 1 - u * (sum(mat[i,:]) - mat[i,i]) - (1 - sum(mat[i,:]))
    end
    mat = mat .* u
    for i = 1:size(mat)[1]
        mat[i,i] /= u
    end
    println("ssubmat before reshaping is ", mat)
    return mat
end

function update_row(row::Vector, submat_row::Vector, index::Vector)
    for (i,a) in zip(index,collect(1:1:length(submat_row)))
        row[i] = submat_row[a]
    end
    println("the row now is ", row)
    return row
end

function update_index_and_reshape_submat(index::Vector, mat::Matrix)
    ind_real = Int[]
    ind = Int[]
    for i = 1:size(mat)[1]
        if mat[i,i] != 0
            push!(ind_real, index[i])
            push!(ind, i)
        end
    end
    mat2 = zeros(Float64, length(ind), length(ind))
    for (i,a) in zip(collect(1:1:length(ind)),ind)
        for (j,b) in zip(collect(1:1:length(ind)),ind)
            mat2[i,j] = mat[a,b]
        end
    end
    return ind_real, mat2
end


function T23(row::Vector, κ::Int)
    if sum(row) > 1
        row = row ./ sum(row)
    end
    index = Int[]

    for i = 1:length(row)
        if row[i] != 0
            push!(index, i)
        end
    end
    println("index is ",index)
    submat = zeros(Float64, length(index), length(index))

    for i = 1:length(index)
        for j = 1:length(index)
            submat[i,j] = row[index[j]]
        end
    end

    println("submat is ",submat)

    while length(index) > 1
        u = min_u(submat)
        submat = update_submat(submat, u)
        c = find(index .== κ)[1]
        row = update_row(row, submat[c,:], index)
        if submat[c,c] == 0
            println("we broke ;)")
            break
        end
        index, submat = update_index_and_reshape_submat(index, submat)
        println(" fatto ")
        println("submat is ",submat)
    end
    println(" finito ")
    return row
end
