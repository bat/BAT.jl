function min_u(mat::Matrix)
    u = typemax(Float64)
    for i = 1:size(mat)[1]
        summa = sum(mat[i,:])
        u_test::Float64 = summa/(summa - mat[i,i])
        if u_test < u
            u = u_test
        end
    end
    u
end

function update_submat(mat::Matrix, u::Float64)
    for i = 1:size(mat)[1]
        mat[i,i] = 1 - u * (sum(mat[i,:]) - mat[i,i]) - (1 - sum(mat[i,:]))
    end
    mat = mat .* u
    for i = 1:size(mat)[1]
        mat[i,i] /= u
    end
    mat
end
function update_row(row::Vector, submat_row::Vector, index::Vector)
    for (i,a) in zip(index,collect(1:1:length(submat_row)))
        row[i] = submat_row[a]
    end
    row
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
    ind_real, mat2
end


doc"""
    T23(row::Vector, κ::Int)

Compute the transition probability (T2) from Tjelmeland (2002) for the row of interest.

The `row` contains the target densities multiplied with the proposal densities,
`κ` is the index of `row` in the full-rank transition matrix.

"""
function T23(row::Vector, κ::Int)
    # check input
    κ <= 0 && throw(ArgumentError("row index κ <=0"))
    κ > length(row) && throw(ArgumentError("row index κ > length(row)"))
    any(x -> x<0, row) && throw(ArgumentError("f"))

    # a normalized row is required
    row /= sum(row)

    index = Int[]

    for i = 1:length(row)
        if row[i] != 0
            push!(index, i)
        end
    end
    submat = zeros(Float64, length(index), length(index))

    for i = 1:length(index)
        for j = 1:length(index)
            submat[i,j] = row[index[j]]
        end
    end

    while length(index) > 1
        u = min_u(submat)
        submat = update_submat(submat, u)
        c = find(index .== κ)[1]
        row = update_row(row, submat[c,:], index)
        if submat[c,c] == 0
            break
        end
        index, submat = update_index_and_reshape_submat(index, submat)
    end
    row
end
