function min_u(mat::SubArray)
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

function update_submat!(mat::SubArray, u::Float64)
    for i = 1:size(mat)[1]
        mat[i,i] = 1 - u * (sum(mat[i,:]) - mat[i,i]) - (1 - sum(mat[i,:]))
    end
    # modify `mat` in place!
    mat .*= u
    for i = 1:size(mat)[1]
        mat[i,i] /= u
    end
    mat
end
function update_row!(row::Vector, submat_row::Vector, index::Vector, indexnew::Vector)
    for (i,a) in zip(index, indexnew)
        row[i] = submat_row[a]
    end
end

function update_indices!(index::Vector, indexnew::Vector, submat::SubArray)
    # can't use for loop because `index`'s size is changed
    j = 1
    while j <= length(index)
        if submat[j,j] == 0
            splice!(indexnew, j)
            splice!(index, j)
            j -= 1
        end
        j += 1
    end
end


doc"""
    multipropT2(row::Vector, κ::Int)

Compute the transition probability (T2) from Tjelmeland (2002) for the row of interest.

The `row` contains the target densities multiplied with the proposal densities,
`κ` is the index of `row` in the full-rank transition matrix.

"""
function multipropT2(row::Vector, κ::Int)
    # check input
    κ <= 0 && throw(ArgumentError("row index κ <=0"))
    κ > length(row) && throw(ArgumentError("row index κ > length(row)"))
    any(x -> x<0, row) && throw(ArgumentError("f"))

    # a normalized row is required
    row /= sum(row)

    # indices of rows in full matrix with non-zero diagonal elements
    index = Int[]
    # index of `row` in initial submatrix
    pos::Int64 = 0
    for i = 1:length(row)
        if row[i] != 0
            push!(index, i)
        end
    end
    # construct matrix from rows with non-zero diagonals
    submat = zeros(Float64, length(index), length(index))
    for i = 1:length(index)
        for j = 1:length(index)
            submat[i,j] = row[index[j]]
        end
    end

    # κ may be updated if there zeros on the diagonal
    κnew = findfirst(x -> x == κ, index)
    indexnew = collect(1:length(index))

    while length(index) > 1
        # is the κ-th element of `row` == zero? Then we're done
        if ∉(κ, index)
            break
        end
        submat_view = view(submat, indexnew, indexnew)
        u = min_u(submat_view)
        update_submat!(submat_view, u)
        update_row!(row, submat[κnew, :], index, indexnew)
        update_indices!(index, indexnew, submat_view)
    end
    row
end

export multipropT2
