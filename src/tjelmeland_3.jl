function min_u(mat::SubArray)
    u = typemax(Float64)
    for i = 1:size(mat)[1]
        summa = sum(mat[i,:])
        u_test::Float64 = summa/(summa - mat[i,i])
        if u_test < u
            u = u_test
        end
    end
    println("the minimal is ", u,"\n")
    return u
end

function update_submat!(mat::SubArray, u::Float64)
    for i = 1:size(mat)[1]
        mat[i,i] = 1 - u * (sum(mat[i,:]) - mat[i,i]) - (1 - sum(mat[i,:]))
    end
    mat = mat .* u
    for i = 1:size(mat)[1]
        mat[i,i] /= u
    end
    println("ssubmat before reshaping is ", mat,"\n")
end

function update_row!(row::Vector, submat_row::Vector, index::Vector)
    for (i,a) in zip(index,collect(1:1:length(submat_row)))
        row[i] = submat_row[a]
    end
    println("the row now is ", row,"\n")
    return row
end

function update_index_and_reshape_submat(index::Vector, mat::SubArray)
    ind_real = Int[]
    #ind = Int[]
    for i = 1:size(mat)[1]
        if mat[i,i] != 0
            push!(ind_real, index[i])
            #push!(ind, i)
        end
    end
    #mat2 = zeros(Float64, length(ind), length(ind))
    #for (i,a) in zip(collect(1:1:length(ind)),ind)
    #    for (j,b) in zip(collect(1:1:length(ind)),ind)
    #        mat2[i,j] = mat[a,b]
    #    end
    #end
    return ind_real
end


function T23(row::Vector, κ::Int)
    row = row ./ sum(row)

    index = Int[]
    pos_in_submat::Int64 = 0
    for i = 1:length(row)
        if row[i] != 0
            push!(index, i)
        end
    if in(κ, index)
        pos = find(index .== κ)[1]
    end
    println("index is ",index)
    submat = zeros(Float64, length(index), length(index))

    for i = 1:length(index)
        for j = 1:length(index)
            submat[i,j] = row[index[j]]
        end
    end
    submat_view = view(submat, :, :)
    println("****submat is ",submat_view)
    passo = 1
    while length(index) > 1
        print(" passo is ",passo,"\n")
        if ∉(κ, index)
            print(" ------------------ the index now is ", index,"\n")
            println(" ---------------- we broke ;) \n")
            break
        end
        u = min_u(submat_view)
        update_submat!(submat_view, u)
        println(" e momomom ", submat_view)
        update_row!(row, submat_view[find(index .== κ)[1], :], index::Vector)
        #index = update_index_and_reshape_submat(index, submat_view)
        for i = 1:length(index)
            if submat_view[i,i] == 0
                index[i] = 0
            end
        end
        j = 1
        while j <= length(index)
            if index[j] == 0
                splice!(index,j)
                j -= 1
            end
            j += 1
        end
    end
        println("after update index is ",index)
        println("ricordiamoci di submat vero ", submat)
        submat_view = view(submat, index, index)
        println(" fatto \n")
        println(" submat is ",submat_view,"\n")
        passo += 1
    end
    println(" finito ")
    return row
end

input = [0.2, 0.5, 0.3]
κ = 3
result_1 =[0.25, 0.75, 0.0]
result_2=T23(input, κ)

println(result_2-result_1)

    #input = [288., 64., 135.] / 487.::Float64
    #κ = 3
    #result_1 =[359., 64, 0.] / 423.::Float64
    #result_2=T23(input, κ)

    #print(result_2-result_1)
    #print(T23(input, κ) ≈ result_1)

# a = [1,2,3,4,5,6,7,8,9]
#
# for i = 1:length(a)
#     print(i)
#     splice!(a,i)
#     i -= 1
#     if i == length(a)
#         println("wewweweeferf")
#         break
#     end
# end
