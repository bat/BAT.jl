# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT

struct ShiftedVector{T,V<:AbstractVector{T}} <: AbstractVector{T}
    values::V
    first::Int
end

Base.size(v::ShiftedVector) = size(v.values)
Base.axes(v::ShiftedVector) = (v.first:(v.first + length(v.values) - 1),)
Base.getindex(v::ShiftedVector, i::Int) = v.values[i - v.first + 1]
Base.setindex!(v::ShiftedVector, value, i::Int) = v.values[i - v.first + 1] = value

function test_length_mismatch(executor, output, input)
    output_before = copy(output)
    err = try
        BAT.exec_map!(identity, executor, output, input)
        nothing
    catch ex
        ex
    end
    @test err isa ArgumentError
    @test sprint(showerror, err) == "ArgumentError: Input and output arrays must have equal lengths."
    @test output == output_before
end

Test.@testset "executor length validation" begin
    for executor in (BAT.SequentialExec(), BAT.MultiThreadedExec())
        @testset "$(typeof(executor))" begin
            test_length_mismatch(executor, [0, 0], [1])
            test_length_mismatch(executor, [0], [1, 2])

            input = ShiftedVector([1, 2, 3], -2)
            output = ShiftedVector([0, 0, 0], 10)
            @test BAT.exec_map!(x -> 2x, executor, output, input) === output
            @test output.values == [2, 4, 6]
        end
    end
end
