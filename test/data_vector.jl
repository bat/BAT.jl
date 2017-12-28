# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

struct dv_test{T} <: BAT.BATDataVector{T}
    data::Vector{T}
end

Base.append!(x::dv_test, y::dv_test) = append!(x.data, y.data)

@testset "data_vector" begin

    @testset "data_vector" begin
        x = @inferred dv_test{Float64}(Float64(1):Float64(3))
        @test typeof(x) <: DenseVector{Float64}
    end

    @testset "merge" begin
        x = @inferred dv_test{Float64}(Float64(1):Float64(3))
        y = @inferred dv_test{Float64}(Float64(4):Float64(6))
        z = @inferred dv_test{Float64}(Float64(7):Float64(9))
        @test merge(x, y, z).data == collect(Float64(1):Float64(9))
        merge!(x, y, z)
        @test x.data == collect(Float64(1):Float64(9))
    end
end
