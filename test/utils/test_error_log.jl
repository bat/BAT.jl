# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Test


@testset "error_log" begin
    @test BAT.enable_error_log(false) === nothing
    @test BAT.error_log() === nothing
    @test_throws ArgumentError BAT.@throw_logged ArgumentError("foo")
    @test BAT.error_log() === nothing

    @test BAT.enable_error_log(true) === nothing
    @test BAT.error_log() isa AbstractVector{<:BAT.ErrLogEntry}
    @test_throws ArgumentError BAT.@throw_logged ArgumentError("foo")
    @test last(BAT.error_log()).error == ArgumentError("foo")

    @test_throws ErrorException try
        BAT.@throw_logged ArgumentError("bar")
    catch
        BAT.@rethrow_logged ErrorException("Some rethrow")
    end

    @test BAT.error_log()[end-1].error == ArgumentError("bar")
    @test BAT.error_log()[end].error == ErrorException("Some rethrow")

    BAT.enable_error_log(false)
    @test BAT.error_log() === nothing
end
