# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieMarginalSelection

using Test
# The selection grammar and transactions can be verified without a rendering backend.
include(joinpath(@__DIR__, "../../ext/makie_impl/makie_marginal_selection.jl"))

@testset "Bounded marginal input grammar" begin
    for (text, expected) in (("", Int[]), ("[]", Int[]), (" [ ] ", Int[]),
        ("2, 7, 8", [2, 7, 8]), ("[2;7 8]", [2, 7, 8]), ("2:4", [2, 3, 4]),
        ("(2,8); (7,8)", [2, 7, 8]), ("1 (2,3)", [1, 2, 3]),
        ("3,2,3,1", [1, 2, 3]), (" 2\n7\t8 ", [2, 7, 8]), ("0002", [2]))
        result = _parse_marginal_dimensions(text, 1000, 3)
        @test isnothing(result.error)
        @test result.dimensions == expected
    end
    @test _parse_marginal_dimensions("(2,8); (7,8)", 1000, 3).has_pairs
    @test _parse_marginal_dimensions("(2,8); (7,8)", 1000, 3).duplicates
    for text in ("0", "-1", "1.2", "2e1", "1001", "[2,3", "2,3]", "1,", "1;;2",
        "3:1", "1:1000", "1:2:3", "(1,2,3)", "(1,2)(2,3)", "1 2 3 4", "sin(1)",
        "1; run(`echo unsafe`)", "９", "9"^100, "1,"^3000)
        result = _parse_marginal_dimensions(text, 1000, 3)
        @test !isnothing(result.error)
        @test isempty(result.dimensions) # Invalid edits never publish a partial selection.
    end
    @test _parse_marginal_dimensions("1:4", 1000, 4).dimensions == [1, 2, 3, 4]
    @test _parse_marginal_dimensions(string(typemax(Int)), typemax(Int), 1).dimensions == [typemax(Int)]
    @test !isnothing(_parse_marginal_dimensions("1:$(typemax(Int))", typemax(Int), 3).error)
    text = join(1:3000, ", ")
    @test ncodeunits(text) <= _marginal_text_limit(3000, 3000)
    @test _parse_marginal_dimensions(text, 3000, 3000).dimensions == collect(1:3000)
end

@testset "Marginal header candidates and owned construction state" begin
    original = [11, 2, 11, 0, 20]
    state = _MarginalSelection(12, 3, original)
    @test state.initial_selection == [11, 2, 11]
    original[1] = 1
    @test state.initial_selection == [11, 2, 11]
    candidate = _marginal_header_candidate(state, 2, "9")
    @test isnothing(candidate.error)
    @test candidate.selection == [9, 11]
    @test state.selection == [11, 2, 11]
    @test candidate.headers[2] == 9
    revision = _accept_marginal_intent!(state, candidate.selection, candidate.headers)
    candidate.selection[1] = 5
    @test state.selection == [9, 11]
    @test state.initial_headers[2] == 2
    @test !isnothing(_marginal_header_candidate(state, 1, "9").error)
    @test !isnothing(_marginal_header_candidate(state, 1, "13").error)
    @test _marginal_header_candidate(state, 1, "8").selection == [8, 9, 11]
    _accept_marginal_intent!(state, [7, 9, 11])
    @test !isnothing(_marginal_header_candidate(state, 1, "8").error)
    @test _accept_marginal_intent!(state, state.initial_selection, state.initial_headers) > revision
    @test state.selection == [11, 2, 11]
    @test state.headers == state.initial_headers
    @test state.headers !== state.initial_headers
    @test state.selection !== state.initial_selection
    @test _marginal_editor_headers(state, [8, 9, 11])[1:3] == [8, 9, 11]
    for n in (0, 1, 2, 3, 6, 1000, typemax(Int))
        @test 0 <= _marginal_slot_count(n) <= min(n, 6)
    end
end

end
