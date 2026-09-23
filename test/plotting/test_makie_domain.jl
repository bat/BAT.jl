# This file is a part of BAT.jl, licensed under the MIT License (MIT).

module TestMakieDomain
using BAT, Makie, Test
const EXT = Base.get_extension(BAT, :BATMakieExt)

function samples(data::AbstractMatrix; elastic=false, weight=ones(size(data, 2)))
    storage = elastic ? EXT.ElasticArray(data) : copy(data)
    return DensitySampleVector(v=EXT.VectorOfSimilarVectors(storage),
        logd=zeros(size(data, 2)), weight=copy(weight))
end

# Frozen pre-change computation, including the bounds-only concatenation.
# Analytic expectations below are independent of this differential oracle.
function old_domain(chains, prior_lo, prior_hi)
    all_samples = vcat((walker for chain in chains for walker in chain)...)
    isempty(all_samples) && return copy(prior_lo), copy(prior_hi)
    data = all_samples.v.data
    lo, hi = similar(prior_lo), similar(prior_hi)
    for d in eachindex(prior_lo)
        finite_row = Iterators.filter(isfinite, view(data, d, :))
        if isempty(finite_row)
            lo[d], hi[d] = prior_lo[d], prior_hi[d]
        else
            a, b = extrema(finite_row)
            lo[d], hi[d] = min(a, prior_lo[d]), max(b, prior_hi[d])
        end
    end
    return lo, hi
end

domain(chains, lo, hi) = EXT._domain_including(Iterators.flatten(chains), lo, hi)
snapshot(s) = (copy(s.v.data), copy(s.logd), copy(s.weight), deepcopy(s.info), deepcopy(s.aux))

function check_domain(chains, lo, hi, expected)
    before = [snapshot(s) for chain in chains for s in chain]
    prior = (copy(lo), copy(hi))
    actual = domain(chains, lo, hi)
    @test isequal(actual, expected)
    @test isequal(actual, old_domain(chains, lo, hi))
    @test isequal((lo, hi), prior)
    @test isequal([snapshot(s) for chain in chains for s in chain], before)
    @test actual[1] isa Vector{Float64} && actual[2] isa Vector{Float64}
    @test actual[1] !== lo && actual[2] !== hi && actual[1] !== actual[2]
    again = domain(chains, lo, hi)
    @test actual[1] !== again[1] && actual[2] !== again[2]
    if !isempty(actual[1])
        actual[1][1] = 1234.0
        actual[2][1] = 5678.0
        @test isequal(again, expected)
        @test isequal((lo, hi), prior)
    end
end

function allocation_fixture(nrows; erased=false, elastic=false)
    walkers = [samples([sin(i + d + k) for d in 1:8, i in 1:nrows]; elastic)
        for k in 1:4]
    if erased
        return Any[Any[walkers[1], walkers[2]], Any[walkers[3], walkers[4]]]
    end
    return Vector{DensitySampleVector}[
        DensitySampleVector[walkers[1], walkers[2]],
        DensitySampleVector[walkers[3], walkers[4]]]
end

function allocation_check(; erased, elastic)
    small = allocation_fixture(16; erased, elastic)
    large = allocation_fixture(8192; erased, elastic)
    lo, hi = fill(-0.5, 8), fill(0.5, 8)
    # Warm both paths, both sizes, and the allocation-measurement specializations.
    @test isequal(domain(small, lo, hi), old_domain(small, lo, hi))
    @test isequal(domain(large, lo, hi), old_domain(large, lo, hi))
    for input in (small, large)
        @allocated domain(input, lo, hi)
        @allocated old_domain(input, lo, hi)
    end
    new_small = minimum([@allocated(domain(small, lo, hi)) for _ in 1:3])
    new_large = minimum([@allocated(domain(large, lo, hi)) for _ in 1:3])
    old_small = minimum([@allocated(old_domain(small, lo, hi)) for _ in 1:3])
    old_large = minimum([@allocated(old_domain(large, lo, hi)) for _ in 1:3])
    # Fixed topology and Float64 arithmetic: no row-proportional scratch copy.
    # A fixed allowance avoids pinning compiler-specific allocation counts.
    @test new_large <= new_small + 4096
    @test new_large < old_large / 10
    @test old_large > old_small + 8 * (32768 - 64) * sizeof(Float64)
    new_seconds = minimum([@elapsed(domain(large, lo, hi)) for _ in 1:5])
    old_seconds = minimum([@elapsed(old_domain(large, lo, hi)) for _ in 1:5])
    @info "Warmed bounds scan; eight dimensions, four walkers, 64 versus 32768 total rows" erased elastic new_small new_large old_small old_large new_seconds old_seconds
end

@testset "Live domain without history materialization" begin
    for T in (Float32, Float64, BigFloat), elastic in (false, true)
        @testset "$T; elastic=$elastic" begin
            first = samples(T[-4 0 2; NaN Inf -Inf; NaN Inf NaN; 5 5 5]; elastic)
            second = samples(T[-2 7 NaN; -3 8 0; -Inf NaN Inf; 5 5 5];
                elastic, weight=zeros(3))
            empty_walker = samples(zeros(T, 4, 0); elastic)
            chains = [[first, empty_walker], [second]]
            check_domain(chains, [-1.0, -2.0, 3.0, 0.0], [1.0, 2.0, 4.0, 1.0],
                ([-4.0, -3.0, 3.0, 0.0], [7.0, 8.0, 4.0, 5.0]))
            check_domain([[empty_walker], [empty_walker]],
                [-1.0, -2.0, 3.0, 0.0], [1.0, 2.0, 4.0, 1.0],
                ([-1.0, -2.0, 3.0, 0.0], [1.0, 2.0, 4.0, 1.0]))
        end
    end
    check_domain(Any[], [-2.0], [4.0], ([-2.0], [4.0]))
    check_domain([Any[], Any[]], [-2.0], [4.0], ([-2.0], [4.0]))
    check_domain([[samples(zeros(0, 2))]], Float64[], Float64[], (Float64[], Float64[]))

    @testset "Signed zero and native finite filtering" begin
        signed = samples(reshape([-0.0, 0.0], 2, 1))
        check_domain([[signed]], [0.0, 0.0], [-0.0, -0.0],
            ([-0.0, 0.0], [-0.0, 0.0]))
        mixed = [[samples(reshape([-0.0], 1, 1))], [samples(reshape([0.0], 1, 1))]]
        check_domain(mixed, [0.0], [-0.0], ([-0.0], [0.0]))
        check_domain(reverse(mixed), [0.0], [-0.0], ([-0.0], [0.0]))
        huge = BigFloat("1e400")
        wide = [[samples([huge BigFloat(Inf); -huge BigFloat(-Inf)])]]
        # Native values are finite, even though the existing Float64 output
        # conversion overflows. Filtering after conversion would lose them.
        check_domain(wide, [0.0, 0.0], [0.0, 0.0], ([0.0, -Inf], [Inf, 0.0]))
    end

    @testset "Warmed allocation growth" begin
        for erased in (false, true), elastic in (false, true)
            allocation_check(; erased, elastic)
        end
    end
end
end
