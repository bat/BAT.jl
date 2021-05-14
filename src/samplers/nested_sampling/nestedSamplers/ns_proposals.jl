export Uniformly
export RandomWalk

abstract type NSProposal end


struct Uniformly <: NSProposal end

@with_kw struct RandomWalk <: NSProposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct RStaggering <: NSProposal
    ratio::Float64 = 0.5
    walks::Int64 = 25
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct Slicing <: NSProposal
    slices::Int64 = 5
    scale::Float64 = 1.0 # >= 0
end

@with_kw struct RSlicing <: NSProposal
    slices::Int64 = 5
    scale::Float64 = 1.0 # >= 0
end

function NSprop(prop::Uniformly)
    return Proposals.Uniform()
end

function NSprop(prop::RandomWalk)
    return Proposals.RWalk(;prop.ratio, prop.walks, prop.scale)
end

function NSprop(prop::RStaggering)
    Proposals.RStagger(;prop.ratio, prop.walks, prop.scale)
end

function NSprop(prop::Slicing)
    return Proposals.Slice(;prop.slices, prop.scale)
end

function NSprop(prop::RSlicing)
    return Proposals.RSlice(;prop.slices, prop.scale)
end

function NSprop(prop::NSProposal) # wenn nichts ausgewählt wird
    return Proposals.RandomWalk()
end
