struct Summary
    nsamples::Int64
    shape::NamedTupleShape
    stats::MCMCBasicStats
    marginalmodes::Vector{Float64}
    posterior::AbstractPosteriorDensity
    samplerinfo::SamplerInfo
end

function Summary(
    samples::DensitySampleVector,
    posterior::AbstractPosteriorDensity,
    samplerinfo::SamplerInfo
)
    nsamples = length(samples)
    shape = varshape(samples)
    stats = MCMCBasicStats(samples)
    marginalmodes = BAT.unshaped(bat_marginalmode(samples).result)
    return Summary(nsamples, shape, stats, marginalmodes, posterior, samplerinfo)
end

function _print_pretty(io::IO, m::MIME, obj::Any)
    showable(m, obj) ?  show(io, m, obj) : print(io, obj)
end
