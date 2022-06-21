# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    logvalof(r::NamedTuple{(...,:log,...)})::Real
    logvalof(r::LogDVal)::Real

**logvalof is deprecated and may be removed in future BAT versions.**
"""
function logvalof end
export logvalof

function logvalof(d::Real)
    throw(ArgumentError("Can't get a logarithmic value from $d, unknown if it represents a lin or log value itself."))
end


@inline function logvalof(x::T) where {T<:NamedTuple}
    if hasfield(T, :logval) + hasfield(T, :logd) + hasfield(T, :log) > 1
        throw(ArgumentError("NamedTuples is ambiguous for logvalof contains fields $(join(map(string, filter(name -> name in (:logval, :logd, :log), fieldnames(T))), " and "))"))
    end
    if hasfield(T, :logval)
        x.logval
    elseif hasfield(T, :logd)
        x.logd
    elseif hasfield(T, :log)
        _logvalof_deprecated(x, Val(:log))
    else
        throw(ArgumentError("NamedTuple with fields $(fieldnames(T)) not supported by logvalof, doesn't have a field like :logval"))
    end
end

Base.@noinline function _logvalof_deprecated(x::NamedTuple, ::Val{name}) where name
    Base.depwarn("logvalof support for NamedTuple field $name is deprecated, use NamedTuples with field :logval instead", :logvalof)
    getfield(x, name)
end


Base.@deprecate logvalof(density::AbstractMeasureOrDensity) DensityInterface.logdensityof(density)



"""
    struct LogDVal{T<:Real}

**LogDVal is deprecated and may be removed in future BAT versions.**
"""
struct LogDVal{T<:Real}
    logval::T
end

export LogDVal

logvalof(d::LogDVal) = d.logval
