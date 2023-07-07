# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    logvalof(r::NamedTuple{(...,:log,...)})::Real
    logvalof(r::LogDVal)::Real

*BAT-internal, not part of stable public API.*
"""
function logvalof end

function logvalof(d::Real)
    throw(ArgumentError("Can't get a logarithmic value from $d, unknown if it represents a lin or log value itself."))
end


Base.@noinline function logvalof(x::T) where {T<:NamedTuple}
    Base.depwarn("logvalof support for NamedTuples is deprecated, construct your density using DensityInterface.logfuncdensity instead of return a NamedTuple with a log field.", :logvalof)
    if hasfield(T, :logval) + hasfield(T, :logd) + hasfield(T, :log) > 1
        throw(ArgumentError("NamedTuples is ambiguous for logvalof contains fields $(join(map(string, filter(name -> name in (:logval, :logd, :log), fieldnames(T))), " and "))"))
    end
    if hasfield(T, :logval)
        x.logval
    elseif hasfield(T, :logd)
        x.logd
    elseif hasfield(T, :log)
        x.log
    else
        throw(ArgumentError("NamedTuple with fields $(fieldnames(T)) not supported by logvalof, doesn't have a field like :logval"))
    end
end



"""
    struct LogDVal{T<:Real}

*LogDVal is deprecated and will be removed in future major or even minor BAT versions.*
"""
struct LogDVal{T<:Real}
    logval::T
end

export LogDVal

Base.@noinline function logvalof(d::LogDVal)
    Base.depwarn("LogDVal is deprecated, construct your density using DensityInterface.logfuncdensity instead returning a LogDVal object.", :logvalof)
    d.logval
end
