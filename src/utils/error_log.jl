# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct ErrLogEntry
    time::DateTime
    thread::Int
    process::Int
    error::Exception
end


const g_error_log = Ref{Union{Vector{ErrLogEntry},Nothing}}(nothing)

"""
    BAT.error_log()

*Experimental feature, not part of stable public API.*

Get a log of certain exceptions throws by BAT, e.g. density evaluation
errors.

The error log is disabled by default, use [`BAT.enable_error_log`](@ref)
to enable it.
"""
error_log() = g_error_log[]


"""
    BAT.enable_error_log(enable::Bool = true)

*Experimental feature, not part of stable public API.*

Enable/disable BAT's error (exception) log.

The error log is disabled by default.

See [`BAT.error_log`](@ref).
"""
function enable_error_log(enable::Bool = true)
    if enable
        if isnothing(g_error_log[])
            g_error_log[] = Vector{ErrLogEntry}[]
        end
    else
        g_error_log[] = nothing
    end
    nothing
end


function store_errlogentry(entry::ErrLogEntry)
    if !isnothing(error_log())
        push!(error_log(), entry)
    end
    if myid() != 1
        remotecall(() -> store_errlogentry(entry), 1)
    end
end


log_error(err::Exception) = store_errlogentry(ErrLogEntry(now(), threadid(), myid(), err))

function ChainRulesCore.rrule(::typeof(log_error), err::Exception)
    return log_error(), _log_error_pullback
end
_log_error_pullback(ΔΩ) = (NoTangent(), NoTangent())


macro throw_logged(expr)
    quote
        err = $(esc(expr))
        log_error(err)
        throw(err)
    end
end


macro rethrow_logged(expr)
    quote
        err = $(esc(expr))
        log_error(err)
        rethrow(err)
    end
end
