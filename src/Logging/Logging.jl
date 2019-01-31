# This file is a part of BAT.jl, licensed under the MIT License (MIT).


module Logging

using Base.Threads
using Distributed


@enum LogLevel LOG_NONE=0 LOG_ERROR=1 LOG_WARNING=2 LOG_INFO=3 LOG_DEBUG=4 LOG_TRACE=5 LOG_ALL=100
export LogLevel
export LOG_NONE, LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG, LOG_TRACE, LOG_ALL


import Base: +,-

Base.convert(::Type{T}, level::LogLevel) where {T<:Integer} = T(level)

function +(level::LogLevel, i::Integer)
    l = Int(level)
    l_min = Int(LOG_ERROR)
    l_max = Int(LOG_TRACE)
    ifelse(
        l_min <= l <= l_max,
        LogLevel(clamp(l + i, l_min, l_max)),
        level
    )
end

-(level::LogLevel, i::Integer) = level + (-i)



const log_colors = Dict(
    LOG_ERROR => Base.error_color(),
    LOG_WARNING => Base.warn_color(),
    LOG_INFO => Base.info_color(),
    LOG_DEBUG => :light_green,
    LOG_TRACE => :green,
)


const log_prefix = Dict(
    LOG_ERROR => "ERROR",
    LOG_WARNING => "WARNING",
    LOG_INFO => "INFO",
    LOG_DEBUG => "DEBUG",
    LOG_TRACE => "TRACE",
)


@static if VERSION >= v"1.2-DEV.28"
    const _global_lock = ReentrantLock()
else
    const _global_lock = RecursiveSpinLock()
end

const _output_io = Ref{IO}()


function output_logging_msg(level::LogLevel, msg...)
    io = _output_io[]
    color = log_colors[level]
    prefix = log_prefix[level]
    printstyled(io, prefix, " ($(myid()), $(threadid())): "; bold = true, color=color)

    printstyled(io, chomp(string(msg...)) * "\n", color=color)
    Base.flush(io)
    nothing
end



export @enable_logging


function enable_logging_macro()
    quote
        _log_level = Base.Threads.Atomic{Int}(BAT.Logging.LOG_INFO)
    end
end

macro enable_logging()
    esc(enable_logging_macro())
end



# get_log_level_macro() = :(LogLevel(_log_level[]))
#
# macro get_log_level()
#     esc(get_log_level_macro())
# end
#
# set_log_level_macro(level) = :(_log_level[] = $level; $(get_log_level_macro()))
#
# macro set_log_level!(level)
#     esc(set_log_level_macro(level))
# end



export get_log_level, set_log_level!

get_log_level(m::Module) = m._log_level[]

function set_log_level!(m::Module, level::LogLevel)
    atomic_level = m._log_level
    atomic_level[] = level
    LogLevel(atomic_level[])
end



export @log_msg
export @log_error, @log_warning, @log_info, @log_debug, @log_trace


function logging_macro(level, msg)
    l_var = gensym("l_")
    m_var = gensym("m_")
    quote
        let $l_var = $level
            if _log_level[] >= Int($l_var) && $l_var != LOG_NONE
                # lock early in case message-generation code is not thread-safe
                lock(BAT.Logging._global_lock)
                try
                    let $m_var = $msg
                        BAT.Logging.output_logging_msg($l_var, $m_var)
                    end
                finally
                    unlock(BAT.Logging._global_lock)
                end
            end
        end
    end
end


macro log_msg(log_level, msg)
    esc(logging_macro(log_level, msg))
end


for (macro_name, log_level) in (
    (:log_error, LOG_ERROR),
    (:log_warning, LOG_WARNING),
    (:log_info, LOG_INFO),
    (:log_debug, LOG_DEBUG),
    (:log_trace, LOG_TRACE)
)

    @eval macro $macro_name(msg)
        esc(logging_macro($log_level, msg))
    end
end



function __init__()
    # STDERR object at runtime differs from precompilation:
    _output_io[] = stderr
end


end # module
