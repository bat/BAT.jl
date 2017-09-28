# This file is a part of BAT.jl, licensed under the MIT License (MIT).


module Logging


@enum LogLevel LOG_NONE=0 LOG_ERROR=1 LOG_WARNING=2 LOG_INFO=3 LOG_DEBUG=4 LOG_TRACE=5 LOG_ALL=100
export LogLevel
export LOG_NONE, LOG_ERROR, LOG_WARNING, LOG_INFO, LOG_DEBUG, LOG_TRACE, LOG_ALL


import Base: +,-

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
    LOG_ERROR => "ERROR: ",
    LOG_WARNING => "WARNING: ",
    LOG_INFO => "INFO: ",
    LOG_DEBUG => "DEBUG: ",
    LOG_TRACE => "TRACE: ",
)


const _output_lock = Base.Threads.RecursiveSpinLock()
const _output_io = Ref{IO}()

function output_logging_msg(level::LogLevel, msg...)
    lock(_output_lock) do
        io = _output_io[]
        color = log_colors[level]
        prefix = log_prefix[level]
        Base.print_with_color(color, io, prefix; bold = true)
        Base.println_with_color(color, io, chomp(string(msg...)))
        nothing
    end
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
    quote
        let l = $level, m = $msg
            if _log_level[] >= Int(l) && l != LOG_NONE
                BAT.Logging.output_logging_msg(l, m)
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
    _output_io[] = STDERR
end


end # module
