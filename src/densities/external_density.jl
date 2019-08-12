# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# External density protocol is big endian


const _bat_proto_default_msgtype = 0x42415430 # "BAT0"


abstract type BATProtocolMessage end


function Base.write(output::IO, message::BATProtocolMessage)
    T = typeof(message)

    buf = IOBuffer()
    tphash = _bat_proto_type_hash(T)
    write(buf, UInt32(tphash))
    _bat_proto_encode!(buf, message)
    data = take!(buf)
    len = length(data)

    # @debug "Sending message of length $len, type hash $tphash, type $T"
    write(output, UInt32(_bat_proto_default_msgtype))
    write(output, Int32(len))
    write(output, data)
    flush(output)
end


function Base.read(input::IO, ::Type{T}) where {T <: BATProtocolMessage}
    msgtype = read(input, UInt32)
    msgtype == _bat_proto_default_msgtype || throw(ErrorException("Unknown message type $msgtype"))
    len = read(input, Int32)
    data = Vector{UInt8}(undef, len)
    read!(input, data)

    buf = IOBuffer(data)
    tphash_recv = read(buf, UInt32)
    tphash_expected = _bat_proto_type_hash(T)
    tphash_recv == tphash_expected || throw(ErrorException("Received type hash $tphash_recv, but expected $tphash_expected"))
    # @debug "Received message of length $len, type hash $tphash_recv, type $T"
    _bat_proto_decode!(buf, T)
end




_bat_proto_encode!(buffer::IOBuffer, x::Any) = write(buffer, x)

_bat_proto_decode!(buffer::IOBuffer, ::Type{T}) where T = read(buffer, T)

function _bat_proto_encode!(buffer::IOBuffer, A::AbstractVector)
    len = length(eachindex(A))
    write(buffer, Int32(len))
    write(buffer, A)
end

function _bat_proto_decode!(buffer::IOBuffer, ::Type{T}) where {T<:Vector}
    len = read(buffer, Int32)
    T(len, undef)
    @assert length(eachindex(A)) == len
    read!(buffer, A)
    A
end



# fnv1a32 hash of type name
function _bat_proto_type_hash end


struct GetLogDensityValueDMsg <: BATProtocolMessage
    request_id::Int32
    density_id::Int32
    params::Vector{Float64}
end


_bat_proto_type_hash(::Type{GetLogDensityValueDMsg}) = UInt32(0x1164e043)

function _bat_proto_encode!(buffer::IOBuffer, x::GetLogDensityValueDMsg)
    _bat_proto_encode!(buffer, x.request_id)
    _bat_proto_encode!(buffer, x.density_id)
    _bat_proto_encode!(buffer, x.params)
end



struct LogDensityValueDMsg <: BATProtocolMessage
    request_id::Int32
    density_id::Int32
    log_density::Float64
end


_bat_proto_type_hash(::Type{LogDensityValueDMsg}) = UInt32(0xc0c6d511)

_bat_proto_decode!(buffer::IOBuffer, ::Type{LogDensityValueDMsg}) = LogDensityValueDMsg(
    _bat_proto_decode!(buffer, Int32),
    _bat_proto_decode!(buffer, Int32),
    _bat_proto_decode!(buffer, Float64)
)



struct ExternalDensity <: AbstractDensity
    n_par::Int
    cmd::Cmd
    density_id::Int
    proc::ThreadLocal{Base.Process}
    lock::ThreadLocal{ThreadSafeReentrantLock}
end

export ExternalDensity

function ExternalDensity(n_par::Int, cmd::Cmd, density_id = 0)
    proc = ThreadLocal{Base.Process}(undef)
    lock = ThreadLocal{ThreadSafeReentrantLock}(undef)
    all_procs = getallvalues(proc)
    all_locks = getallvalues(lock)
    for i in eachindex(all_locks)
        all_locks[i] = ThreadSafeReentrantLock()
    end
    ExternalDensity(n_par, cmd, density_id, proc, lock)
end


BAT.nparams(density::ExternalDensity) = density.n_par

function BAT.density_logval(density::ExternalDensity, params::AbstractVector{Float64})
    # TODO: Fix multithreading support

    result = Ref(NaN)
    lock(density.lock[]) do
        request_id = rand(0:typemax(Int32))
        req = GetLogDensityValueDMsg(request_id, density.density_id, params)
        # @debug "Sending request $req"
        if !isassigned(density.proc)
            @info "Starting external process $(density.cmd)"
            density.proc[] = open(density.cmd, read = true, write = true)
        end
        proc = density.proc[]
        write(proc, req)
        resp = read(proc, LogDensityValueDMsg)
        # @debug "Received response $resp"
        resp.request_id == req.request_id || throw(ErrorException("Unexpexted response id $(resp.request_id) for request id $(req.request_id)"))
        resp.density_id == req.density_id || throw(ErrorException("Unexpexted density_id $(resp.density_id) in response to requested id $(req.density_id)"))
        result[] = resp.log_density
    end
    result[]
end


function Base.close(density::ExternalDensity)
    let all_procs = getallvalues(density.proc)
        for i in eachindex(all_procs)
            @critical begin
                if isassigned(all_procs, i)
                    p = all_procs[i]
                    @info "Closing external process $i: $(p.cmd)"
                    close(p)
                end
            end
        end
    end
end
