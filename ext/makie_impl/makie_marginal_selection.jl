# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Input limits also admit the canonical serialization of every supported selection.
# Use wide arithmetic before converting so even extreme configuration values cannot wrap.
_marginal_text_limit(N::Integer, capacity::Integer) =
    Int(min(typemax(Int), max(Int128(4096), 4 * Int128(capacity) * (ndigits(N) + 2))))

_marginal_error(title, message) = (title = title, message = message)
_marginal_unavailable(N) = _marginal_error("Dimension is unavailable",
    "Use whole-number dimension indices from 1 to $N.")
_marginal_capacity_error(capacity) = _marginal_error("Selection limit",
    "Choose at most $capacity distinct dimensions, including those in pairs. " *
    "The limit is set when the figure is created.")

struct _MarginalParseResult
    dimensions::Vector{Int}
    duplicates::Bool
    has_pairs::Bool
    error::Union{Nothing,NamedTuple{(:title, :message),Tuple{String,String}}}
end

_marginal_parse_failure(error) = _MarginalParseResult(Int[], false, false, error)

# A small data grammar, never Julia syntax evaluation. Pair notation selects the union
# of its dimensions: the existing corner plot continues to show all combinations.
function _parse_marginal_dimensions(source::AbstractString, N::Integer, capacity::Integer)
    ncodeunits(source) <= _marginal_text_limit(N, capacity) ||
        return _marginal_parse_failure(_marginal_error("Input is too long", "Enter a shorter dimension list."))
    text = strip(source)
    if startswith(text, '[')
        endswith(text, ']') || return _marginal_parse_failure(
            _marginal_error("Unclosed list", "Close the list with ]."))
        text = strip(SubString(text, nextind(text, firstindex(text)), prevind(text, lastindex(text))))
    end
    selected = Set{Int}()
    duplicates = false
    has_pairs = false
    pos = firstindex(text)
    while pos <= lastindex(text)
        rest = SubString(text, pos)
        pair = match(r"^\(\s*([0-9]+)\s*,\s*([0-9]+)\s*\)", rest)
        range = isnothing(pair) ? match(r"^([0-9]+)\s*:\s*([0-9]+)", rest) : nothing
        single = isnothing(pair) && isnothing(range) ? match(r"^([0-9]+)", rest) : nothing
        token = !isnothing(pair) ? pair : !isnothing(range) ? range : single
        isnothing(token) && return _marginal_parse_failure(_marginal_error("Cannot read this input",
            "Use a list such as 2, 7, 8; a range such as 2:4; or pairs such as (2,8)."))
        values = map(s -> tryparse(Int, s), token.captures)
        any(v -> isnothing(v) || !(1 <= v <= N), values) &&
            return _marginal_parse_failure(_marginal_unavailable(N))
        if !isnothing(range)
            lo, hi = values
            lo <= hi || return _marginal_parse_failure(
                _marginal_error("Range is reversed", "Use an increasing range, such as $hi:$lo."))
            hi - lo < capacity || return _marginal_parse_failure(_marginal_capacity_error(capacity))
            values = lo:hi
        end
        has_pairs |= !isnothing(pair)
        for value in values
            duplicates |= value in selected
            push!(selected, value)
            length(selected) <= capacity || return _marginal_parse_failure(_marginal_capacity_error(capacity))
        end
        pos += ncodeunits(token.match)
        pos > lastindex(text) && break
        separator = match(r"^(?:\s*[,;]\s*|\s+)", SubString(text, pos))
        isnothing(separator) && return _marginal_parse_failure(_marginal_error("Missing separator",
            "Separate dimensions or pairs with spaces, commas, or semicolons."))
        pos += ncodeunits(separator.match)
        pos > lastindex(text) && return _marginal_parse_failure(_marginal_error("Missing dimension",
            "Enter a dimension after the final separator, or remove the separator."))
    end
    return _MarginalParseResult(sort!(collect(selected)), duplicates, has_pairs, nothing)
end


function _marginal_slot_count(N::Integer)
    N == 0 && return 0
    header = max(26, 7 * ndigits(N) + 10)
    height = N > 5 ? 220 : 156
    return min(N, 6, max(min(3, N), fld(height - header, header + 4)))
end


# Construction values are trusted after the existing clamp: preserve their order and
# multiplicity for reset. New user edits use unique sorted dimension sets.
mutable struct _MarginalSelection
    N::Int
    capacity::Int
    initial_selection::Vector{Int}
    initial_headers::Vector{Int}
    selection::Vector{Int}
    headers::Vector{Int}
    revision::UInt64
end

function _MarginalSelection(N::Integer, capacity::Integer, initial_selection::AbstractVector{<:Integer})
    selected = Int[i for i in initial_selection if 1 <= i <= N]
    resize!(selected, min(length(selected), capacity))
    headers = collect(1:_marginal_slot_count(N))
    return _MarginalSelection(N, capacity, copy(selected), copy(headers), selected, headers, 0)
end

function _accept_marginal_intent!(state::_MarginalSelection, selection, headers = state.headers)
    state.revision += 1
    state.selection = copy(selection)
    state.headers = copy(headers)
    return state.revision
end

function _marginal_header_candidate(state::_MarginalSelection, slot::Integer, source::AbstractString)
    text = strip(source)
    value = occursin(r"^[0-9]+$", text) ? tryparse(Int, text) : nothing
    if isnothing(value) || !(1 <= value <= state.N)
        return (selection = state.selection, headers = state.headers, error = _marginal_unavailable(state.N))
    end
    duplicate = findfirst(i -> i != slot && state.headers[i] == value, eachindex(state.headers))
    if !isnothing(duplicate)
        return (selection = state.selection, headers = state.headers,
            error = _marginal_error("Dimension is already shown",
                "Dimension $value is assigned to matrix position $duplicate. Choose another dimension."))
    end
    headers = copy(state.headers)
    selected = Set(state.selection)
    if value != headers[slot]
        delete!(selected, headers[slot])
        push!(selected, value)
        headers[slot] = value
    end
    error = length(selected) <= state.capacity ? nothing : _marginal_capacity_error(state.capacity)
    return (selection = sort!(collect(selected)), headers = headers, error = error)
end

function _marginal_editor_headers(state::_MarginalSelection, selection)
    headers = unique(vcat(selection, state.headers))
    resize!(headers, min(length(headers), length(state.initial_headers)))
    return headers
end
