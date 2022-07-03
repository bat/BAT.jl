# This file is a part of BAT.jl, licensed under the MIT License (MIT).


function markdown_append!(md::Markdown.MD, content::AbstractString)
    push!(md.content, Markdown.parse(content))
    return md
end


_table_columnnames(tbl) = keys(Tables.columns(tbl))
_default_table_headermap(tbl) = Dict(k => string(k) for k in _table_columnnames(tbl))

_tbl_cell_content(x) = x
_tbl_cell_content(x::Symbol) = string(x)
_tbl_cell_content(x::Expr) = string(x)

function _tbl_cell_content(x::Union{Number,Interval,Array})
    buf = IOBuffer()
    io = IOContext(buf, :compact => true)
    show(io, x)
    String(take!(buf))
end
    
function markdown_table(
    tbl;
    headermap::Dict{Symbol,<:AbstractString} = _default_table_headermap(tbl),
    align::AbstractVector{Symbol} = fill(:l, length(Tables.columns(tbl)))
)
    content = Vector{Any}[Any[headermap[k] for k in keys(Tables.columns(tbl))]]
    for r in Tables.rows(tbl)
        push!(content, [_tbl_cell_content(x) for x in values(r)])
    end
    Markdown.Table(content, align)
end
