# This file is a part of BAT.jl, licensed under the MIT License (MIT).


"""
    bat_report(obj...)::Markdown.MD

Generate a report on the given object(s).

Specialize [`bat_report!`](@ref) instead of `bat_report`.
"""
function bat_report end
export bat_report

function bat_report(obj...)
    md = Markdown.MD()
    for o in obj
        bat_report!(md, o)
    end
    return md
end


"""
    BAT.bat_report!(md::Markdown.MD, obj)

Add report on `obj` to `md`.

See [`bat_report`](@ref).
"""
function bat_report! end


function marginal_table(smplv::DensitySampleVector)
    parnames = map(string, all_active_names(elshape(smplv.v)))

    usmplv = unshaped.(smplv)

    credible_intervals = smallest_credible_intervals(usmplv)

    mhist = hist_unicode.(marginal_histograms(usmplv))

    TypedTables.Table(
        parameter = parnames,
        mean = mean(usmplv),
        std = std(usmplv),
        global_mode = mode(usmplv),
        marginal_mode = bat_marginalmode(usmplv).result,
        credible_intervals = credible_intervals,
        marginal_histogram = mhist,
    )
end


function fixed_parameter_table(smplv::DensitySampleVector)
    vs = elshape(smplv.v)
    parkeys = Symbol.(get_fixed_names(vs))
    parvalues = [getproperty(vs, f).shape.value for f in parkeys]
    TypedTables.Table(parameter = parkeys, value = string.(parvalues))
end


function bat_report!(md::Markdown.MD, smplv::DensitySampleVector)
    usmplv = unshaped.(smplv)
    nsamples = length(eachindex(smplv))
    total_weight = sum(smplv.weight)
    ess = round.(Int, bat_eff_sample_size(usmplv).result)

    markdown_append!(md, """
    ### Sampling result
    """)

    markdown_append!(md, """
    * Total number of samples: $nsamples
    * Total weight of samples: $total_weight
    * Effective sample size: between $(minimum(ess)) and $(maximum(ess))

    #### Marginals
    """)

    only_one_ci(viv::AbstractVector{<:Interval}) = length(viv) == 1 ? only(viv) : :multiple

    marg_tbl = marginal_table(smplv)
    mod_marg_tbl = merge(Tables.columns(marg_tbl), (credible_intervals = map(only_one_ci, marg_tbl.credible_intervals),))
    marg_headermap = Dict(:parameter => "Parameter", :mean => "Mean", :std => "Std. dev.", :global_mode => "Gobal mode", :marginal_mode => "Marg. mode", :credible_intervals => "Cred. interval", :marginal_histogram => "Histogram")
    push!(md.content, BAT.markdown_table(Tables.columns(mod_marg_tbl), headermap = marg_headermap, align = [:l, :l, :l, :l, :l, :c, :l]))

    fixed_tbl = fixed_parameter_table(smplv)
    if !isempty(fixed_tbl)
        markdown_append!(md, """
        ### Fixed parameters
        """)
        marg_headermap = Dict(:parameter => "Parameter", :value => "Value")
        push!(md.content, BAT.markdown_table(fixed_tbl, headermap = marg_headermap, align = [:l, :l]))
    end

    return md
end
