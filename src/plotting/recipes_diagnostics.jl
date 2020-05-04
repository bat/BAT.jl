# This file is a part of BAT.jl, licensed under the MIT License (MIT).
struct MCMCDiagnostics
    samples::DensitySampleVector{<:AbstractVector{<:Real}}
    chainresults::Array{}
end

MCMCDiagnostics(samples::DensitySampleVector, chainresults = []) =
    MCMCDiagnostics(unshaped.(samples), chainresults)


@recipe function f(mcmc::MCMCDiagnostics;
                    vsel = collect(1 : size(mcmc.samples.v.data, 1)),
                    chains = collect(1 : size(mcmc.chainresults, 1)),
                    diagnostics = [:histogram, :kde, :trace, :acf],
                    trace = Dict(),
                    kde = Dict(),
                    acf = Dict(),
                    histogram = Dict(),
                    description = true
                    )

    seriestype = get(plotattributes, :seriestype, :histogram)

    nparams = length(vsel)
    nchains = length(chains)
    ndiagnostics = length(diagnostics)
    description ? ndescription = 1 : ndescription = 0

    size --> (ndiagnostics*350+100, nparams*nchains*200)
    layout --> Main.Plots.grid(nparams*nchains, ndiagnostics+ndescription)

    ctr = 1

	unique_chain_ids = sort(unique(mcmc.samples.info.chainid))

    for chain in chains

        #indices of samples from current chain
        r = mcmc.samples.info.chainid .== unique_chain_ids[chain]

        for p in vsel
            if description
                @series begin
                    subplot := ctr
                    label --> ""
                    annotations --> (1.5, 2.5, "Chain $chain  \n Parameter $p \n $(mcmc.chainresults[chain].nsamples) Samples")
                    grid --> false
                    xaxis --> false
                    yaxis --> false
                    [1,1], [2, 2]
                end
                ctr +=1
            end

            # samples from current chain
            s = flatview(mcmc.samples[r].v)[p, :]

            for d in diagnostics

                # samples histogram
                if d == :histogram

                    chain_samples = mcmc.samples[r]

                    @series begin
                        subplot := ctr
                        seriestype --> get(histogram, "seriestype", :stephist)
                        intervals --> get(histogram, "intervals", standard_confidence_vals)
                        bins --> get(histogram, "bins", 200)
                        normalize --> get(histogram, "normalize", true)
                        colors --> get(histogram, "colors", standard_colors)
                        mean --> get(histogram, "mean", false)
                        std --> get(histogram, "std", false)
                        globalmode --> get(histogram, "globalmode", false)
                        localmode --> get(histogram, "localmode", false)
                        legend --> get(histogram, "legend", true)

                        label --> get(histogram, "label", "")
                        title --> get(histogram, "title", "Histogram")
                        xguide --> get(histogram, "xlabel", "\$\\theta_$(p)\$")
                        yguide --> get(histogram, "ylabel", "\$p(\\theta_$(p))\$")

                        (chain_samples, p)
                    end


                # trace plot
                elseif d == :trace
                    x = collect(1:1:length(s))

                    @series begin
                        subplot := ctr
                        seriestype --> get(trace, "seriestype", :line)
                        label --> get(trace, "label", "")
                        title --> get(trace, "title", "Trace")
                        xguide --> get(trace, "xlabel", "Iterations")
                        yguide --> get(trace, "ylabel", "")
                        seriescolor --> get(trace, "seriescolor", :dodgerblue)
                        linecolor --> get(trace, "linecolor", :dodgerblue)
                        linestyle --> get(trace, "linestyle", :solid)
                        linewidth --> get(trace, "linewidth", 1)
                        linealpha --> get(trace, "linealpha", 1)
                        seriesalpha --> get(trace, "seriesalpha", 1)
                        markershape --> get(trace, "markershape", :none)
                        markersize --> get(trace, "markersize", 1)
                        markeralpha --> get(trace, "markeralpha", 1)
                        markercolor --> get(trace, "markercolor", :dodgerblue)
                        markerstrokealpha --> get(trace, "markerstrokealpha", 1)
                        markerstrokecolor --> get(trace, "markerstrokecolor", :dodgerblue)
                        markerstrokestyle --> get(trace, "markerstrokestyle", :solid)
                        markerstrokewidth --> get(trace, "markerstrokewidth", 1)
                        legend --> get(trace, "legend", true)

                        x, s
                    end


                # kernel density estimate
                elseif d == :kde

                    npoints = get(kde, "npoints", 2048)
                    bandwidth = get(kde, "bandwidth", KernelDensity.default_bandwidth(s))
                    boundary = get(kde, "boundary", KernelDensity.kde_boundary(s, bandwidth))
                    kernel = get(kde, "kernel", Distributions.Normal)

                    k = KernelDensity.kde(
                        s,
                        bandwidth = bandwidth,
                        boundary = boundary,
                        npoints = npoints,
                        kernel = kernel
                    )

                    @series begin
                        subplot := ctr
                        seriestype --> get(kde, "seriestype", :line)
                        label --> get(kde, "label", "")
                        title --> get(kde, "title", "KDE")
                        xguide --> get(kde, "xlabel", "\$\\theta_$(p)\$")
                        yguide --> get(kde, "ylabel", "\$p(\\theta_$(p))\$")
                        seriescolor --> get(kde, "seriescolor", :dodgerblue)
                        linecolor --> get(kde, "linecolor", :dodgerblue)
                        linestyle --> get(kde, "linestyle", :solid)
                        linewidth --> get(kde, "linewidth", 1)
                        linealpha --> get(kde, "linealpha", 1)
                        seriesalpha --> get(kde, "seriesalpha", 1)
                        markershape --> get(kde, "markershape", :none)
                        markersize --> get(kde, "markersize", 1)
                        markeralpha --> get(kde, "markeralpha", 1)
                        markercolor --> get(kde, "markercolor", :dodgerblue)
                        markerstrokealpha --> get(kde, "markerstrokealpha", 1)
                        markerstrokecolor --> get(kde, "markerstrokecolor", :dodgerblue)
                        markerstrokestyle --> get(kde, "markerstrokestyle", :solid)
                        markerstrokewidth --> get(kde, "markerstrokewidth", 1)
                        legend --> get(kde, "legend", true)

                        k.x, k.density
                    end


                # autocorrelation function
                elseif d == :acf

                    lags = get(acf, "lags", :none)
                    demean = get(acf, "demean", true)
                    lags != :none ? autocorr = StatsBase.autocor(s, lags, demean=demean) : autocorr = StatsBase.autocor(s, demean=demean)

                    @series begin
                        subplot := ctr

                        seriestype --> get(acf, "seriestype", :bar)
                        label --> get(acf, "label", "")
                        title --> get(acf, "title", "ACF")
                        xguide --> get(acf, "xlabel", "Lags")
                        yguide --> get(acf, "ylabel", "")
                        seriescolor --> get(acf, "seriescolor", :dodgerblue)
                        linecolor --> get(acf, "linecolor", :dodgerblue)
                        linestyle --> get(acf, "linestyle", :solid)
                        linewidth --> get(acf, "linewidth", 0)
                        linealpha --> get(acf, "linealpha", 0)
                        seriesalpha --> get(acf, "seriesalpha", 1)
                        markershape --> get(acf, "markershape", :none)
                        markersize --> get(acf, "markersize", 1)
                        markeralpha --> get(acf, "markeralpha", 1)
                        markercolor --> get(acf, "markercolor", :dodgerblue)
                        markerstrokealpha --> get(acf, "markerstrokealpha", 1)
                        markerstrokecolor --> get(acf, "markerstrokecolor", :dodgerblue)
                        markerstrokestyle --> get(acf, "markerstrokestyle", :solid)
                        markerstrokewidth --> get(acf, "markerstrokewidth", 1)
                        legend --> get(acf, "legend", true)
                        lags != :none ? (lags, autocorr) : (0-0.5:1:length(autocorr)-0.5, autocorr)
                    end
                end
                ctr +=1
            end
        end
    end
end
