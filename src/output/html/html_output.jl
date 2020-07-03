function Base.show(io::IO, m::MIME"text/html", prior::ConstDensity{<:HyperRectBounds})
    print(io, "HyperRectBounds")
    
   for p in 1:length(prior.bounds.vol.hi)
        print(io, "<br> &nbsp; $p. ")
        print(io, prior.bounds.bt[p])
        println(io, " [", prior.bounds.vol.lo[p],", ", prior.bounds.vol.hi[p],"]")
    end 
end


function Base.show(io::IO, m::MIME"text/html", algorithm::MetropolisHastings)
    print(io, "MetropolisHastings")
end


function Base.show(io::IO, m::MIME"text/html", summary::Summary)

    stats = summary.stats
    nparams = length(stats.param_stats.mean)

    color1 = "#f2f2f2"; color2 = "#f7f7f7"; color3 = "#e5e5e5";

    width1 = 200; width2 = 480;
    totalwidth = width1 + width2 

    width3 = width1-20;  width4 = 120;
    totalwidth2 = width3 + width4 


    print(io, """<h1><span style="text-decoration: underline;"> BAT.jl - Summary </span></h1>""")


    # Model
    print(io, """<h2>Model</h2>""")
    print(io, """ 
        <table style="height: 69px; font-size:100%" width="$(width1+width2)">
        <tbody>
        <tr style="background-color: $(color1)" >
        <td style="width: $(width1)px;"><strong><p>likelihood:</p></strong></td>
        <td style="width: $(width2)px;"><p>$(summary.chainresults[1].spec.model.likelihood)</p></td>
        </tr>
        <tr style="background-color: $(color2)" >
        <td valign="top" style="width: $(width1)px;"><strong><p>prior:</strong></td>
        <td style="width: $(width2)px;"><p>"""
        )

    display_rich(io, m, summary.chainresults[1].spec.model.prior)

    print(io, """</p></td></tr></tbody></table>""")


    # Sampling
    print(io, """<h2>Sampling</h2>""")
    print(io, """ 
        <table style="height: 99px; font-size:100%" width="$(totalwidth)">
        <tbody>
        <tr style="background-color: $(color1)" >
        <td style="width: $(width1)px;"> <strong><p>algorithm:</p></strong></td>
        <td style="width: $(width2)px;"><p>"""
        )

    display_rich(io, m, summary.chainresults[1].spec.algorithm)

    print(io, """</p></td>
        </tr>
        <tr style="background-color: $(color2)" >
        <td style="width: $(width1)px;"><strong><p>number of chains:</p></strong></td>
        <td style="width: $(width2)px;"><p>$(length(summary.chainresults))</p></td>
        </tr>
        <tr style="background-color: $(color1)" >
        <td style="width: $(width1)px;"><strong><p>total number of samples:</p></strong></td>
        <td style="width: $(width2)px;"><p> $(stats.param_stats.cov.n)</p></td>
        </tr>
        </tbody>
        </table>"""
        )


    # Results
    print(io, """<h3 style="margin-left: 10px">Results</h3>""")

    for p in 1:nparams

        print(io, """ 
            <table style="margin-left: 20px; font-size:100%" width="$(totalwidth2)"; >
            <tbody>"""
        )

        print(io, """<tr style="background-color: $(color3)" >
            <td style="width: $(width3)px;"><strong><p>parameter $p</p></strong></td>
            <td style="width: $(width4)px;"><p> &nbsp; </p></td>
            </tr>"""
        )

        print(io, """<tr style="background-color: $(color1)" >
            <td style="width: $(width3)px;"><strong><p>mean ± std.dev.</p></strong></td>
            <td style="width: $(width4)px;"><p> $(@sprintf("%.3f",stats.param_stats.mean[p])) ± $(@sprintf("%.3f", sqrt(stats.param_stats.cov[p, p]))) </p></td>
            </tr>"""
        )

        print(io, """<tr style="background-color: $(color2)" >
            <td style="width: $(width3)px;"><strong><p>global mode</p></strong></td>
            <td style="width: $(width4)px;"><p> $(@sprintf("%.3f",stats.mode[p])) </p></td>
            </tr>"""
        )

        print(io, """ </tbody> </table> <p> &nbsp; </p>""")     
    end


    print(io, """<table style="margin-left: 20px; font-size:100%"> <tbody>""")
    print(io, """<tr style="background-color: $(color3)" > """)
    print(io, """ <td colspan="$(size(stats.param_stats.cov, 1))"><p> <strong> covariance matrix </strong></p></td>""")

    for i in 1:nparams
        if(i%2 == 0)
            print(io, """<tr style="background-color: $(color2)" > """)
        else
            print(io, """<tr style="background-color: $(color1)" > """)
        end

        for j in 1:nparams
            s = @sprintf("%.3f", stats.param_stats.cov[i, j])
            print(io, """ <td><p>$s</p></td>""")
        end
    end

    print(io, """ </tbody> </table> <p>&nbsp;</p> """) 



end
