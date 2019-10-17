# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(prior::NamedPrior; 
                params=collect(1:5), 
                diagonal = Dict(),
                upper = Dict(),
                lower = Dict(),
                param_labels = [])

    
    parsel = params
    parsel = parsel[parsel .<= length(keys(prior))]


    if Base.size(param_labels, 1) == 0
        param_labels = [latexstring("\\theta_$i") for i in parsel]
        param_labels_y = [latexstring("p(\\theta_$i)") for i in parsel]
    else
        param_labels_y = [latexstring("p("*param_labels[i]*")") for i in 1:length(param_labels)]
        param_labels = [latexstring(param_labels[i]) for i in 1:length(param_labels)]
    end

    nparams = length(parsel)   
    layout --> nparams^2
    size --> (1000, 600)
    

    for i in 1:nparams

        # diagonal
        @series begin
            subplot := i + (i-1)*nparams

            seriestype --> get(diagonal, "seriestype", :stephist)
            bins --> get(diagonal, "bins", 200)
            colors --> get(diagonal, "colors", standard_colors)
            intervals --> get(diagonal, "intervals", standard_confidence_vals)
            legend --> get(diagonal, "legend", false)
            xlims --> get(diagonal, "xlims", :auto)
            ylims --> get(diagonal, "ylims", :auto)
            linecolor --> get(diagonal, "linecolor", :black)
            xguide --> param_labels[i]
            yguide --> param_labels_y[i]
            
            prior, (parsel[i])
        end

        
        # upper right plots
        for j in i+1:nparams

            @series begin
                subplot := j + (i-1)*nparams

                seriestype --> get(upper, "seriestype", :smallest_intervals_contour)
                bins --> get(upper, "bins", 50)
                more_colors = []
                colors --> get(upper, "colors", colormap("Blues", 10))
                more_intervals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
                intervals --> get(upper, "intervals", more_intervals)
                legend --> get(upper, "legend", false)
                xlims --> get(upper, "xlims", :auto)
                ylims --> get(upper, "ylims", :auto)
                xguide --> param_labels[i]
                yguide --> param_labels[j]

                prior, (parsel[i], parsel[j])
            end

            # lower left plots
            @series begin
                subplot := i + (j-1)*nparams

                seriestype --> get(lower, "seriestype", :smallest_intervals_contour)
                bins --> get(lower, "bins", 50)
                colors --> get(lower, "colors", standard_colors)
                intervals --> get(lower, "intervals", standard_confidence_vals)
                legend --> get(lower, "legend", false)
                xlims --> get(lower, "xlims", :auto)
                ylims --> get(lower, "ylims", :auto)
                xguide --> param_labels[i]
                yguide --> param_labels[j]

                prior, (parsel[i], parsel[j])
            end

        end 
    end 

end
