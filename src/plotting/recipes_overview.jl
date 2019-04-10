# This file is a part of BAT.jl, licensed under the MIT License (MIT).

@recipe function f(samples::DensitySampleVector; 
                params=collect(1:5), 
                mean=false,
                std_dev=false,
                globalmode=false,
                localmode=false,
                diagonal = Dict(),
                upper = Dict(),
                lower = Dict(),
                param_labels = [])

    
    parsel = params
    parsel = parsel[parsel .<= Base.size(samples.params[1], 1)]


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

            seriestype --> get(diagonal, "seriestype", :smallest_intervals)
            bins --> get(diagonal, "bins", 200)
            colors --> get(diagonal, "colors", standard_colors)
            intervals --> get(diagonal, "intervals", standard_confidence_vals)
            legend --> get(diagonal, "legend", false)
            mean --> get(diagonal, "mean", mean)
            std_dev --> get(diagonal, "std_dev", std_dev)
            globalmode --> get(diagonal, "globalmode", globalmode)
            localmode --> get(diagonal, "localmode", localmode)
        
            xguide --> param_labels[i]
            yguide --> param_labels_y[i]
            
            samples, (parsel[i])
        end

        
        # upper right plots
        for j in i+1:nparams

            @series begin
                subplot := j + (i-1)*nparams

                seriestype --> get(upper, "seriestype", :histogram)
                bins --> get(upper, "bins", 200)
                colors --> get(upper, "colors", standard_colors)
                intervals --> get(upper, "intervals", standard_confidence_vals)
                legend --> get(upper, "legend", false)
                mean --> get(upper, "mean", mean)
                std_dev --> get(upper, "std_dev", std_dev)
                globalmode --> get(upper, "globalmode", globalmode)
                localmode --> get(upper, "localmode", localmode)
                xguide --> param_labels[i]
                yguide --> param_labels[j]

                samples, (parsel[i], parsel[j])
            end

            # lower left plots
            @series begin
                subplot := i + (j-1)*nparams

                seriestype --> get(lower, "seriestype", :smallest_intervals)
                bins --> get(lower, "bins", 200)
                colors --> get(lower, "colors", standard_colors)
                intervals --> get(lower, "intervals", standard_confidence_vals)
                legend --> get(lower, "legend", false)
                mean --> get(lower, "mean", mean)
                std_dev --> get(lower, "std_dev", std_dev)
                globalmode --> get(lower, "globalmode", globalmode)
                localmode --> get(lower, "localmode", localmode)
                xguide --> param_labels[i]
                yguide --> param_labels[j]

                samples, (parsel[i], parsel[j])
            end

        end 
    end 

end
