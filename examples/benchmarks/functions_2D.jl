stats_names2D = ["mode","mean","var"]
run_stats_names2D = ["nsamples","nchains","Times"]
##########################################multi normal###########################################
sig = Matrix{Float64}([1.5^2 1.5*2.5*0.4 ; 1.5*2.5*0.4 2.5^2])
mu = [15,10]
name_gauss2D = "gaussian"
gauss2D = let #μ = [15, 10], σ = [1.5, 2.5]
    params -> begin
        return LogDVal(logpdf(MvNormal([15,10],sig),[params.x,params.y]))
    end
end

function analytical_function_gauss2D(x,y)
    sig = Matrix{Float64}([1.5^2 1.5*2.5*0.4 ; 1.5*2.5*0.4 2.5^2])
    return pdf(MvNormal([15,10],sig),[x,y])
end

prior_gauss2D = NamedTupleDist(
x = 0.0..30.0,
y = 0.0..30.0
)

analytical_stats_gauss2D = Vector{Any}(undef,length(stats_names2D))
analytical_stats_gauss2D[1] = [15,10]       #analytical_mode_gauss2D
analytical_stats_gauss2D[2] = [15,10]       #analytical_mean_gauss2D
analytical_stats_gauss2D[3] = [1.5^2,2.5^2] #analytical_var_gauss2D
analytical_integral_gauss2D = 1
posterior_gauss2D = PosteriorDensity(gauss2D, prior_gauss2D)
################################################################################################

########################################multi cauchy###########################################
name_multi_cauchy2D = "multi cauchy"
multi_cauchy2D = let
    params -> begin
        return LogDVal(logpdf(BAT.MultimodalCauchy(µ=5.,σ=4.,n=1),[params.x,params.y]))
    end
end

function analytical_function_multi_cauchy2D(x,y)
        return pdf(BAT.MultimodalCauchy(µ=5.,σ=4.,n=1),[x,y])
end

prior_multi_cauchy2D = NamedTupleDist(
x = -40.0..40.0,
y = -40.0..40.0
)

analytical_stats_multi_cauchy2D = Vector{Any}(undef,length(stats_names2D))
analytical_stats_multi_cauchy2D[1] = [[-5,-5],[-5,5],[5,-5],[5,5]]      #analytical_mode_multi_cauchy2D
analytical_stats_multi_cauchy2D[2] = [0,0]                              #analytical_mean_multi_cauchy2D
analytical_stats_multi_cauchy2D[3] = [71.98080,71.98080]                #analytical_var_multi_cauchy2D
analytical_integral_multi_cauchy2D = 0.936429
posterior_multi_cauchy2D = PosteriorDensity(multi_cauchy2D, prior_multi_cauchy2D)
################################################################################################

################function gaussian shell########################################################
name_gaussian_shell2D = "gaussian shell"
analytical_integral_gaussian_shell2D = 31.44111
gaussian_shell2D = let r=5, w=2, c=0
    params -> begin
        return LogDVal(log( ((1)/(sqrt(2*pi*w^2))) * exp(-((abs(sqrt(params.x^2+params.y^2)) - r)^2 / (2*w^2)) )  ))
    end
end

function analytical_function_gaussian_shell2D(x,y)
    return ((1)/(sqrt(2*pi*2^2))) * exp(-((abs(sqrt(x^2+y^2)) - 5)^2 / (2*2^2)) )
end

prior_gaussian_shell2D = NamedTupleDist(
        x = -25..25,
        y = -25..25
)

posterior_gaussian_shell2D = PosteriorDensity(gaussian_shell2D,prior_gaussian_shell2D)
analytical_stats_gaussian_shell2D = Vector{Any}(undef,length(stats_names2D))
q1 = [[sqrt(x),sqrt(25-x)] for x in 0:0.01:25]
q2 = [[sqrt(x),-sqrt(25-x)] for x in 0:0.01:25]
q3 = [[-sqrt(x),-sqrt(25-x)] for x in 0:0.01:25]
q4 = [[sqrt(x),sqrt(25-x)] for x in 0:0.01:25]
analytical_stats_gaussian_shell2D[1] = [q1...,q2...,q3...,q4...] #all modes within the max circle
#analytical_stats_gaussian_shell2D[1] = [[sqrt(x),sqrt(25-x)] for x in 0:0.01:25]
analytical_stats_gaussian_shell2D[2] = [0,0]
analytical_stats_gaussian_shell2D[3] = [18.485989,18.485989]
#analytical solution is really long
#29 + (5 sqrt(2/π))/e^(25/8) - 29/2 Q(1/2, 25/8) Q is regularized incomplete gamma function
################################################################################################

################function funnel 2D##############################################################
name_funnel2D = "funnel"
analytical_integral_funnel2D = 0.2288974
funnel2D = let a=1., b=0.5
    params -> begin
        return LogDVal(logpdf(BAT.FunnelDistribution(a,b,3),[params.x,params.x,params.y]))
    end
end

function analytical_function_funnel2D(x,y)
    return pdf(BAT.FunnelDistribution(1.,0.5,3),[x,x,y])
end

prior_funnel2D = NamedTupleDist(
    x = -10..10,
    y = -10..10
)

posterior_funnel2D = PosteriorDensity(funnel2D,prior_funnel2D)
analytical_stats_funnel2D = Vector{Any}(undef,length(stats_names2D))
analytical_stats_funnel2D[1] = [-0.443755,0.0]
analytical_stats_funnel2D[2] = [0.011632983,0.0]
analytical_stats_funnel2D[3] = [0.304725652,2.04919002]
analytical_stats_funnel2D = [-0.298002,0.0163721,0.314562]
################################################################################################


posterior2D=[
            posterior_gauss2D,
            posterior_multi_cauchy2D,
            posterior_gaussian_shell2D,
            posterior_funnel2D
        ]

name2D=[
            name_gauss2D,
            name_multi_cauchy2D,
            name_gaussian_shell2D,
            name_funnel2D
]

analytical_integral2D=[
            analytical_integral_gauss2D,
            analytical_integral_multi_cauchy2D,
            analytical_integral_gaussian_shell2D,
            analytical_integral_funnel2D
]

analytical_stats2D=[
            analytical_stats_gauss2D,
            analytical_stats_multi_cauchy2D,
            analytical_stats_gaussian_shell2D,
            analytical_stats_funnel2D
]

func2D=[
            analytical_function_gauss2D,
            analytical_function_multi_cauchy2D,
            analytical_function_gaussian_shell2D,
            analytical_function_funnel2D
]

sample_stats2D=[Vector{Any}(undef,length(stats_names2D)) for i in 1:length(name2D)]
run_stats2D=[Vector{Any}(undef,length(run_stats_names2D)) for i in 1:length(name2D)]
