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
posterior_gauss2D = PosteriorDensity(gauss2D, prior_gauss2D)
################################################################################################

########################################multi cauchy###########################################
name_multi_cauchy2D = "multi cauchy"
multi_cauchy2D = let
    params -> begin
        return LogDVal(log(1/0.936429)+logpdf(BAT.MultimodalCauchy(µ=5.,σ=4.,n=1),[params.x,params.y]))
    end
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

################function funnel 2D##############################################################
name_funnel2D = "funnel"
funnel2D = let a=1., b=0.5
    params -> begin
        return LogDVal(log(1/0.2288974)+logpdf(BAT.FunnelDistribution(a,b,3),[params.x,params.x,params.y]))
    end
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
################################################################################################

posterior2D=[
            posterior_gauss2D,
            posterior_multi_cauchy2D,
            posterior_funnel2D
        ]

name2D=[
            name_gauss2D,
            name_multi_cauchy2D,
            name_funnel2D
]

analytical_stats2D=[
            analytical_stats_gauss2D,
            analytical_stats_multi_cauchy2D,
            analytical_stats_funnel2D
]

sample_stats2D=[Vector{Any}(undef,length(stats_names2D)) for i in 1:length(name2D)]
run_stats2D=[Vector{Any}(undef,length(run_stats_names2D)) for i in 1:length(name2D)]
