stats_names2D = ["mode","mean","var"]
run_stats_names2D = ["nsamples","nchains","Times"]
##########################################multi normal###########################################
sig = Matrix{Float64}([1.5^2 1.5*2.5*0.4 ; 1.5*2.5*0.4 2.5^2])
analytical_stats_gauss2D = Vector{Any}(undef,length(stats_names2D))
analytical_stats_gauss2D[1] = [15,10]       #analytical_mode_gauss2D
analytical_stats_gauss2D[2] = [15,10]       #analytical_mean_gauss2D
analytical_stats_gauss2D[3] = [1.5^2,2.5^2] #analytical_var_gauss2D
gauss2D = (posterior=MvNormal([15,10],sig),mode=analytical_stats_gauss2D[1],mean=analytical_stats_gauss2D[2],var=analytical_stats_gauss2D[3],chi2=[9999],ks=[9999.,9999.],ahmi=[9999.,9999.])

########################################multi cauchy###########################################
analytical_stats_multi_cauchy2D = Vector{Any}(undef,length(stats_names2D))
analytical_stats_multi_cauchy2D[1] = [[-5,-5],[-5,5],[5,-5],[5,5]]      #analytical_mode_multi_cauchy2D
analytical_stats_multi_cauchy2D[2] = [0,0]                              #analytical_mean_multi_cauchy2D
analytical_stats_multi_cauchy2D[3] = [Inf,Inf]                #analytical_var_multi_cauchy2D
multi_cauchy2D = (posterior=BAT.MultimodalCauchy(µ=5.,σ=4.,n=2),mode=analytical_stats_multi_cauchy2D[1],mean=analytical_stats_multi_cauchy2D[2],var=analytical_stats_multi_cauchy2D[3],chi2=[9999],ks=[9999.,9999.],ahmi=[9999.,9999.])

################function funnel 2D##############################################################
analytical_stats_funnel2D = Vector{Any}(undef,length(stats_names2D))
analytical_stats_funnel2D[1] = [-1.0,0.0]
analytical_stats_funnel2D[2] = [0.0,0.0]
analytical_stats_funnel2D[3] = [1.0,7.406718]
funnel2D = (posterior=BAT.FunnelDistribution(1.,0.5,2),mode=analytical_stats_funnel2D[1],mean=analytical_stats_funnel2D[2],var=analytical_stats_funnel2D[3],chi2=[9999],ks=[9999.,9999.],ahmi=[9999.,9999.])

################################################################################################
testfunctions_2D = Dict(
	"normal" 		=> gauss2D,
	"multi cauchy"	=> multi_cauchy2D,
	"funnel"		=> funnel2D
)

sample_stats2D=[Vector{Any}(undef,length(stats_names2D)) for i in 1:length(testfunctions_2D)]
run_stats2D=[Vector{Any}(undef,length(run_stats_names2D)) for i in 1:length(testfunctions_2D)]
