########1D functions for benchmarking###########
################################################
normal = (posterior=Normal(0,1),mode=0,mean=0,var=1,chi2=[9999],ks=[9999.],ahmi=[9999.])
exp1 = (posterior= Exponential(1),mode=0,mean=1,var=1,chi2=[9999],ks=[9999.],ahmi=[9999.])
cauchy = (posterior=Cauchy(0,1),mode=0,mean=0,var=Inf,chi2=[9999],ks=[9999.],ahmi=[9999.])
#cauchy_variance_for_edges = x -> 0.6365969042714238*x+-0.9834816540023621

funnel = (posterior=BAT.FunnelDistribution(1.2,0.,1),mode=0,mean=0,var=2.07350,chi2=[9999],ks=[9999.],ahmi=[9999.])
multi_cauchy = (posterior=BAT.MultimodalCauchy(µ=5.,σ=2.,n=1),mode=[-5,5],mean=0,var=Inf,chi2=[9999],ks=[9999.],ahmi=[9999.])

testfunctions_1D = Dict(
	"normal" 		=> normal,
	"exp"			=> exp1,
	"cauchy"		=> cauchy,
	#"multi_cauchy"	=> multi_cauchy,
	"funnel"		=> funnel
)

analytical_stats_name=["mode","mean","var"]	#could be taken into NamedTuple for easier addtions but would be needed to implmented into calcs anyway
sample_stats=[Vector{Float64}(undef,length(analytical_stats_name)) for i in 1:length(testfunctions_1D)]

run_stats_names = ["nsteps","nchains","Times"]
run_stats=[Vector{Float64}(undef,length(run_stats_names)) for i in 1:length(testfunctions_1D)]
