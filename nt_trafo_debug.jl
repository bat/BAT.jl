using BAT, ValueShapes, Distributions
using InverseFunctions, DensityInterface
N_total = 500 # total number of parameters
N_free = 50   # number of free parameters
N_samples = 10^4 
# create a BAT prior with N_total parameters, where the first N_free are free and the rest is fixed to 0
parameter_names = [Symbol("p_$i") for i in 1:N_total]
parameter_dists = []
for i in eachindex(parameter_names)
    if i <= N_free
        push!(parameter_dists, Normal(0, 1))
    else 
        push!(parameter_dists, 0.)
    end
end
parameter_nt = NamedTuple{tuple(parameter_names...)}(parameter_dists)
prior = BAT.distprod(parameter_nt,)
# create some random samples and put into DensitySampleVector
rand_samples = [rand(Normal(0., 1.), N_free) for i in 1:N_samples]
sample_id = fill(BAT.MCMCSampleID(1, 0, 0, 0), N_samples)
samples = DensitySampleVector(rand_samples, rand(N_samples), info=sample_id)
# get the trafo
density_notrafo = convert(BAT.AbstractMeasureOrDensity, prior);
density, trafo = BAT.transform_and_unshape(PriorToGaussian(), density_notrafo, get_batcontext());
shape = varshape(density)
samples_trafo = shape.(samples)
# apply inverse trafo 
samples_notrafo = inverse(trafo).(samples_trafo)
#@profview inverse(trafo).(samples_trafo)

using BAT: apply_dist_trafo
trg_d = prior
src_d = BAT.StandardMvUniform(totalndof(varshape(trg_d)))

apply_dist_trafo(unshaped(trg_d), src_d, src_v)
