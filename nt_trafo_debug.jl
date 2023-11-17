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


using BAT: apply_dist_trafo, _flat_ntd_eff_accessors, _stdmv_to_flat_ntdistelem, _flat_ntd_orig_accessors,
    eff_totalndof, StdMvDist

trg_d = prior;
src_d = BAT.StandardMvUniform(totalndof(varshape(trg_d)))
src_v = rand(src_d)

nms = propertynames(trg_d)


trg_dists = values(trg_d)
N = length(trg_dists)
n_vars = [Symbol("n$i") for i in 1:N]
d_vars = [Symbol("d$i") for i in 1:N]
r_vars = [Symbol("r$i") for i in 1:N]
i = 1

_resize_stdmv(::T, n::Int) where {T<:StdMvDist} = T(n)

quote
    src_v_pos = firstindex(src_v)
end
quote
    $(d_vars[i]) = trg_dists[$i]
    $(n_vars[i]) = eff_totalndof($(d_vars[i]))
    $(r_vars[i]) = apply_dist_trafo($(d_vars[i]), _resize_stdmv(src_d, $(n_vars[i])), view(src_v, src_v_pos:src_v_pos+$(n_vars[i])-1))
    src_v_pos += $(n_vars[i])
end


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


