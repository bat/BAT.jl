using BAT
using Distributions, Test

@testset "mcmc_sampleid" begin
    mvnorm = @inferred(product_distribution([Normal(), Normal()]))
    sampling_method = @inferred(MCMCSampling(mcalg=MetropolisHastings(), nchains=2, trafo=NoDensityTransform(), nsteps=10^3))

    samples_1 = @inferred(bat_sample(mvnorm, sampling_method)).result
    samples_2 = @inferred(bat_sample(mvnorm, sampling_method)).result

    id_vector_1 = samples_1.info
    id_vector_2 = samples_2.info
    id_vector_12 = @inferred(merge(id_vector_1, id_vector_2))

    num_sample_ids_1 = @inferred(length(id_vector_1))
    num_sample_ids_2 = @inferred(length(id_vector_2))

    @test @inferred(length(id_vector_12)) == num_sample_ids_1 + num_sample_ids_2

    @test id_vector_12[1:num_sample_ids_1] == id_vector_1
    @test id_vector_12[num_sample_ids_1+1:end] == id_vector_2

    @test @inferred(isempty(@inferred(BAT.MCMCSampleIDVector())))

    merge!(id_vector_1, id_vector_2)
    @test id_vector_1 == id_vector_12
end