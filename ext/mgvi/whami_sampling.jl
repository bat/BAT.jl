# This file is a part of BAT.jl, licensed under the MIT License (MIT).



function BAT.bat_sample_impl(m::BATMeasure, algorithm::WHAMISampling, context::BATContext)
    start_time = time()
    log_time = start_time
    (; pretransform, mapalg, ninit, nsamples, mineff, maxcycles) = algorithm


    transformed_m, f_pretransform = transform_and_unshape(pretransform, m, context)
    transformed_m_uneval = unevaluated(transformed_m)

    likelihood, prior = getlikelihood(transformed_m_uneval), getprior(transformed_m_uneval)
    if !is_std_mvnormal(prior)
        throw(ArgumentError("$(nameof(typeof(algorithm))) can't be used for measures that do not have a standard multivariate normal prior after `pretransform`"))
    end

    f_model, obs = try
        BAT._get_model(likelihood), BAT._get_observation(likelihood)
    catch err
        if err isa MethodError
            throw(ArgumentError("$(nameof(typeof(algorithm))) requires a likelihood based on a forward model and observed data, but don't know how to extract them from a likelihood of type $(nameof(typeof(likelihood)))."))
        else
            rethrow()
        end
    end

    pstr = transformed_m
    (;approx_dist, samples_p) = make_init_samples(pstr, ninit, nsamples, context)

    @debug "Beginning WHAMIS cycles."
    (;approx_dist, samples_p, ess, n_cycles) = whack_many_moles(pstr, (;approx_dist, samples_p); mineff, target_samplesize=Inf, maxcycles=maxcycles, n_parallel=Threads.nthreads(), cache_dir=nothing)
    elapsed_time = time() - start_time
    @debug "Generated final WHAMIS samples in transformed space after $n_cycles cycles, $(@sprintf "%.1f s" elapsed_time)."

    transformed_smpls = samples_p

    @debug "Generated final WHAMIS samples in transformed space after $nsteps, produced $n_samples_indep independent samples after $(@sprintf "%.1f s" elapsed_time)."

    smpls = inverse(f_pretransform).(transformed_smpls)

    elapsed_time = time() - start_time
    @debug "Completed WHAMIS sampling after $(@sprintf "%.1f s" elapsed_time)."

    return (
        result = smpls, result_pretransform = transformed_smpls, f_pretransform = f_pretransform, 
        ess = ess, approx_dist_pretransform = approx_dist, n_cycles = n_cycles
    )
end


function make_init_samples(pstr, ninit, nsamples, context)
    dummy_mvnormal = MvNormal(rand(gen, 1), rand(gen, 1, 1))
    components = fill(dummy_mvnormal, ninit)
    mode_logd_p_approx = rand(gen, ninit)

    @debug "WHAMIS initial mode search and FI-based local posterior approximation for $(ninit) components."

    Threads.@nthreads for i in eachindex(components)
        r_findmap = bat_findmode_impl(pstr, mapalg, context)
        x_map = r_findmap.result
        components[i] = local_metric_approx(pstr, x_map)
        # ToDo: Use existing logdensity computed by findmode if available:
        mode_logd_p_approx[i] = logdensityof(pstr, x_map)
    end

    approx_dist = MixtureModel(components)

    # ToDo: Run in parallel:
    mode_logd_q_approx = [logdensityof(approx_dist, mode(ad)) for ad in approx_dist.components]

    raw_mixture_logw = mode_logd_p_approx .- mode_logd_q_approx
    raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
    mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

    approx_dist = MixtureModel(approx_dist.components, mixture_w)::typeof(approx_dist)

    @debug "Generating initial WHAMIS samples"
    smpls_p = importance_sampling(pstr, approx_dist, nsamples)

    return (approx_dist = approx_dist, samples_p = smpls_p)
end

function local_metric_approx(pstr, θ_sel, adsel)
    m_tr=pstr.likelihood.k
    FI_inner = MGVI.fisher_information(m_tr(θ_sel))
    # ToDo: Don't use fixed choice of `Matrix` here:
    _, J = with_jacobian(MGVI.flat_params ∘ m_tr, θ_sel, Matrix, adsel)
    Σ_raw = inv(Matrix(J' * FI_inner * J + I))
    Σ = PDMat(cholesky(Positive, Σ_raw))
    approx_dist = MvNormal(θ_sel, Σ)
    #approx_dist = MvTDist(1, θ_sel, Σ)
    return approx_dist
end


# ToDo: Use standard BAT functionality for this (resp. make this a standard BAT tool):
function importance_sampling(pstr, approx_dist, nsamples)
    smpls_q, _ = bat_sample(approx_dist, IIDSampling(nsamples = nsamples))
    x_q = smpls_q.v
    logd_q = smpls_q.logd;
    logd_p = similar(logd_q)
    @showprogress Threads.@threads for i in eachindex(x_q)
        logd_p[i] = logdensityof(pstr, x_q[i])
    end
    logw_raw = logd_p .- logd_q;
    w = exp.(logw_raw .- maximum(logw_raw));
    smpls_p = DensitySampleVector(x_q, logd_p, weight=w)
    return smpls_p
end


function whack_many_moles(posterior, init_samples; target_efficiency=Inf, target_samplesize=Inf, maxcycles=100, n_parallel=Threads.nthreads(), cache_dir=nothing)

    pstr, f_trafo = bat_transform(PriorToNormal(), posterior)
    smpls_p = init_samples.samples_p
    
    #μ = mean(smpls_p.v, ProbabilityWeights(smpls_p.weight))
    #Σ = cov(Matrix(flatview(smpls_p.v)'), ProbabilityWeights(smpls_p.weight))
    #approx_dist = MvNormal(μ, Σ)
    approx_dist = init_samples.approx_dist

    if approx_dist isa MixtureModel
        approx_mix = init_samples.approx_dist
        mode_logd_p_mix = [logdensityof(pstr, mode(d)) for d in approx_mix.components]
        #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist)) for approx_dist in approx_mix.components]
    else
        approx_mix = Distributions.MixtureModel([approx_dist], [1])
        mode_logd_p_mix = [logdensityof(pstr, mode(approx_dist))]
        #mode_logd_q_mix = [logdensityof(approx_dist, mode(approx_dist))]
    end
    
    samples_mix = smpls_p
    iter = 0

    if !isnothing(cache_dir)
        if !isdir(cache_dir)
            mkdir(cache_dir)
        end
    end
    
    ess = 0.0

    while true
        ess = oftype(ess, bat_eff_sample_size(samples_mix, KishESS()).result)
        @info "Effective sample size = $ess"
        eff = ess / length(samples_mix)
        @info "Efficiency = $eff"

        if (eff > target_efficiency) | (iter > maxcycles) | (ess > target_samplesize)
            break
        end
        
        idxs = partialsortperm(samples_mix.weight, 1:n_parallel, rev=true)
        
        approx_dists = Array{MvNormal}(undef, n_parallel)
        mode_logd_p_approx = Array{Any}(undef, n_parallel)

        Threads.@threads for i in 1:n_parallel
            θ_iter = samples_mix.v[idxs[i]]
            approx_dists[i] = local_MGVI_approx(pstr, θ_iter)
            mode_logd_p_approx[i] = logdensityof(pstr, mode(approx_dists[i]))
        end
                
        append!(mode_logd_p_mix, mode_logd_p_approx)
        approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, approx_dists))

        mode_logd_q_mix = [logdensityof(approx_mix, mode(ad)) for ad in approx_mix.components]
        
        raw_mixture_logw = mode_logd_p_mix .- mode_logd_q_mix
        raw_mixture_w = exp.(raw_mixture_logw .- maximum(raw_mixture_logw))
        mixture_w = raw_mixture_w ./ sum(raw_mixture_w)

        approx_mix = MixtureModel(approx_mix.components, mixture_w)

        # still need to parallelize
        new_nsamples = [floor(Int, w * length(samples_mix)) for w in last(mixture_w, n_parallel)]

        sampls_ps = Array{Any}(undef, n_parallel)

        for (i, n) in enumerate(new_nsamples)
            if n > 0
                smpls_p = importance_sampling(pstr, approx_dists[i], n)
                samples_mix = vcat(samples_mix, smpls_p)
            end
        end
        
        #approx_mix = Distributions.MixtureModel(vcat(approx_mix.components, [approx_dist]), mixture_w)
        
        logd_p = samples_mix.logd
        logd_q = logdensityof.(Ref(approx_mix), samples_mix.v)
        logw_raw = logd_p .- logd_q;
        w = exp.(logw_raw .- maximum(logw_raw));
        samples_mix.weight .= w;

        iter += 1

        if !isnothing(cache_dir)
            FileIO.save(joinpath(cache_dir, "molewhacker_iter_$(iter).jld2"), Dict("approx_dist"=>approx_mix, "samples_p"=>samples_mix, "samples_user"=>bat_transform(inverse(f_trafo), samples_mix).result))
        end

    end

    return (approx_dist=approx_mix, samples_p=samples_mix, ess = ess, n_cycles = iter)
end
