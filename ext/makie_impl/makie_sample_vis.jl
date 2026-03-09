
function bat_sample_and_visualize(target, algorithm::TransformedMCMC, context::BATContext)
        measure = convert_for(bat_sample, target)
        orig_context = deepcopy(context)
        r = bat_sample_vis_impl(measure, algorithm, context)
        result_with_args(Val(:samples), target, r, (algorithm=algorithm, context=orig_context))
end


function bat_sample_vis_impl(m::BATMeasure, samplingalg::TransformedMCMC, context::BATContext)
        transformed_m, f_pretransform = transform_and_unshape(samplingalg.pretransform, m, context)

        mcmc_states, chain_outputs = mcmc_init!(
                samplingalg,
                transformed_m,
                apply_trafo_to_init(f_pretransform, samplingalg.init),
                samplingalg.store_burnin ? samplingalg.callback : nop_func,
                context
        )

        for i in eachindex(mcmc_states)
                get_samples!(chain_outputs[i], mcmc_states[i], false)
        end
        chain_outputs_obs = Observable(chain_outputs)
        samples_obs = lift(chain_outputs_obs) do c_out
                transform_samples(inverse(f_pretransform), _merge_chain_outputs(first(mcmc_states), c_out))
        end

        if !samplingalg.store_burnin
                chain_outputs = _empty_chain_outputs.(mcmc_states)
                for i in eachindex(mcmc_states)
                        get_samples!(chain_outputs[i], mcmc_states[i], false)
                end
                chain_outputs_obs[] = chain_outputs
        end

        # is_sampling = Threads.Atomic{Bool}(true)
        #
        # @async begin
        #         while is_sampling[]
        #                 sleep(0.01)
        #                 notify(chain_outputs_obs)
        #                 println("notified makie")
        #         end
        #         notify(chain_outputs_obs)
        #         println("final notification")
        # end

        # last_yield_time = time()
        # vis_callback = (args...) -> begin
        #         samplingalg.callback(args...)
        #         if time() - last_yield_time > 0.01
        #                 yield()
        #                 last_yield_time = time()
        #         end
        # end

        mcmc_states = mcmc_burnin!(
                samplingalg.store_burnin ? chain_outputs : nothing,
                mcmc_states,
                samplingalg,
                samplingalg.store_burnin ? samplingalg.callback : nop_func
        )

        next_cycle!.(mcmc_states)

        @info "Generate main samples using $(length(mcmc_states)) MCMC chain(s)."

        fig = bat_makie_plot(samples_obs)
        display(fig)

        steps_frame = 1
        nsteps = 0

        while nsteps < samplingalg.nsteps
                mcmc_states = mcmc_iterate!!(
                        chain_outputs,
                        mcmc_states;
                        max_nsteps=steps_frame,
                        nonzero_weights=samplingalg.nonzero_weights,
                        callback=samplingalg.callback
                )

                println("stepped, notifying observables")
                nsteps += steps_frame
                notify(chain_outputs_obs)
                sleep(2.5)
                println("slept 2.5")
        end

        @debug "Merge samples of chains and transform to original space."

        samples_transformed = _merge_chain_outputs(first(mcmc_states), chain_outputs)

        smpls = transform_samples(inverse(f_pretransform), samples_transformed)

        (result=smpls, result_trafo=samples_transformed, f_pretransform=f_pretransform, generator=MCMCSampleGenerator(mcmc_states))
end

