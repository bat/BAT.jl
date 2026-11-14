using CUDA, MLDataDevices, Adapt, LinearAlgebra
import DensityInterface: DensityKind, IsDensity

struct _GPUShiftLikelihood{V}
    shift::V
end
DensityKind(::_GPUShiftLikelihood) = IsDensity()
Adapt.@adapt_structure _GPUShiftLikelihood
function logdensityof(d::_GPUShiftLikelihood, x)
    value = zero(eltype(x))
    for i in eachindex(x)
        value -= abs2(x[i] - d.shift[i]) / 2
    end
    return value
end

if CUDA.functional()
    CUDA.allowscalar(false)
    @testset "BAT CUDA sampling and coordinate restoration" begin
        device = MLDataDevices.with_eltype(MLDataDevices.CUDADevice(), nothing)
        unit = BAT.HeterogeneousComputing.DeviceUnit(device)
        context(seed) = BATContext(cunit = unit, rng = Xoshiro(seed))
        bank = IS.ProposalBank([
            IS.SphericalGaussian(fill(x, 2), 1.0f0) for x in (-0.5f0, 0.5f0)
        ])
        prior =
            NamedTupleDist(rate = LogNormal(0.0f0, 0.5f0), fraction = Uniform(0.0f0, 1.0f0))
        target = PosteriorMeasure(logfuncdensity(x -> -x.rate - x.fraction), prior)
        for sampler in (
            IS.ImportanceSampling(
                IS.SphericalGaussian(zeros(Float32, 2), 1.0f0);
                nsamples = 256,
            ),
            IS.ImportanceSampling(bank; nsamples = 256),
            IS.AMIS(
                IS.FactorGaussian(zeros(Float32, 2), Matrix{Float32}(I, 2, 2));
                rounds = 3,
                round_size = [64, 96, 128],
            ),
            IS.DeterministicMixturePMC(bank; rounds = 2, round_size = 128),
        )
            result =
                evalmeasure(target, ImportanceSamplersSampling(; sampler), context(853))
            gpu = samplesof(result)
            host = MLDataDevices.cpu_device()(gpu)
            raw = MLDataDevices.cpu_device()(evalinfo(result).result.samples)
            @test host.logd ≈ logdensityof.(Ref(target), host.v) rtol = 16eps(Float32)
            @test host.weight ≈ IS.normalized_weights(raw) rtol = 8eps(Float32)
            @test all(
                x -> x isa CUDA.AnyCuArray,
                (gpu.v.rate, gpu.v.fraction, gpu.logd, gpu.weight),
            )
        end

        likelihood = _GPUShiftLikelihood(Float32[0.3, -0.2])
        gaussian = MvNormal(Float32[0.2, -0.4], Float32[1.0 0.3; 0.3 2.0])
        sampler = IS.ImportanceSampling(
            IS.SphericalGaussian(zeros(Float32, 2), 1.0f0);
            nsamples = 64,
        )
        for target in (gaussian, PosteriorMeasure(likelihood, gaussian))
            result =
                evalmeasure(target, ImportanceSamplersSampling(; sampler), context(854))
            host = MLDataDevices.cpu_device()(samplesof(result))
            @test host.logd ≈ logdensityof.(Ref(target), host.v) rtol = 16eps(Float32)
            @test samplesof(result).weight isa CUDA.AnyCuArray
        end
        if Base.find_package("Enzyme") !== nothing
            import Enzyme
            bank = IS.ProposalBank([
                IS.FactorGaussian(fill(x, 2), Matrix{Float32}(I, 2, 2)) for
                x in (-0.2f0, 0.2f0)
            ])
            gradient_sampler = IS.FirstOrderGRAMIS(
                bank;
                rounds = 2,
                round_size = 32,
                repulsion_strength = 0.0f0,
            )
            for (pretransform, gradient_likelihood) in (
                (DoNotTransform(), likelihood),
                (NormalBased(), _GPUShiftLikelihood((0.3f0, -0.2f0))),
            )
                target = PosteriorMeasure(gradient_likelihood, gaussian)
                result = evalmeasure(
                    target,
                    ImportanceSamplersSampling(; sampler = gradient_sampler, pretransform),
                    BATContext(cunit = unit, rng = Xoshiro(856), ad = Enzyme),
                )
                host = MLDataDevices.cpu_device()(samplesof(result))
                @test host.logd ≈ logdensityof.(Ref(target), host.v) rtol = 32eps(Float32)
                @test samplesof(result).weight isa CUDA.AnyCuArray
            end
        end
        @test likelihood.shift == Float32[0.3, -0.2]

        if length(CUDA.devices()) > 1
            previous = CUDA.device()
            other = first(filter(!=(previous), collect(CUDA.devices())))
            other_device =
                MLDataDevices.with_eltype(MLDataDevices.CUDADevice(other), nothing)
            other_context = BATContext(
                cunit = BAT.HeterogeneousComputing.DeviceUnit(other_device),
                rng = Xoshiro(855),
            )
            target = PosteriorMeasure(logfuncdensity(x -> -x.rate - x.fraction), prior)
            result =
                evalmeasure(target, ImportanceSamplersSampling(; sampler), other_context)
            gpu = samplesof(result)
            @test CUDA.device(gpu.v.rate) == CUDA.device(gpu.weight) == other
            @test CUDA.device() == previous
        end
    end
end
