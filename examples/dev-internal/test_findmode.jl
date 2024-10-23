using BAT
using Optim

posterior = BAT.example_posterior()

optalg = OptimAlg(;
    optalg = Optim.NelderMead(parameters=Optim.FixedParameters()),
    maxiters=200,
    kwargs = (f_calls_limit=100,),
)

my_mode = bat_findmode(posterior, optalg)

fieldnames(typeof(my_mode.info.res))


using BAT
#using Optim
#using Optimization
using OptimizationOptimJL

using InverseFunctions, FunctionChains, DensityInterface


posterior = BAT.example_posterior()
optalg = OptimizationAlg(; optalg = OptimizationOptimJL.ParticleSwarm(n_particles=10), maxiters=200, kwargs=(f_calls_limit=500,))
my_result = bat_findmode(posterior, optalg)

a = my_result.info

@test a.cache.solver_args.maxiters == 500

dump(a.alg)

fieldnames(typeof(a.cache.solver_args))

fieldnames(typeof(a.original.method))

my_mode.info.original




# Define a NamedTuple with keyword arguments
nt = (a=1, b=2)

# Define a function that accepts keyword arguments
function my_function(; a=0, b=0, c=0)
    println("a = $a")
    println("b = $b")
    println("c = $c")
end

# Call the function and unpack the NamedTuple
my_function(; nt...) 




context = get_batcontext()
target = posterior
transformed_density, f_transform = BAT.transform_and_unshape(PriorToNormal(), target, context)
inv_trafo = inverse(f_transform)
initalg = BAT.apply_trafo_to_init(f_transform, InitFromTarget())
x_init = collect(bat_initval(transformed_density, initalg, context).result)

f = fchain(inv_trafo, logdensityof(target), -)
f2 = (x, p) -> f(x)


optimization_function = Optimization.OptimizationFunction(f2, Optimization.SciMLBase.NoAD())
optimization_problem = Optimization.OptimizationProblem(optimization_function, x_init)
optimization_result = Optimization.solve(optimization_problem,OptimizationOptimJL.NelderMead())


optalg = OptimizationAlg(;optalg = OptimizationOptimJL.NelderMead())
my_mode = bat_findmode(posterior, optalg)

my_mode.info.original
fieldnames(typeof(my_mode.info))

rosenbrock(x, p) = (p[1] - x[1])^2 + p[2] * (x[2] - x[1]^2)^2
f = rosenbrock


b = Optimization.SciMLBase.NoAD()
supertype(typeof(b))

adm = ADSelector(ForwardDiff)

adsel = BAT.get_adselector(context)
supertype(typeof(adsel))



adm2 = convert_ad(ADTypes.AbstractADType, adm)
ADTypes.AutoForwardDiff()

optimization_function = Optimization.OptimizationFunction(f2, adm2)
