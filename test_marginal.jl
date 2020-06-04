# # BAT.jl plotting tutorial

using BAT
using Distributions
using IntervalSets


likelihood = params -> begin

    r1 = logpdf.(
    MixtureModel(Normal[
    Normal(-10.0, 1.2),
    Normal(0.0, 1.8),
    Normal(10.0, 2.5)], [0.1, 0.3, 0.6]), params.a)

    r2 = logpdf.(
    MixtureModel(Normal[
    Normal(-5.0, 2.2),
    Normal(5.0, 1.5)], [0.3, 0.7]), params.b[1])

    r3 = logpdf.(Normal(2.0, 1.5), params.b[2])

    return LogDVal(r1+r2+r3)
end

prior = BAT.NamedTupleDist(
    a = Normal(-3, 4.5),
    b = [-20.0..20.0, -10..10]
)

posterior = PosteriorDensity(likelihood, prior);

samples, chains = bat_sample(posterior, (10^5, 4), MetropolisHastings());

unshaped_samples = BAT.unshaped.(samples)

BAT.bat_marginalize(samples, (:(b[1]),:(b[2])))


BAT.bat_marginalize(unshaped_samples, (1, 2))





# ## Set up plotting
# Set up plotting using the [Plots.jl](https://github.com/JuliaPlots/Plots.jl) package:
using Plots


plot(samples, :a) #default seriestype = :smallest_intervals (alias :HDR)

plot(prior, :a)
#or: plot(prior, 1)

# ## Knowledge update plot
# The knowledge update after performing the sampling can be visualized by plotting the prior and the samples of the psterior together in one plot using `plot!()`:
plot(samples, :(b[1]))
plot!(prior, :(b[1]))
0.7^15*100


lost = 0
win = 0
x=0
for i in 1:20
    while(win<=lost)
        global x +=0.5
        res = 18*x

        win = res-lost
    end
    lost = lost + 6*x
    println(x)
end

function f(x, lost)
    j=0
    for i in 1:30
        x += 0.5
        res = 18*x

        #println("lost before: $lost")
        win = res
        plus = win-6*x-lost

        if(plus > 0)
            j = j+1
            println("\nj: $j")
            println("x: $x")
            println("6*x: $(6*x)")
            println("win: $win")
            println("plus: $plus")
            println("total: $total")
            total = total + 6*x
            lost = lost + 6*x
            # println("i: $i")
            #println("lost after: $lost\n")
        end


    end
end

f(0, 0)

p = 1-12/37
p^15*100

0.25^10*100
