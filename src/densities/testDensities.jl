struct Rastrigin{B<:Vector{Float64}, P<:Real} <: AbstractDensity
    bounds::B
    a::P
    b::P
    dist
end


function Rastrigin(;bounds=[-3.,3.],a=30,b=10)
    integral_val = -a*bounds[1] + b*bounds[1] + (bounds[1]^3)/3 + a*bounds[2] - b*bounds[2] - (bounds[2]^3)/3 + (b*(-sin(2*bounds[1]*pi) + sin(2*bounds[2]*pi)))/(2*pi)
    dist = x-> (1/integral_val) * (a - b - x^2 + b*cos(2*pi*x))
    Rastrigin(bounds,a,b,dist)
end

BAT.eval_logval_unchecked(density::Rastrigin,v::Any) = log(broadcast(density.dist,v))

"""
    Rastrigin <: AbstractDensity
"""

struct SineSquared{B<:Vector{Float64}} <: AbstractDensity
    bounds::B
    dist
end

function SineSquared(;bounds=[-0.,25.])
    integral_val = 1/40*(-4*(bounds[1])^5 + 10*(2*(bounds[1])^2 - 3)*(bounds[1])*cos(2*(bounds[1])) + 5*(2*(bounds[1])^4 - 6*(bounds[1])^2 + 3)*sin(2*(bounds[1])) + 4*(bounds[2])^5 + 10*(bounds[2])*(3 - 2*(bounds[2])^2)*cos(2*(bounds[2])) - 5*(2*(bounds[2])^4 - 6*(bounds[2])^2 + 3)*sin(2*(bounds[2])))
    dist = x-> (1/integral_val) * (x^4*sin(x)^2)
    SineSquared(bounds,dist)
end

BAT.eval_logval_unchecked(density::SineSquared,v::Any) = log(broadcast(density.dist,v))

"""
    SineSquared <: AbstractDensity
"""

struct HoelderTable{B<:Vector{Float64}} <: AbstractDensity
    bounds::B
    dist
end
export HoelderTable

function HoelderTable()
    bounds = [-10.,10.]
    integral_val = (-18*exp(1)^2*π^2 - 12*exp(1)^3*π^2 + exp(1)*(500 + 488*π^2) - 6*exp(1)^(10/π)*π*(π*cos(10) - sin(10)))/(3*exp(1)*(1 + π^2))
    dist = x-> (1/integral_val) * (10-((x^2)/20) - abs( sin(x)*exp(abs(1-((sqrt(x^2))/(pi))))) )
    HoelderTable(bounds,dist)
end

BAT.eval_logval_unchecked(density::HoelderTable,v::Any) = log(broadcast(density.dist,v))


"""
    HoelderTable <: AbstractDensity
    Fixed bounds, integral not easily calculable
"""
