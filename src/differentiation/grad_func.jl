# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct GradFunc{
    OF<:Function,
    UF<:Function,
    VS<:AbstractValueShape,
    GS<:AbstractValueShape,
    Alg<:DifferentiationAlgorithm
} <: GradientFunction
    _orig_f::OF
    _unshaped_f::UF
    _input_shape::VS
    _grad_shape::GS
    _diffalg::Alg
end


function GradFunc(f::Function, diffalg::DifferentiationAlgorithm)
    input_shape = varshape(f)
    unshaped_f = unshaped(f)
    grad_shape = gradient_shape(input_shape)
    GradFunc(f, unshaped_f, input_shape, grad_shape, diffalg)
end


function Base.show(io::IO, gf::GradFunc)
    print(io, Base.typename(typeof(gf)).name, "(")
    show(io, gf._orig_f)
    print(io, ", ")
    show(io, gf._diffalg)
    print(io, ")")
end

function Base.show(io::IO, M::MIME"text/plain", gf::GradFunc)
    print(io, Base.typename(typeof(gf)).name, "(")
    show(io, M, gf._orig_f)
    print(io, ", ")
    show(io, M, gf._diffalg)
    print(io, ")")
end


function (gf::GradFunc)(v::Any)
    input_shape = gf._input_shape
    v_unshaped = unshaped(v, input_shape)

    grad_f_unshaped = similar(v_unshaped)

    value = unshaped_gradient!(grad_f_unshaped, gf._unshaped_f, v_unshaped, gf._diffalg)

    (value, gf._grad_shape(grad_f_unshaped))
end


function (gf::GradFunc)(::typeof(!), grad_f::Any, v::Any)
    input_shape = gf._input_shape
    v_unshaped = unshaped(v, input_shape)

    if isnothing(grad_f)
        R(gf._unshaped_f(v_unshaped))
    else
        grad_f_unshaped = unshaped(grad_f, gf._grad_shape)
        unshaped_gradient!(grad_f_unshaped, gf._unshaped_f, v_unshaped, gf._diffalg)
    end
end


function valgradof(f::Function, algorithm::DifferentiationAlgorithm = some_vjp_algorithm(f))
    GradFunc(f, algorithm)
end
