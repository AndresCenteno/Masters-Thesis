using ForwardDiff, LinearAlgebra, LightGraphs, SparseArrays, BenchmarkTools

function kerexp(X,s=3)
    exp_X = I
    for i=1:s
        exp_X = exp_X + (X^s)/factorial(s)
    end
    return exp_X
end

function D(L,τ)
    return hcat([kerexp(-tau_s*L) for tau_s in τ]...)
end

function Z(X,L,H,τ)
    return norm(X-D(L,τ)*H)^2
end

# need to create signals!
function create_data(N,p=0.3)
    graph = erdos_renyi(N,p)
    L = Matrix(laplacian_matrix(graph))
    X = exp(-L)
    return X, L
end

function create_random(X,S)
    N = size(X,1)
    M = size(X,2)
    τ = 1:S
    H = Matrix(sprand(N*S,M,0.1))
    L = Matrix(laplacian_matrix(erdos_renyi(N,0.3)))
    return L, H, τ
end

Nvec = map(x -> 2^x,4)
res = zeros(length(Nvec))

N = 200
X, L_ground = create_data(N)
L0, H0, τ0 = create_random(X,1)
b = @benchmark ForwardDiff.gradient(L->Z(X,L,H0,τ0),L_ground)

for i=1:length(Nvec)
    N = Nvec[i]
    X, L_ground = create_data(N)
    L0, H0, τ0 = create_random(X,1)
    b = @benchmark ForwardDiff.gradient(L->Z(X,L,H0,τ0),L_ground)
    res[i] = mean(b).time
end

res

using Plots

using JLD2
save_object("time_autodiff_3_11.jld2",res)
