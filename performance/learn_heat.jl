using LinearAlgebra

function lipschitz_c1(L,τ)
   return norm(2*transpose(D(L,τ))*D(L,τ))
end


