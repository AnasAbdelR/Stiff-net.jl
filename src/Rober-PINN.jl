using OrdinaryDiffEq, Flux, Plots
using DifferentialEquations
using ForwardDiff
using LinearAlgebra
using BSON: @save
using BSON: @load
LinearAlgebra.BLAS.set_num_threads(16)


pz = [4.0e1, 3.0e2, 1.0e2]
u0 = [1.0,0.0,0.0]
tspan = (0,1.0)

function rober(du,u,p,t)
    y₁,y₂,y₃ = u
    k₁,k₂,k₃ = p
    du[1] = -k₁*y₁+k₃*y₂*y₃
    du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
    du[3] =  k₂*y₂^2
    nothing
end

solver = Rosenbrock23(autodiff=false,diff_type=Val{:forward})
prob = ODEProblem(rober, u0, tspan, pz)
sol = solve(prob)
sol.t

#Create the NN
dudt2 = Chain(Dense(1,8, gelu),
             Dense(8, 8, gelu),
             Dense(8, 8, gelu),
             Dense(8, 3))


p,re = Flux.destructure(dudt2) # use this p as the initial condition!
dudt(u,p) = re(p)(u)

function grab_∇u(p)
    reduce(hcat,[ForwardDiff.jacobian(u-> dudt(u,p), [t]) for t in sol.t])
end
function grab_pred(p)
    reduce(hcat, [dudt([t],p) for t in sol.t])
end
#test grab_∇u
grab_∇u(p)
grab_pred(p)

function loss_PINNS(p)
   #loss for stiff-PINNS
   uₛ = grab_pred(p)
   u₁ = uₛ[1, :]
   u₂ = uₛ[2, :]
   u₃ = uₛ[3, :]
   k₁,k₂,k₃ = pz
   duₛ = grab_∇u(p)
   du₁ = duₛ[1, :]
   du₂ = duₛ[2, :]
   du₃ = duₛ[3, :]
   e1 = abs.(du₁ .- k₁*u₁ .+ k₃.*u₂.*u₃)
   e2 = abs.(du₂ .- k₁*u₁ .- k₂*u₂.^2 .- k₃.*u₂.*u₃)
   e3 = abs.(du₃ .- k₂*u₂.^2)
   #loss_phys = sum(abs, e1 + e2 + e3)
   loss_ic = sum(abs, uₛ[:, 1] - u0)
   #total_loss= loss_phys
   total_loss = loss_ic
   return total_loss
end
#test loss_PINNS
loss_PINNS(p*1e-4)

function grab_∇L(par)
 #Grabs gradients of NN loss w.r.t params
 ForwardDiff.gradient(loss_PINNS, par)
end
#test grab_∇L
@time grab_∇L(p)

function dθ_dt!(du,u,k,t)
 #ODE used to evolve parmaeters based on the negative gradient definition (evolving in the direction of steepest descent)
 du .= -1 .* grab_∇L(u)
end
#test dθ_dt!
dθ_dt!(zeros(length(p)), p, 0.2, 0.2)


tspan_θ = (0.0,1.0)
prob_θ = ODEProblem(dθ_dt!,p*1e-4,tspan_θ)
integrator = DifferentialEquations.init(prob_θ,Tsit5())
loss_arr = []
params_arr = []

for (u,t) in tuples(integrator)
    loss = loss_PINNS(u)
    push!(loss_arr, loss)
    push!(params_arr, u)
    println(loss)
end

plot(log.(loss_arr))

#savefig("Loss_PINN-ROBER.png")

test_p = params_arr[end]
grab_pred(test_p)
plot(sol.t, grab_pred(test_p)')
plot!(sol.t, Array(sol)')
