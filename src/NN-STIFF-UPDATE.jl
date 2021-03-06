using DiffEqFlux, OrdinaryDiffEq, Flux, Plots
using Zygote
using DifferentialEquations
using ForwardDiff
using LinearAlgebra
using BSON: @save
using BSON: @load
LinearAlgebra.BLAS.set_num_threads(16)



function rober(du,u,p,t)
    y₁,y₂,y₃ = u
    k₁,k₂,k₃ = p
    du[1] = -k₁*y₁+k₃*y₂*y₃
    du[2] =  k₁*y₁-k₂*y₂^2-k₃*y₂*y₃
    du[3] =  k₂*y₂^2
    nothing
end

function generate_data(u0, pz, tspan, solver)
    ode_prob = ODEProblem(rober,u0,tspan,pz)
    ode_sol = solve(ode_prob, solver)
    data = Array(ode_sol)
    return data, ode_sol.t
end

function predict_n_ode(k)
    Array(solve(prob, Rosenbrock23(), u0=u0,p=k, saveat=tsteps))
end

function loss_n_ode(k)
    #loss for stiff-net
      pred = predict_n_ode(k)
      loss = sum(abs2,ode_data .- pred)
      return loss
end

function grab_∇(par)
    #Grabs gradients of NN loss w.r.t params
    ForwardDiff.gradient(loss_n_ode, par)
end

function dθ_dt!(du,u,k,t)
    #ODE used to evolve parmaeters based on the negative gradient definition (evolving in the direction of steepest descent)
    du .= -1 .* grab_∇(u)
end

#################################################################################################################################################################################
#Generate the data
pz = [0.04,3e7,1e4]
u0 = [1.0,0.0,0.0]
tspan = (1.0e-5,1.0e5)
solver = Rosenbrock23(autodiff=false,diff_type=Val{:forward})
ode_data, tsteps = generate_data(u0, pz, tspan, solver)
plot(tsteps, ode_data')


#Create the NN
dudt2 = Chain(Dense(3,3, tanh),
             Dense(3, 3, tanh),
             Dense(3, 3, tanh),
             Dense(3,3))

@load "stiff-net.bson" dudt2

p,re = Flux.destructure(dudt2) # use this p as the initial condition!
dudt(u,p,t) = re(p)(u) # need to restrcture for backprop!
prob = ODEProblem(dudt,u0,tspan)
#Test predicion functions
predict_n_ode(p)
#Test loss functions, check for good initialisation (loss must be <100)
loss_n_ode(p)
#Save well behaved initialisation
#@save "stiff-net.bson" dudt2

#Train with solver
tspan_θ = (0.0,0.125)
prob_θ = ODEProblem(dθ_dt!,p,tspan_θ)
integrator = init(prob_θ,Rosenbrock23(autodiff=false))
loss_arr = []
params_arr = []
for (u,t) in tuples(integrator)
    loss = loss_n_ode(u)
    push!(loss_arr, loss)
    push!(params_arr, u)
    println(loss)
end

plot(loss_arr)




#Plotting first concentration
plotly()
plot(tsteps, predict_n_ode(param_his[:, end])[1, :])
scatter!(tsteps, ode_data[1, :], title = "First concentration", label = ["NN-pred" "Ground truth"])

#Plotting second concentration
plot(tsteps, predict_n_ode(param_his[:, end])[2, :])
scatter!(tsteps, ode_data[2, :], title = "Second concentration", label = ["NN-pred" "Ground truth"])

#Plotting third concentration
plot(tsteps, predict_n_ode(param_his[:, end])[3, :])
scatter!(tsteps, ode_data[3, :], title = "Third concentration", label = ["NN-pred" "Ground truth"])


#Plotting losss
loss = [loss_n_ode(param_his[:,i]) for i in 1:size(param_his)[2]]
plot(loss, title = "Loss")
