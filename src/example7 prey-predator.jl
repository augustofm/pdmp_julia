using Stan

using StanSample

functions {
  real[] dz_dt(real t,       // time
               real[] z,     // system state {prey, predator}
               real[] theta, // parameters
               real[] x_r,   // unused data
               int[] x_i) {
    real u = z[1];
    real v = z[2];

    real alpha = theta[1];
    real beta = theta[2];
    real gamma = theta[3];
    real delta = theta[4];

    real du_dt = (alpha - beta * v) * u;
    real dv_dt = (-gamma + delta * u) * v;
    return { du_dt, dv_dt };
  }
}
data {
  int<lower = 0> N;           // number of measurement times
  real ts[N];                 // measurement times > 0
  real y_init[2];             // initial measured populations
  real<lower = 0> y[N, 2];    // measured populations
}
parameters {
  real<lower = 0> theta[4];   // { alpha, beta, gamma, delta }
  real<lower = 0> z_init[2];  // initial population
  real<lower = 0> sigma[2];   // measurement errors
}
transformed parameters {
  real z[N, 2]
    = integrate_ode_rk45(dz_dt, z_init, 0, ts, theta,
                         rep_array(0.0, 0), rep_array(0, 0),
                         1e-5, 1e-3, 5e2);
}
model {
  theta[{1, 3}] ~ normal(1, 0.5);
  theta[{2, 4}] ~ normal(0.05, 0.05);
  sigma ~ lognormal(-1, 1);
  z_init ~ lognormal(log(10), 1);
  for (k in 1:2) {
    y_init[k] ~ lognormal(log(z_init[k]), sigma[k]);
    y[ , k] ~ lognormal(log(z[, k]), sigma[k]);
  }
}
generated quantities {
  real y_init_rep[2];
  real y_rep[N, 2];
  for (k in 1:2) {
    y_init_rep[k] = lognormal_rng(log(z_init[k]), sigma[k]);
    for (n in 1:N)
      y_rep[n, k] = lognormal_rng(log(z[n, k]), sigma[k]);
  }
}


using DifferentialEquations

f(u,p,t) = 0.98u
u0 = 1.0
tspan = (0.0,1.0)
prob = ODEProblem(f,u0,tspan)

sol = solve(prob)
sol

global p
z(u,p,t) = [(p[1]-p[2]*u[2])*u[1], (-p[3]+p[4]*u[1])*u[2]]

p=[1,0.05,1,0.05]
u0 = [35.,5]
tspan = (0.0,56.0)
prob = ODEProblem(z,u0,tspan,p)

sol = solve(prob)

using Plots; gr()
plot(sol)
disc=sol(1:56)
plot(disc)

using CSV
df = CSV.read("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/examples/lotka-volterra.csv")


using DifferentialEquations
using DiffEqSensitivity, OrdinaryDiffEq, Zygote
using LinearAlgebra         #For Cholesky
import Random               #For generating random vectors               #For plotting
using DelimitedFiles
using Distributions


using Flux, DiffEqFlux

using DifferentialEquations
using DiffEqSensitivity
#using OrdinaryDiffEq
using Zygote

#using LinearAlgebra, SparseArrays, StaticArrays, ArrayInterface, Requires
Pkg.add("Zygote")
Pkg.add("DifferentialEquations")
Pkg.rm("DiffEqSensitivity")
Pkg.rm("Zygote")

Pkg.add("DiffEqSensitivity")
Pkg.add("FFTW")
Pkg.add(Pkg.PackageSpec(;name="DiffEqSensitivity", version="6.0.0"))
Pkg.rm("BandedMatrices")
Pkg.add(Pkg.PackageSpec(;name="ArrayLayouts", version="0.2.0"))
Pkg.add("BandedMatrices")
using BandedMatrices
using ArrayLayouts
#Pkg.add(Pkg.PackageSpec(;name="BandedMatrices", version="0.3.3"))

using ArrayLayouts
using BandedMatrices

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [1.0,1.0]
tspan = (0.0,56.0)
p = [1.,.05,1.,.05]
prob = ODEProblem(lotka_volterra,u0,tspan,p)

function sum_of_solution(u0,p)
  _prob = remake(prob,u0=u0,p=p)
  sum(solve(prob,Tsit5(),saveat=0.1,sensealg=QuadratureAdjoint()))
end
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)


using DiffEqBase, ForwardDiff, Tracker, DiffEqDiffTools, Statistics
using DiffEqCallbacks, QuadGK, RecursiveArrayTools, LinearAlgebra
using DataFrames, GLM, RecursiveArrayTools

using Pkg
Pkg.resolve()
Pkg.rm("")
Pkg.add("StaticArrays")

concrete_solve(prob,Tsit5(),u0,p,saveat=0.1)



sol = solve(prob)
using Plots; gr()
plot(sol)

concrete_solve(prob,Tsit5(),u0,p,saveat=0.1)


params = Flux.params(p)
function predict_rd_dde()
  concrete_solve(prob,MethodOfSteps(Tsit5()),u0,p,sensealg=TrackerAdjoint(),saveat=0.1)[1,:]
end
loss_rd_dde() = sum(abs2,x-1 for x in predict_rd_dde())
loss_rd_dde()

function sum_of_solution(u0,p)
  _prob = remake(prob,u0=u0,p=p)
  sum(solve(prob,Tsit5(),saveat=0.1,sensealg=BacksolveAdjoint()))
end
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)


function fiip(du,u,p,t)
  du[1] = dx = p[1]*u[1] - p[2]*u[1]*u[2]
  du[2] = dy = -p[3]*u[2] + p[4]*u[1]*u[2]
end
p = [1.5,1.0,3.0,1.0]; u0 = [1.0;1.0]
prob = ODEProblem(fiip,u0,(0.0,10.0),p)
sol = solve(prob,Tsit5())


function sum_of_solution(u0,p)
  _prob = remake(prob,u0=u0,p=p)
  sum(solve(prob,Tsit5(),saveat=0.1,sensealg=QuadratureAdjoint()))
end
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)

using DiffEqSensitivity

]add DiffEqSensitivity

function delay_lotka_volterra(du,u,h,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)*h(p,t-0.1)[1]
  du[2] = dy = (δ*x - γ)*y
end
h(p,t) = ones(eltype(p),2)
prob = DDEProblem(delay_lotka_volterra,[1.0,1.0],h,(0.0,10.0),constant_lags=[0.1])

p = [2.2, 1.0, 2.0, 0.4]
params2 = Flux.params(p)
function predict_rd_dde()
  concrete_solve(prob,MethodOfSteps(Tsit5()),u0,p,sensealg=TrackerAdjoint(),saveat=0.1)[1,:]
end
loss_rd_dde() = sum(abs2,x-1 for x in predict_rd_dde())
loss_rd_dde()



using Pkg
Pkg.resolve()
Pkg.clone("https://github.com/SciML/DiffEqSensitivity.jl.git")
Pkg.clone("https://github.com/mrxiaohe/RobustStats.jl.git")
Pkg.resolve()
Pkg.rm("DiffEqSensitivity"; mode = PKGMODE_MANIFEST)
Pkg.GitTools.clone("https://github.com/SciML/DiffEqSensitivity.jl", "~Documents/dev/julia_dev")
cd("~Documents/dev/julia_dev")
Pkg.activate(".")
Pkg.instantiate()
Pkg.add("Distributions")
Pkg.add("DiffEqSensitivity")

using DifferentialEquations
using DiffEqSensitivity

function f(du,u,p,t)
  dx = p[1]*u[1] - p[2]*u[1]*u[2]
  dy = -p[3]*u[2] + u[1]*u[2]
end

p = [1.5,1.0,3.0]
prob = ODELocalSensitivityProblem(f,[1.0;1.0],(0.0,10.0),p)


using BandedMatrices
using ArrayLayouts
using Plots

Pkg.add("BoundaryValueDiffEq")
Pkg.add("BandedMatrices")
Pkg.add("ArrayLayouts")

Pkg.add("DifferentialEquations")
Pkg.clone("https://github.com/JuliaMatrices/BandedMatrices.jl.git")
using BandedMatrices

using DifferentialEquations
using DiffEqSensitivity
using Zygote

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -δ*y + γ*x*y
end
u0 = [100.0,24.0]

tspan = (0.0,56.0)

p = [1.0,1.0,0.05,0.05]

prob = ODEProblem(lotka_volterra,u0,tspan,p)
#prob = ODEForwardSensitivityProblem(lotka_volterra,u0,tspan,p)
#sol = solve(prob)

function test_f(p)
  _prob = remake(prob;u0=convert.(eltype(p),prob.u0),p=p)
  solve(_prob,Tsit5(),save_everystep=false)[end]
#  sol.(1:56)[10]
end
#using ForwardDiff
res = ForwardDiff.jacobian(test_f,p)

#Pkg.add("DiffResults")
using DiffResults
#res = DiffResults.JacobianResult(u0,p) # Build the results object
DiffResults.jacobian!(res,p) # Populate it with the results
val = DiffResults.value(res) # This is the sol[end]
jac = DiffResults.jacobian(res) # This is dsol/dp

sol = solve(prob)
plot(sol)

prob = ODELocalSensitivityProblem(lotka_volterra,u0,tspan,p)
#sol = solve(prob,DP8())
prob = ODELocalSensitivityProblem(lotka_volterra,u0,tspan,p)
sol = solve(prob,Tsit5(),saveat=0.1,sensealg=QuadratureAdjoint())

sol_aux = sol.(1:56)
x,dp = extract_local_sensitivities(sol)
plot(sol.t,x[1,:])
  plot!(sol.t,x[2,:])

x,dp = extract_local_sensitivities(sol)

z,dz_dp = extract_local_sensitivities(sol,1.)
df_dz = (log.(y)-log.(z)).*z.^(-1)
reshape(dz_dp, (4,1))
dz_dp=convert(Array,VectorOfArray(dz_dp))'

# AGORA VAI
using CSV
#using Pkg
Pkg.add("DiffEqBase")
#using Compat
using DifferentialEquations
using DiffEqSensitivity
using DiffResults
using Plots
using Zygote
using DiffEqObjective

using CSV
using ForwardDiff, DiffEqBase, ForwardDiff, Calculus, Distributions, LinearAlgebra, DiffEqSensitivity
using DifferentialEquations, DiffResults

DiffEqBase, ForwardDiff, Calculus, Distributions, LinearAlgebra, DiffEqSensitivity

Pkg.update("DiffEqSensitivity")
Pkg.status("DiffEqSensitivity")

Pkg.rm("DiffEqSensitivity"; mode = PKGMODE_MANIFEST)
Pkg.add("DiffEqSensitivity")


df = CSV.read("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/examples/lotka-volterra2.csv")

tsvec = Array(df[:year])
hare = df[:,3]
lynx = df[:,2]#Array(df[:lynx])
y=hcat(hare,lynx)
n=length(tsvec)

#u0=y[1,:]
u0=[33.9,5.93]
sigma1=0.25
sigma2=0.252

tspan = (0.0,20.0)
p = [0.545,.028,.803,0.024]
#u0 = [33.,5.]
#sigma=0.25

function extract!(zvec::AbstractMatrix{T}, der_total::AbstractArray{T}, i::Int) where T
     zvec[i,:] = extract_local_sensitivities(sol,Float32(i-1))[1]
     der_total[:,:,i]=convert(Array,VectorOfArray(extract_local_sensitivities(sol,Float32(i-1))[2]))'
     nothing
end


function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, γ, δ = p
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -γ*y + δ*x*y
end


function gradll2(param::Vector{<:Real})
    prob = ODEForwardSensitivityProblem(lotka_volterra,u0,tspan,param)
    sol = solve(prob,Tsit5(),saveat=1,sensealg=QuadratureAdjoint(),tstops = [7,12,18],alg_hints = [:stiff],abstol=1/10^14,reltol=1/10^14)
    global sol
  #  prob = ODEProblem(lotka_volterra,u0,tspan,p)
  #  sol = solve(prob,Tsit5())
    z = Matrix{Float64}(undef, n,2)
    dz_dp = zeros(4, 2, n)

    for i = 1:n
          extract!(z, dz_dp, i)#lambdavec[i]=max(0,dot(-gll(cos(t).*x+sin(t).*v,y),-sin(t).*x+cos(t).*v))#/txinc!(ret, i)
    end
#    dz_dp =
#    z,dz_dp = extract_local_sensitivities.(sol,1.:56)
#    dz_dp = convert(Array,VectorOfArray(dz_dp))'
    df_dz=(log.(y)-log.(abs.(z))).*abs.(z).^(-1).*([sigma1, sigma2]'.^(-2)) #How to remove this abs constraint?

    cums=zeros(length(param))

    for j = 1:n
    #      println(cums)
#      if j==1
#        cums=zeros(length(param))
#      end
      cums.+=dz_dp[:,:,j]*df_dz[j,:]
    end
    return -cums
end

x = copy(p)
prior = MvNormal([1,.05,1,.05],0.05*Matrix(I,4,4))
v = Distributions.rand(prior,1)[:,1]
v=[1,.05,1,.05]

function lambda!(lambdavec::AbstractVector{T}, i::Int, t::T,
    x::Vector{<:Real},v::Vector{<:Real}) where T
   lambdavec[i] = max(0,dot(-gradll2(cos(t).*x+sin(t).*v),-sin(t).*x+cos(t).*v))#/t
   nothing
end;

lambdavec = zeros(16)#Vector{Float64}(undef, 8)
tvec = range(0.0, stop=2*pi, length=16)
x=copy(p)
for i = 1:16
#    println(lambdavec[i])
    lambda!(lambdavec, i, tvec[i],x,v)#lambdavec[i]=max(0,dot(-gll(cos(t).*x+sin(t).*v,y),-sin(t).*x+cos(t).*v))#/txinc!(ret, i)
end

cut = quantile(lambdavec,0.9)
idx = findall(x->x<cut, lambdavec)
lambdavec = lambdavec[idx]
tvec=tvec[idx]

a = max(0,lambdavec[1])
b, idx = findmax(vcat(0,deleteat!((lambdavec.-a)./tvec, 1)))

lambdahat = findmax(lambdavec)[1]#maximum(lambdavec.+a)
# affine segment
using Random
if (b!=0) && (length(idx)!=0)
  if a!=0
      tau = (-a+sqrt(a^2+2*b*Random.randexp()))/b
  else
      tau = sqrt(2*Random.randexp()/b)
  end
      # In case tau simulatedd falls in the cte bound instead of affine
      # adjust it, since real tau shouldd be even greater
      if tau-tvec[idx] > 0
          delta = tau-tvec[idx]
          tau+=(delta^2*b/2)/lambdahat
      end
  #  constant only segment
  else
      tau = Random.randexp()/lambdahat
  end
  #plot(tvec,lambdavec.+a)
  #    plot!(tvec,min.(maximum(lambdavec.+a),tvec.*b.+a))
  lambdabar = min(a+b*tau, lambdahat)
end



nextevent_boo_ode_numeric

using PDSampler

lref = 0.01

algname = "BOOMERANG"
T    = 100  # length of path generated

p0 = Distributions.rand(prior,1)[:,1]
#for i in 1:20
v0 = Distributions.rand(prior,1)[:,1]
# Define a simulation
nextev_mhwithinpdmp(p, v) = nextevent_boo_ode_cte(lambda!,p, v)
sim_mhwithinpdmp1 = Simulation(p0, v0, T, nextev_mhwithinpdmp, gradll2,
              nextbd, lref, algname, MHsampler=false;
              mass=massmatrix, maxsegments=5000)





a1=gradll2([0.545,.028,.803,0.024])


a2=gradll2([3.545,.028,1.803,10.024])

＃prob = ODEForwardSensitivityProblem(lotka_volterra,u0,tspan,[0.545,.028,.803,0.024])
using Plots
plot(solve(ODEProblem(lotka_volterra,u0,tspan,[0.545,.028,.803,0.024]),Tsit5()))
  plot!(1:21,y)
  plot!(solve(ODEProblem(lotka_volterra,u0,tspan,[0.845,.028,1.203,.124]),Tsit5()))
  #plot!(tsvec,y)

plot(1:21,y)
gradient = 1/sigma^2*cums
    z[j,:]
    for i = 1:n
          extract!(z, dz_dp, i)#lambdavec[i]=max(0,dot(-gll(cos(t).*x+sin(t).*v,y),-sin(t).*x+cos(t).*v))#/txinc!(ret, i)
    end

    return (sigma^2)^(-1.)*dz_dp*df_dz
end

function lambda!(lambdavec::AbstractVector{T}, i::Int, t::T,
    x::Vector{<:Real},y::Vector{<:Real},v::Vector{<:Real}) where T
   lambdavec[i] = max(0,dot(-gradll(cos(t).*x+sin(t).*v,y),-sin(t).*x+cos(t).*v))#/t
   nothing
end;


using Logging
Logging.handle_message()


problem(p) = ODELocalSensitivityProblem(lotka_volterra,u0,tspan,p)
solution(problem) = solve(problem,DP8())
zed(solution) = extract_local_sensitivities(solution,1.)[1]
dz_dp_aux(solution) = convert(Array,VectorOfArray(extract_local_sensitivities(solution,1.)[2]))'
df_dz_aux(zed)=(log.(y)-log.(zed(solution(problem(p))))).*zed.^(-1)

gradll(p) = (sigma^2)^(-1)*dz_dp_aux(solution(problem(p)))*df_dz
Matrix{Float64}(dz_dp,4,2)
y=[36,20]


da = dp[1]
plot(sol.t,da',lw=3)

function sum_of_solution(u0,p)
  _prob = remake(prob,u0=u0,p=p)
  sum(solve(prob,Tsit5(),saveat=0.1,sensealg=QuadratureAdjoint()))
end
du01,dp1 = Zygote.gradient(sum_of_solution,u0,p)

Pkg.add("Flux")
Pkg.add("Plots")

using Flux, Plots

p = [2.2, 1.0, 2.0, 0.4] # Initial Parameter Vector
sol=solve(prob,Tsit5(),p=p,saveat=0.0:0.1:10.0,sensealg=BacksolveAdjoint())

  Array(solve(prob,Tsit5(),p=p,saveat=0.0:0.1:10.0,sensealg=BacksolveAdjoint())) # Concretize to a matrix
loss_adjoint() = sum(abs2,x-1 for x in predict_adjoint())

data = Iterators.repeated((), 100)
opt = ADAM(0.1)
cb = function () #callback function to observe training
  display(loss_adjoint())
  # using `remake` to re-create our `prob` with current parameters `p`
  display(plot(solve(remake(prob,p=p),Tsit5(),saveat=0.0:0.1:10.0),ylim=(0,6)))
end

# Display the ODE with the initial parameter values.
cb()

Flux.train!(loss_adjoint, Flux.params(p), data, opt, cb = cb)

x,dp = extract_local_sensitivities(sol)
