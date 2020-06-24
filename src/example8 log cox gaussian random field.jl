using LinearAlgebra         #For Cholesky
using Random               #For generating random vectors               #For plotting
using Plots
using DelimitedFiles
using Distributions
using Random


d=30
p=d^2
s=1/d^2
b=1/6

# hiper parameters for MH
sigma2=1.9
mu = log(126)-sigma2/2

sigmaf = function(i,j,idash,jdash){
  return((i-idash)^2+(j-jdash)-2)
}
kernel = function(i,j,idash,jdash,sigma2,b,d)
  return(sigma2*exp(-sigma(i,j,idash,jdash))/(b*d))
end

Z=Matrix{Float64}(I, p, p)

for n in 1:p
  ni = ceil((n+1)/d)
  nj = (n+1) % d
  if nj == 0
    nj=d
  end
  for m in 1:d^2
    mi=ceil((m+1)/d)
    mj = (m+1)%d
    if mj==0
       mj=d
    end
    Z[n,m]=sqrt((ni-mi)^2+(nj-mj)^2)
  end
end
Z=sigma2*exp.(-Z/(b*d))
Zinv = inv(Z)
L = cholesky(Z).L

X = randn(d^2)'*L.+mu
X = Array(X')
Y=zeros(d^2)#c()#np.zeros((d^22,))
for i in 1:d^2
 #Y[i]=rand(Poisson(s*exp(X[i])))
 Y[i]=rand(Poisson(s*exp(X[i])))
end

x = 1:d
y = 1:d

using ColorSchemes
solar = ColorSchemes.solar.colors

latentfield = reshape(X,d,d)
heatmap(latentfield, aspect_ratio = 1, color = :matter,
  clim=(-5.,5.),
  xaxis = false,
  yaxis = false,
  legend = false,
  title = "Latent Field")

lp = reshape(s.*X,d,d)
heatmap(lp, aspect_ratio = 1,color=:matter,
  xaxis = false,
  yaxis = false,
  legend = false,
  title = "Latent Process")

data = reshape(Y,d,d)
heatmap(data, aspect_ratio = 1,color = :matter,
  clim=(0.,10.),
  xaxis = false,
  yaxis = false,
  legend = false,
  title = "Observed Data")


# boundaries, if any
ns, a = Matrix{Float64}(I, p, p), repeat([-Inf];outer=[p])#zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)



#theta0 = [7., 2.5]
#theta0 = [10, 5]
#theta0 = [rand(u1,1)[1], rand(u2,1)[1]]
massmatrix = Matrix{Float64}(I, p, p)
massmatrix = Z
prior = MvNormal(zeros(p),massmatrix)

x0 = Distributions.rand(prior,1)[:,1]
v0 = Distributions.rand(prior,1)[:,1]


T=10000
lref=5#1.5#0.05
lmh=0.000001
epsilon=0.7
nmh=20
#theta0 = [rand(u1,1)[1], rand(u2,1)[1]]
theta0=[7.,2.5]


u1 = Gamma(1,3)
u2 = InverseGamma(2,10)

ytarget(x,y) = (y[2]>0.0) ? pdf(MvNormal(zeros(p),Kernelmatrix(y)),x)*pdf(u1,y[1])*pdf(u2,y[2]) : 0
caltau = function(x,v,Y)
      if v>0.
          tau = 1/v*(log(Random.randexp()/s+exp(x))-x)
      elseif Y>0.
          tau = -Random.randexp()/(v*Y)
      else
          tau = Inf
      end
      return(tau)
end





algname="BPS"
# Define a simulation
gradll(x,invmatrix) = Y-s*exp.(x)-invmatrix*(x.-mu)
nextev_mhwithinpdmp(x, theta, v, invmass) = nextevent_bps_lcp_super(gradll, x, theta, v, invmass)
sim_mhwithinpdmp1 = Simulation(x0, v0, T, nextev_mhwithinpdmp, gradll,
              nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
              Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
              mass=massmatrix, maxsegments=100000)
(path_mhwithinpdmp1, details_mhwithinpdmp1) = simulate(sim_mhwithinpdmp1)

part1=Path(path_mhwithinpdmp1.xs, path_mhwithinpdmp1.ts)
Tp = 0.999 * path_mhwithinpdmp1.ts[end]
gg = range(0, stop=Tp, length=200000)
samples1=samplepath(part1,gg)#[0.0:0.1:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])

plot(samples1[804,:],samples1[805,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    #fmt = :png,
    color=:black)
    scatter!(part1.xs[804,:],part1.xs[805,:],legend=false)

path = reshape(xs[1:(200000*d)], (d,200000))

fbar = mean(samples1,dims=2)
fvar = cov(samples1,dims=2)

latentfield = reshape(X,d,d)
heatmap(latentfield, aspect_ratio = 1, color = :matter,
  clim=(-5.,5.),
  xaxis = false,
  yaxis = false,
  legend = false,
  title = "True Latent Field")

latentfield = reshape(fbar,d,d)
heatmap(latentfield, aspect_ratio = 1, color = :matter,
  clim=(-5.,5.),
  xaxis = false,
  yaxis = false,
  legend = false,
    title = "Simulated Latent Field")

lpvar = reshape(diag(fvar),d,d)
heatmap(lpvar, aspect_ratio = 1, color = :matter,
    xaxis = false,
    yaxis = false,
    legend = false,
    title = "Posterior Variance")


scatter(vcat(x0,x1),(exp.(-fbar).+1).^(-1))
scatter(vcat(x0,x1),fbar)




algname="BOOMERANG"
# Define a simulation
gradll(x,invmatrix) = Y-s*exp.(x)
nextev_mhwithinpdmpboo(x, theta, v) = nextevent_boo_lcp_cte(gradll, x, theta, v)
sim_mhwithinpdmpboo = Simulation(x0, v0, T, nextev_mhwithinpdmpboo, gradll,
              nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
              Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
              mass=massmatrix, maxsegments=10000)
(path_mhwithinpdmpboo, details_mhwithinpdmpboo) = simulate(sim_mhwithinpdmpboo)

partboo=Path(path_mhwithinpdmpboo.xs, path_mhwithinpdmpboo.ts)
Tp = 0.999 * path_mhwithinpdmpboo.ts[end]
gg = range(0, stop=Tp, length=20000)
samplesboo=samplepath_boo(partboo,gg)#[0.0:0.1:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])

plot(samplesboo[804,:],samplesboo[805,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    #fmt = :png,
    color=:black)
    scatter!(partboo.xs[804,:],partboo.xs[805,:],legend=false)


fbar = mean(samplesboo,dims=2)
fvar = cov(samplesboo,dims=2)

latentfield = reshape(X,d,d)
latentfield = reshape(fbar,d,d)
heatmap(latentfield, aspect_ratio = 1, color = :matter,
  clim=(-5.,5.),
  xaxis = false,
  yaxis = false,
  legend = false,
  title = "Latent Field")

#df = CSV.read("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/examples/lotka-volterra.csv")
df = CSV.read("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/examples/lotka-volterra2.csv")

ns, a = Matrix{Float64}(I, 4, 4), repeat([-Inf];outer=[4])#zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

tsvec = Array(df[:year])
hare = df[:,3]
lynx = df[:,2]#Array(df[:lynx])
y=hcat(hare,lynx)
n=length(tsvec)
plot(tsvec,y)
#u0=y[1,:]
sigma1=0.25
sigma2=0.252

sigma1=.5
sigma2=.5

u0=[33.9,5.93]
u1 = Normal(4,3)
u2 = Normal(3,3)
ytarget(x,y) = (y[1]>0.0 && y[2]>0.0) ? pdf(MvNormal(zeros(p),Kernelmatrix(y)),x)*pdf(u1,y[1])*pdf(u2,y[2]) : 0


tspan = (0.0,20.0)

0.064  0.465  0.542  0.630  1076    1
theta[2]   0.028   0.000 0.004  0.022  0.027  0.033  1195    1
theta[3]   0.803   0.003 0.092  0.692  0.797  0.926   993    1
theta[4]   0.024   0.000 0.004


initial = Distributions.MvNormal([0.545,.028,.803,0.024],[0.6, 0.01, 0.6, 0.01].*Matrix{Float64}(I, 4, 4))
v0 = Distributions.rand(initial,1)[:,1]
p0 = [0.545,.028,.803,0.024]
#p0 = Distributions.rand(initial,1)[:,1]

function extract!(zvec::AbstractMatrix{T}, der_total::AbstractArray{T}, i::Int) where T
     zvec[i,:] = extract_local_sensitivities(sol,Float32(i-1))[1]
     der_total[:,:,i]=convert(Array,VectorOfArray(extract_local_sensitivities(sol,Float32(i-1))[2]))'
     nothing
end

function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, γ, δ = p #, δ
  du[1] = dx = α*x - β*x*y
  du[2] = dy = -γ*y + δ*x*y #δ
end

#prob = ODEForwardSensitivityProblem(LotkaVolterraSensitivity,u0,tspan,param)
#sol = solve(prob,DP8())
#x,dp = extract_local_sensitivities(sol)
#da = dp[4]
#plot(sol.t,da',lw=3,title="dz/dtheta_4",xlabel="t")


function gradll(param::Vector{<:Real})
    prob = ODEForwardSensitivityProblem(lotka_volterra,u0,tspan,param)
  #  sol = solve(prob,Rodas4(autodiff=false),dt = 1e-30, adaptive=false, alg_hints = [:stiff])
  #  sol = solve(prob,Tsit5(),saveat=1,sensealg=QuadratureAdjoint(),alg_hints = [:stiff],maxiters=1e7,abstol=1/10^14,reltol=1/10^14)
    sol = solve(prob,Rodas4(autodiff=false),force_dtmin=true,dt = tspan[2]/1000,abstol=1e-8, reltol=1e-4, maxiters=10^6)
  #  dt = 1e-3, adaptive=false
  #  sol = solve(prob,BS5(),saveat=1,sensealg=QuadratureAdjoint(),alg_hints = [:stiff],abstol=1/10^14,reltol=1/10^14)
    println(sol.retcode)
    if sol.retcode ∈ [:Success,:Unstable]
      global sol
    #  prob = ODEProblem(lotka_volterra,u0,tspan,p)
    #  sol = solve(prob,Tsit5())
      n=21
      z = Matrix{Float64}(undef,n,2)
      dz_dp = zeros(4, 2, n)

      for i = 1:n
            extract!(z, dz_dp, i)#lambdavec[i]=max(0,dot(-gll(cos(t).*x+sin(t).*v,y),-sin(t).*x+cos(t).*v))#/txinc!(ret, i)
      end
  #    dz_dp =
  #    z,dz_dp = extract_local_sensitivities.(sol,1.:56)
  #    dz_dp = convert(Array,VectorOfArray(dz_dp))'
      df_dz=(log.(y)-log.(abs.(z))).*(-z).^(-1).*([sigma1, sigma2]'.^(-2)) #*abs.(z).^(-1).#How to remove this abs constraint?

      cums=zeros(length(param))

      for j = 1:n
      #      println(cums)
  #      if j==1
  #        cums=zeros(length(param))
  #      end
        cums.+=dz_dp[:,:,j]*df_dz[j,:]
      end
      return -cums
    else
      return repeat([NaN];outer=4)
    end
end


function lambda!(lambdavec::AbstractVector{T}, i::Int, t::T,
    x::Vector{<:Real},v::Vector{<:Real}, nointerrupt::Bool) where T
   if nointerrupt
     lambdavec[i] = max(0,dot(-gradll(cos(t).*x+sin(t).*v),-sin(t).*x+cos(t).*v))#/t
   else
     lambdavec[i] = NaN
   end
   nothing
end;


algname = "BOOMERANG"
lref = 1
massmatrix = [0.06, 0.01, 0.06, 0.01].*Matrix{Float64}(I, 4, 4)
T = 10000  # length of path generated

epsilon=0.1
#y0=u0, ytarget=ytarget,
#     Sigmay = Kernelmatrix
#, epsilon=epsilon, nmh=nmh,lambdamh=lmh

nextev_mhwithinpdmp(p, v) = nextevent_boo_ode_numeric(lambda!,p, v)
sim_mhwithinpdmp = Simulation(p0, v0, T, nextev_mhwithinpdmp, gradll,
              nextbd, lref, algname,  MHsampler=false;
              mass=inv(massmatrix),maxsegments=50)
(path_mhwithinpdmp, details_mhwithinpdmp) = simulate(sim_mhwithinpdmp)
boo_part=Path(path_mhwithinpdmp.xs, path_mhwithinpdmp.ts)
Tp = 0.999 * path_mhwithinpdmp.ts[end]
gg = range(0, stop=Tp, length=2000)
samples1=samplepath_boo(boo_part,gg)#[0.0:0.1:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])

plot(samples1[2,:],samples1[4,:])
plot(samples1[1,:],samples1[3,:])
plot(samples1[1,:],samples1[2,:])
plot(samples1[1,:])
plot(samples1[2,:])
plot(samples1[3,:])
plot(samples1[4,:])
mean(samples1[1,:])
mean(samples1[2,:])
mean(samples1[3,:])
mean(samples1[4,:])

mean(samples1,dims=2)
sqrt.(diag(cov(samples1,dims=2)))

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
