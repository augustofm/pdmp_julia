using PDSampler
using LinearAlgebra         #For Cholesky
import Random               #For generating random vectors               #For plotting
using Plots
using DelimitedFiles
using Distributions

Random.seed!(123);

# Metropolis within PDMP
x0=rand(Normal(0,0.8), 30)
x1=vcat(rand(Normal(-6,0.8),20), rand(Normal(2,0.8),10))

x=vcat(x0,x1)
Y=vcat(repeat([-1];outer=[30]),repeat([1];outer=[30]))
plot(x0,Y[1:30].+1,
    legend=:bottomleft,
    size=(600,600),
    #title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    ylabel = "p(y = +1 | x)",
    xlabel = "input, x",
    ylims = (-0.1,1.1),
    xlims = (-9,5),
    #fmt = :png,
    color=:black,shape=:x,
    label = "Class -1")
    scatter!(x1,Y[31:60],color=:black,label = "Class +1")
    #savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/scatter_gpc")

p=length(x)
# Boundaries
ns, a = Matrix{Float64}(I, p, p), repeat([-Inf];outer=[p])#zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

# Dist
kx(a,b,theta) = theta[1]^2*exp((-0.5*(1/theta[2]^2)*(a-b)'*(a-b)))
function Kernelmatrix(theta)
    K = Matrix{Float64}(I, p, p)
    for i in 1:p
        for j in 1:p
            K[i,j]=kx(x[i],x[j],theta)
        end
    end
    K += norm(K)/100000*Matrix{Float64}(I,p,p)
    K
end
function Kernelvector(theta,xstar)
    k = Matrix{Float64}(I, 1, p)
    for j in 1:p
        k[1,j]=kx(x[j],xstar,theta)
    end
    k'
end

#----------------------------------------------------------------
## MH within Boomerang CTE BOUND


theta0 = [7., 2.5]
massmatrix = Kernelmatrix(theta0)

lref = 0.01

prior = MvNormal(zeros(p),massmatrix)
algname = "BOOMERANG"
T    = 1000000  # length of path generated

MH = true

lmh　= 0.01
epsilon=0.2
nmh=20


u1 = Uniform(0.1, 10)# (maximum(x)-minimum(x))/2)Gamma(2.5,2.5)##Gamma(2.5,2.5)
u2 = Uniform(0.1, 10)#(maximum(x)-minimum(x))/2) Gamma(2.5,2.5)##Gamma(2.5,2.5)#
ytarget(x,y) = pdf(MvNormal(zeros(p),Kernelmatrix(y)),x)*pdf(u1,y[1])*pdf(u2,y[2])#pdf(MvNormal([7.,3.],Matrix{Float64}(I, 2, 2)), y)
#nextev_mhwithinpdmp(f, theta, v) = nextevent_boo_gpc_affine(gradll, f, theta, v)

theta0 = [7., 2.]


ess_vector_boo = Vector{Float64}()
ess_vector_raw_boo = Vector{Float64}()
mh_acc_vector_boo = Vector{Float64}()

#for i in 1:20
f0 = Distributions.rand(prior,1)[:,1]
v0 = Distributions.rand(prior,1)[:,1]

# Define a simulation
gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
nextev_mhwithinpdmp(f, theta, v) = nextevent_boo_gpc_cte(gradll,f, theta, v)
sim_mhwithinpdmp1 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                  mass=massmatrix, maxsegments=10000)
(path_mhwithinpdmp1, details_mhwithinpdmp1) = simulate(sim_mhwithinpdmp1)
#plot(path_mhwithinpdmp3.ys[1,:],path_mhwithinpdmp3.ys[2,:])
boo_part1=Path(path_mhwithinpdmp1.xs, path_mhwithinpdmp1.ts)
ess1=esspath_boo(boo_part1)#samples3)
ess_sec1=mean(ess1[1])/details_mhwithinpdmp1["clocktime"]
#print(ess_sec)
#print("\n")
#    push!(ess_vector_raw_boo, ess_sec)
#    push!(ess_vector_boo,ess_sec/details_mhwithinpdmp3["clocktime"])
#    push!(mh_acc_vector_boo,details_mhwithinpdmp3["MHaccrejratio"])
#end

Tp = 0.999 * path_mhwithinpdmp1.ts[end]
gg = range(0, stop=Tp, length=10000)
samples1=samplepath_boo(boo_part1,gg)#[0.0:0.1:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])
fbarboo = mean(samples1,dims=2)
scatter(vcat(x0,x1),fbarboo)
scatter(vcat(x0,x1),(exp.(-fbarboo).+1).^(-1))

plot(samples1[2,:],samples1[30,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-20,20),
    xlims = (-20,20),
    #fmt = :png,
    color=:black)
    scatter!(boo_part1.xs[2,:],boo_part1.xs[30,:],legend=false)

#esspath_boo(boo_part3)

#----------------------------------------------------------------
# MH within BOOMERANG AFFINE BOUND

gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
nextev_mhwithinpdmp(f, theta, v) = nextevent_boo_gpc_affine(gradll,f, theta, v)
sim_mhwithinpdmp2 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                  mass=massmatrix, maxsegments=10000)
(path_mhwithinpdmp2, details_mhwithinpdmp2) = simulate(sim_mhwithinpdmp2)
#plot(path_mhwithinpdmp3.ys[1,:],path_mhwithinpdmp3.ys[2,:])
boo_part2=Path(path_mhwithinpdmp2.xs, path_mhwithinpdmp2.ts)
ess2=esspath_boo(boo_part2)#samples3)
ess_sec2=mean(ess2[1])/details_mhwithinpdmp2["clocktime"]
#print(ess_sec)
#print("\n")
#    push!(ess_vector_raw_boo, ess_sec)
#    push!(ess_vector_boo,ess_sec/details_mhwithinpdmp3["clocktime"])
#    push!(mh_acc_vector_boo,details_mhwithinpdmp3["MHaccrejratio"])
#end

Tp = 0.999 * path_mhwithinpdmp2.ts[end]
gg = range(0, stop=Tp, length=10000)
samples2=samplepath_boo(boo_part2,gg)#[0.0:0.1:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])
fbarboo = mean(samples2,dims=2)
scatter(vcat(x0,x1),fbarboo)
scatter(vcat(x0,x1),(exp.(-fbarboo).+1).^(-1))

plot(samples2[2,:],samples2[30,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-20,20),
    xlims = (-20,20),
    #fmt = :png,
    color=:black)
    scatter!(boo_part2.xs[2,:],boo_part2.xs[30,:],legend=false)

#----------------------------------------------------------------
# MH within BOOMERANG NUMERICAL BOUND

gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
nextev_mhwithinpdmp(f, theta, v) = nextevent_boo_gpc_numeric(gradll,f, theta, v)
sim_mhwithinpdmp3 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                  mass=massmatrix, maxsegments=10000)
(path_mhwithinpdmp3, details_mhwithinpdmp3) = simulate(sim_mhwithinpdmp3)
#plot(path_mhwithinpdmp3.ys[1,:],path_mhwithinpdmp3.ys[2,:])
boo_part3=Path(path_mhwithinpdmp3.xs, path_mhwithinpdmp3.ts)
ess3=esspath_boo(boo_part3)#samples3)
ess_sec3=mean(ess3[1])/details_mhwithinpdmp3["clocktime"]
#print(ess_sec)
#print("\n")
#    push!(ess_vector_raw_boo, ess_sec)
#    push!(ess_vector_boo,ess_sec/details_mhwithinpdmp3["clocktime"])
#    push!(mh_acc_vector_boo,details_mhwithinpdmp3["MHaccrejratio"])
#end

Tp = 0.999 * path_mhwithinpdmp3.ts[end]
gg = range(0, stop=Tp, length=10000)
samples3=samplepath_boo(boo_part3,gg)#[0.0:0.1:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])
fbarboo = mean(samples3,dims=2)
scatter(vcat(x0,x1),fbarboo)
scatter(vcat(x0,x1),(exp.(-fbarboo).+1).^(-1))

plot(samples3[2,:],samples3[30,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-20,20),
    xlims = (-20,20),
    #fmt = :png,
    color=:black)
    scatter!(boo_part3.xs[2,:],boo_part3.xs[30,:],legend=false)

#----------------------------------------------------------------

# MH within BPS ------------------------------------------------
algname = "BPS"
#gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)
gradll(f,invmatrix) = (Y.+1)./2-(exp.(-f).+1).^(-1)-invmatrix*f#gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)
nextev_mhwithinpdmp(f, theta, v, invmatrix) = nextevent_bps_gpc_affine(gradll, f, theta, v, invmatrix)

#ess_vector_bps = Vector{Float64}()
#ess_vector_raw_bps = Vector{Float64}()
#mh_acc_vector_bps = Vector{Float64}()

#for i in 1:20
#f0 = Distributions.rand(prior,1)[:,1]
#v0 = Distributions.rand(prior,1)[:,1]
# Define a simulation
sim_mhwithinpdmp4 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                  mass=massmatrix, maxsegments=10000)
(path_mhwithinpdmp4, details_mhwithinpdmp4) = simulate(sim_mhwithinpdmp4)
#plot(path_mhwithinpdmp4.ys[1,:],path_mhwithinpdmp4.ys[2,:])
boo_part4=Path(path_mhwithinpdmp4.xs, path_mhwithinpdmp4.ts)
ess4=esspath(boo_part4)#samples3)
ess_sec4=mean(ess4[1])/details_mhwithinpdmp4["clocktime"]
print(ess_sec)
print("\n")
#    push!(ess_vector_raw_bps, ess_sec)
#    push!(ess_vector_bps,ess_sec/details_mhwithinpdmp4["clocktime"])
#    push!(mh_acc_vector_bps,details_mhwithinpdmp4["MHaccrejratio"])
#end
ess_vector_bps

Tp = 0.999 * path_mhwithinpdmp4.ts[end]
gg = range(0, stop=Tp, length=100000)
samples4=samplepath(boo_part4,gg)#[0.0:0.1:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])
fbarbps = mean(samples4,dims=2)
scatter(vcat(x0,x1),fbarbps)
scatter(vcat(x0,x1),(exp.(-fbarbps).+1).^(-1))

plot(samples4[2,:],samples4[30,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-20,20),
    xlims = (-20,20),
    #fmt = :png,
    color=:black)
    scatter!(boo_part4.xs[2,:],boo_part4.xs[30,:],legend=false)


using StatsPlots: @df, StatsPlots
using DataFrames
df = DataFrame(Algorithm =vcat(repeat(["BOO"];outer=[20]),repeat(["BPS"];outer=[20])),
                ESS_sec= vcat(log.(ess_vector_boo),log.(ess_vector_bps)))
@df df boxplot(:Algorithm,:ESS_sec, alpha=0.5,legend=false,
    ylabel = "log ESS/sec")
    savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/box_gpc_ess_sec")

df = DataFrame(Algorithm =vcat(repeat(["BOO"];outer=[20]),repeat(["BPS"];outer=[20])),
                ESS= vcat(log.(ess_vector_raw_boo),log.(ess_vector_raw_bps)))
@df df boxplot(:Algorithm,:ESS, alpha=0.5,legend=false,
    ylabel = "raw ESS")
    savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/box_gpc_ess_raw")

@df df boxplot([:Algorithm],[:ESS_sec],color=[:pink :white])


scatter(x,Y,
    legend=false,
    size=(600,600),
    #title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    ylabel = "p(y = +1 | x)",
    xlabel = "input, x",
    ylims = (-1.25,1.25),
    xlims = (-10,5),
    #fmt = :png,
    color=:black)

scatter(x, fbarboo,
    legend=false,
    size=(600,600),
    #title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    ylabel = ",latent function, f(x)",
    xlabel = "input, x",
    color=:black)
    scatter!(x, fbarbps,color=:blue)
fig, axs = PyPlot.subplots(2, 2, figsize=(10,10))

theta1bar = mean(path_mhwithinpdmp3.ys[1,:],dims=1)
theta2bar = mean(path_mhwithinpdmp3.ys[2,:],dims=1)

theta1bar = mean(path_mhwithinpdmp4.ys[1,:],dims=1)
theta2bar = mean(path_mhwithinpdmp4.ys[2,:],dims=1)


axs[1].boxplot(data) # Basic
axs[3].boxplot(data,true)

boxplot(rand(1:4, 1000), randn(1000))
B=0.05
x=copy(f0)
propose(x) = MvNormal(sqrt(1-B^2)*x,B*massmatrix)
blocksize=100000
d=length(f0)
# storing xs as a single column for efficient resizing
xs, ts = zeros(d*100*blocksize), zeros(blocksize)

t, i = 0.0, 1
xs[1:d] = x

stdNormal = MvNormal(zeros(d),Matrix{Float64}(I, d, d))
invmatrix = inv(massmatrix)
aux(u) = -log(prod(logistic.(u)))+0.5*sqrt(sum(massmatrix^-0.5*u).^2)
acc(x,y) = min(1,exp(aux(x)-aux(y)))

naccepted = 0
nrejected = 0
let i = 1, naccepted=0, nrejected=0
    while i < 200000
        x=copy(f0)
        x_star = Distributions.rand(propose(x),1)
        if rand() < min(1.0, acc(x,x_star))
            x = copy(x_star)
            naccepted += 1
        else
            x = copy(x)
            nrejected += 1
        end
        #print(i)
        xs[((i-1) * d + 1):(i * d)] = x
        i+=1
    end
end

path = reshape(xs[1:(200000*d)], (d,200000))

fbar = mean(path,dims=2)
scatter(vcat(x0,x1),(exp.(-fbar).+1).^(-1))
scatter(vcat(x0,x1),fbar)

plot(path[30,:],path[60,:],
    legend=false,
    size=(600,600),
    title="pCN ",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    #fmt = :png,
    color=:black)
    scatter!(path[30,:],path[60,:],legend=false)





## PROBIT LIKELIHOOD

f=copy(f0)
v=copy(v0)

n0 = pdf(Normal(0,1),0)
m = Y.*n0*(-2)
m = sqrt(sum(m.^2))

aux = sqrt.(f.^2+v.^2)
cdf_aux = cdf.(Normal(0,1),-aux)
M = Y.*aux*n0./cdf_aux.^2



m=1/4096
x=f0[1]
v=v0[1]
c = log(Random.randexp()+m*exp(x))
a = copy(x)
b = copy(v)
t = asin(c*sqrt(b^2+a^2)/(b^2+a^2))-atan(a/b)
