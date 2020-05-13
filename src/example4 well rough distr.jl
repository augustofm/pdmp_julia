## Well Rough Distribution


using PDSampler             #Main library
using LinearAlgebra         #For Cholesky
import Random               #For generating random vectors
using Plots                 #For plotting
using DelimitedFiles
using Distributions         #For mvg, Poisson...


#normal to faces and intercepts
#ns, a = Matrix{Float64}(I, p, p), [-Inf,Inf]#zeros(p)
p = 2
# normal to faces and intercepts
#ns, a = Matrix{Float64}(I, p, p), zeros(p)#repeat([-Inf];outer=[p])#zeros(p)
ns, a = Matrix{Float64}(I, p, p), repeat([-Inf];outer=[p])#zeros(p)
geom  = Polygonal(ns, a)
# for a given ray, which boundary does it hit?
nextbd(x, v) = nextboundary(geom, x, v)

Random.seed!(12)

sigma1=100
sigma2=4

P1 = [[1/(sigma1) 0];[0 1/(sigma1)]]
P1 *= P1'
P1 += norm(P1)/(sigma1^2)*Matrix{Float64}(I,p,p)
C1  = inv(P1); C1 += C1'; C1/=2;
L1  = cholesky(C1)
mu = zeros(p)#+[3.,3.]
mvg = MvGaussianStandard(mu,C1)

massmatrix = [[sigma1^2 0];[0 sigma1^2]]#Matrix{Float64}(I, p, p)

# Building a BPS Simulation
U(x) = -(sum(x.^2)/20000+cos(pi*x[1]/4)+cos(pi*x[2]/4))+dot(x,x)/2#gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU

#x0 = randn(2)#[4,4]#,0.1,0.1,0.1]
prior = MvNormal(zeros(2),C1)
v0 = Distributions.rand(prior,1)[:,1]
x0 = Distributions.rand(prior,1)[:,1]
T    = 10000.0   # length of path generated
lref = 1     # rate of refreshment


gradll(x) = (pi/sigma2)*sin.(pi*x/sigma2)#cos(pi*x[1]/4)+cos(pi*x[2]/4))+x#gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU
nextev_boo(x, v) = nextevent_boo_rough_affine(gradll,mvg, x, v)

# Define a simulation
algname = "BOOMERANG"
sim_boo = Simulation( x0, v0, T, nextev_boo, gradll,
                  nextbd, lref,  mass=massmatrix, algname; maxsegments = 80000000, maxgradeval = 20000000)
(path_boo, details_boo) = simulate(sim_boo)
samples = samplepath_boo(path_boo,[0.0:0.01:round(path_boo.ts[end]-0.5,digits=0);])

plot(samples[1,:],samples[2,:],
    legend=false,
    size=(600,600),
    #title="Boomerang Sampler (2-dimensional Gaussian target)",
    xlabel = "X1(t)",
    ylabel = "X2(t)",
    #ylims = (0,50),
    #xlims = (0,50),
    #ylims = (-300,300),
    #xlims = (-300,300),
    #fmt = :png,
    color=:black)
    #scatter!(path_boo.xs[1,:],path_boo.xs[2,:],legend=false)
    #annotate!(4, -3, text(string("nsegments: ", details_boo["nsegments"],"\n",
    #"nbounce: ", details_boo["nbounce"],"\n",
    #"nrefresh: ", details_boo["nrefresh"]), :black, :right, 10))
#cor(path_bps.xs[2,:], path_bps.xs[1,:])
details_boo

axis1(i)=samples[1,i]
axis2(i)=samples[2,i]

anim = Animation()
p = plot(samples[1,1:1000],samples[2,1:1000],lab="x(t)",color=:black)
#scatter!(1, [axis1,axis2])
scatter!(1, lab="")
for i in 1:500#length(samples)
    p[2] = [axis1(i)], [axis2(i)]
    frame(anim)
end
gif(anim)



# Building a BPS Simulation
gradll(x) = (pi/sigma2)*sin.(pi*x/sigma2)-x./(sigma1^2)#cos(pi*x[1]/4)+cos(pi*x[2]/4))+x#gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU
nextev_bps(x, v) = nextevent_bps_rough_affine(gradll, x, v)

T    = 10000.0   # length of path generated
lref = 1     # rate of refreshment
algname = "BPS"

#x0 = randn(2)#[50,50]
#v0 = randn(2)#[1,1]
sim_bps = Simulation(  x0, v0, T, nextev_bps, gradll,
                  nextbd, lref,  mass=massmatrix, algname; maxsegments = 2000000, maxgradeval = 2000000)
(path_bps, details_bps) = simulate(sim_bps)

plot(path_bps.xs[1,:],path_bps.xs[2,:],
    legend=false,
    size=(600,600),
    #title="Bouncy Particle Sampler (2-dimensional Gaussian target)",
    xlabel = "x1(t)",
    ylabel = "x2(t)",
    ylims = (0,50),
    xlims = (0,50),
    color=:black)
    #scatter!(path_bps.xs[1,:],path_bps.xs[2,:],legend=false)
    #annotate!(-200, -200, text(string("nsegments: ", details_bps["nsegments"],"\n",
    #"nbounce: ", details_bps["nbounce"],"\n",
    #"nrefresh: ", details_bps["nrefresh"]), :black, :right, 10))
    #savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/scatter_rough")

samples = samplepath(path_bps,[0.0:0.1:round(path_bps.ts[end]-1,digits=0);])


axis1(i)=samples[1,i]
axis2(i)=samples[2,i]

anim = Animation()
p = plot(samples[1,51000:52350],samples[2,51000:52350],color=:black,
    xlabel = "x1(t)",
    ylabel = "x2(t)",
    xlims = (-50,50),
    ylims = (50,100),
    legend=false)
#scatter!(1, [axis1,axis2])
scatter!(1, lab="")
for i in 51000:52350#length(samples)
    p[2] = [axis1(i)], [axis2(i)]
    frame(anim)
end
gif(anim)








## In a very high dimensional scenario

p=100

ns, a = Matrix{Float64}(I, p, p), repeat([-Inf];outer=[p])#zeros(p)
geom  = Polygonal(ns, a)
# for a given ray, which boundary does it hit?
nextbd(x, v) = nextboundary(geom, x, v)

sigma1=100
sigma2=4

P1 = diagm(repeat([1/sigma1];outer=[p]))#diag(1/sigma1)#[[1/(sigma1) 0];[0 1/(sigma1)]]
P1 *= P1'
P1 += norm(P1)/(sigma1^2)*Matrix{Float64}(I,p,p)
C1  = inv(P1); C1 += C1'; C1/=2;
L1  = cholesky(C1)
mu = zeros(p)#+[3.,3.]
mvg = MvGaussianStandard(mu,C1)

massmatrix = diagm(repeat([sigma1^2];outer=[p]))#[[sigma1^2 0];[0 sigma1^2]]#Matrix{Float64}(I, p, p)

# Building a BOO Simulation
U(x) = -(sum(x.^2)/20000+cos(pi*x[1]/4)+cos(pi*x[2]/4))+dot(x,x)/2#gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU

#x0 = randn(2)#[4,4]#,0.1,0.1,0.1]
prior = MvNormal(zeros(p),C1)
v0 = Distributions.rand(prior,1)[:,1]
x0 = Distributions.rand(prior,1)[:,1]
T    = 100.0   # length of path generated
lref = 1     # rate of refreshment


gradll(x) = (pi/sigma2)*sin.(pi*x/sigma2)#cos(pi*x[1]/4)+cos(pi*x[2]/4))+x#gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU
nextev_boo(x, v) = nextevent_boo_rough_affine(gradll,mvg, x, v)

# Define a simulation
algname = "BOOMERANG"


ess_vector_boo = Vector{Float64}()
ess_vector_raw_boo = Vector{Float64}()

for i in 1:10
    v0 = Distributions.rand(prior,1)[:,1]
    x0 = Distributions.rand(prior,1)[:,1]

    # Define a simulation
    sim_boo = Simulation( x0, v0, T, nextev_boo, gradll,
                      nextbd, lref,  mass=massmatrix, algname; maxsegments = 1000000, maxgradeval = 20000000)
    (path_boo, details_boo) = simulate(sim_boo)
    samples = samplepath_boo(path_boo,[0.0:0.001:round(path_boo.ts[end]-0.5,digits=0);])
    ess=esspath_boo(path_boo)#samples3)
    ess_sec=mean(ess[1])#/details_mhwithinpdmp3["clocktime"]
    print(ess_sec)
    print("\n")
    push!(ess_vector_raw_boo, ess_sec)
    push!(ess_vector_boo,ess_sec/details_boo["clocktime"])
end


plot(samples[1,:],samples[2,:],
    legend=false,
    size=(600,600),
    #title="Boomerang Sampler (2-dimensional Gaussian target)",
    xlabel = "X1(t)",
    ylabel = "X2(t)",
    #ylims = (-300,300),
    #xlims = (-300,300),
    #fmt = :png,
    color=:black)

ess=esspath_boo(path_boo)#samples3)
ess_sec=mean(ess[1])#/details_mhwithinpdmp3["clocktime"]
print(ess_sec/details_boo["clocktime"])




# BPS
gradll(x) = (pi/sigma2)*sin.(pi*x/sigma2)-x./(sigma1^2)#cos(pi*x[1]/4)+cos(pi*x[2]/4))+x#gradloglik(mvg, x) # = - P1*(x-mu) = - nablaU
nextev_bps(x, v) = nextevent_bps_rough_affine(gradll, x, v)

T    = 100.0   # length of path generated
lref = 1     # rate of refreshment
algname = "BPS"

ess_vector_bps = Vector{Float64}()
ess_vector_raw_bps = Vector{Float64}()

for i in 1:10
    v0 = Distributions.rand(prior,1)[:,1]
    x0 = Distributions.rand(prior,1)[:,1]

    sim_bps = Simulation(  x0, v0, T, nextev_bps, gradll,
                      nextbd, lref,  mass=massmatrix, algname; maxsegments = 1000000, maxgradeval = 2000000)
    (path_bps, details_bps) = simulate(sim_bps)
    ess=esspath_boo(path_bps)#samples3)
    ess_sec=mean(ess[1])#/details_mhwithinpdmp3["clocktime"]
    print(ess_sec)
    print("\n")
    push!(ess_vector_raw_bps, ess_sec)
    push!(ess_vector_bps,ess_sec/details_bps["clocktime"])
end

plot(path_bps.xs[1,:],path_bps.xs[2,:],
    legend=false,
    size=(600,600),
    #title="Bouncy Particle Sampler (2-dimensional Gaussian target)",
    xlabel = "x1(t)",
    ylabel = "x2(t)",
    color=:black)

ess=esspath_boo(path_bps)#samples3)
ess_sec=mean(ess[1])#/details_mhwithinpdmp3["clocktime"]
print(ess_sec/details_bps["clocktime"])


using StatsPlots: @df, StatsPlots
using DataFrames
df = DataFrame(Algorithm =vcat(repeat(["BOO"];outer=[10]),repeat(["BPS"];outer=[10])),
                ESS_sec= vcat(ess_vector_boo,ess_vector_bps))
@df df boxplot(:Algorithm,:ESS_sec, alpha=0.5,legend=false,
    ylabel = "ESS/sec")
