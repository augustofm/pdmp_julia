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
scatter(x,Y)
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



gradll(f) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x

#ytarget(y) = pdf(MvNormal([4.,2.],Matrix{Float64}(I, 2, 2)), y)

T    = 1000  # length of path generated
lref = 0.1#0.5#0.15#0.1    # rate of refreshment

theta0 = [7., 2.5]
massmatrix = Kernelmatrix(theta0)
#starting point


function Kernelmatrix(theta,p,x)
    K = Matrix{Float64}(I, p, p)
    for i in 1:p
        for j in 1:p
            K[i,j]=kx(x[i],x[j],theta)
        end
    end
    K += norm(K)/100000*Matrix{Float64}(I,p,p)
    K
end

for p in 60:1000
    x1=rand(Normal(-6,0.8), Integer(round(p/2)))
    x2=rand(Normal(-6,0.8), p-length(x1))
    x=vcat(x1, x2)
    massmatrix = Kernelmatrix(theta0,p,x)
#    massmatrix = Matrix{Float64}(I, p, p)
    #x0   = mu+L1.L*randn(p) # sensible s
    prior = MvNormal(zeros(p),massmatrix)
    f0 = Distributions.rand(prior,1)[:,1]
    v0 = Distributions.rand(prior,1)[:,1]
    y1=repeat([-1];outer=[Integer(round(p/2))])
    Y=vcat(y1, repeat([1];outer=[p-length(y1)]))


    a =sqrt.(f0.^2+v0.^2)
    #sum(aux./(exp.(-aux).+1)+((Y.+1)./2).*aux)
    Ind=-(Y.-1)/2
    neg = Ind.*(exp.(a).*a./(exp.(a).+1))
    pos = (-Ind.+1).*(exp.(-a).+1).^(-1).*a
    print(sum(neg.+pos)/p)
    print("\n")
end

algname = "BOOMERANG"
MH = false

## EXAMPLE 1, BOOMERANG GPC CTE BOUND

nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_cte(gradll, f, v)
# Define a simulation
sim_mhwithinpdmp1 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 20000000)
(path_mhwithinpdmp1, details_mhwithinpdmp1) = simulate(sim_mhwithinpdmp1)

boo_part1=Path(path_mhwithinpdmp1.xs, path_mhwithinpdmp1.ts)
samples1=samplepath_boo(boo_part1,[0.0:0.01:round(path_mhwithinpdmp1.ts[end]-1,digits=0);])

plot(samples1[20,:],samples1[55,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Cte Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part1.xs[20,:],boo_part1.xs[55,:],legend=false)

fbar = mean(samples1,dims=2)
scatter(vcat(x0,x1),fbar)
scatter(vcat(x0,x1),(exp.(-fbar).+1).^(-1))

ess = esspath_boo(boo_part1)
mean(ess[1])/details_mhwithinpdmp1["clocktime"]

## EXAMPLE 2, BOOMERANG GPC AFFINE BOUND

nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_affine(gradll, f, v)
# Define a simulation
sim_mhwithinpdmp2 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 200000)
(path_mhwithinpdmp2, details_mhwithinpdmp2) = simulate(sim_mhwithinpdmp2)

boo_part2=Path(path_mhwithinpdmp2.xs, path_mhwithinpdmp2.ts)
samples2=samplepath_boo(boo_part2,[0.0:0.01:round(path_mhwithinpdmp2.ts[end]-1,digits=0);])

plot(samples2[57,:],samples2[60,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part2.xs[57,:],boo_part2.xs[60,:],legend=false)
dot(-((Y.+1.)/2).+1, v0)


MH = true
T    = 10000
lmh　= 0.05
epsilon=0.2
nmh=20

gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x


# Example 3 MH-within-PDMP (Gamma)

u1 = Gamma(2.5,2.5)#Uniform(0, 10)# (maximum(x)-minimum(x))/2)
u2 = Gamma(2.5,2.5)#Uniform(0, 10)#(maximum(x)-minimum(x))/2)
ytarget(x,y) = pdf(MvNormal(zeros(p),Kernelmatrix(y)),x)*pdf(u1,y[1])*pdf(u2,y[2])#pdf(MvNormal([7.,3.],Matrix{Float64}(I, 2, 2)), y)
theta0 = [1., 1.]
nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_affine(gradll, f, v)
# Define a simulation
sim_mhwithinpdmp3 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                  mass=massmatrix, maxgradeval = 2000000)
(path_mhwithinpdmp3, details_mhwithinpdmp3) = simulate(sim_mhwithinpdmp3)

plot(path_mhwithinpdmp3.ys[1,:],path_mhwithinpdmp3.ys[2,:])

boo_part3=Path(path_mhwithinpdmp3.xs, path_mhwithinpdmp3.ts)
samples3=samplepath_boo(boo_part3,[0.0:0.1:round(path_mhwithinpdmp3.ts[end]-1,digits=0);])

plot(samples3[56,:],samples3[60,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part3.xs[56,:],boo_part3.xs[60,:],legend=false)

fbar = mean(samples3,dims=2)
scatter(vcat(x0,x1),fbar)
scatter(vcat(x0,x1),samples3[:,8000])
scatter(vcat(x0,x1),(exp.(-fbar).+1).^(-1))
scatter(x,Y)


theta1bar = mean(path_mhwithinpdmp3.ys[1,:],dims=1)
theta2bar = mean(path_mhwithinpdmp3.ys[2,:],dims=1)

## Example 4 Uniform prior on hyperparameters  (UNIFORM)

u1 = Uniform(0.1, 10)# (maximum(x)-minimum(x))/2)
u2 = Uniform(0.1,10)
ytarget(x,y) = pdf(MvNormal(zeros(p),Kernelmatrix(y)),x)*pdf(u1,y[1])*pdf(u2,y[2])#pdf(MvNormal([7.,3.],Matrix{Float64}(I, 2, 2)), y)
theta0 = [1., 1.]
sim_mhwithinpdmp4 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                  mass=massmatrix, maxgradeval = 2000000)
(path_mhwithinpdmp4, details_mhwithinpdmp4) = simulate(sim_mhwithinpdmp4)

plot(path_mhwithinpdmp4.ys[1,:],path_mhwithinpdmp4.ys[2,:])

boo_part4=Path(path_mhwithinpdmp4.xs, path_mhwithinpdmp4.ts)
samples4=samplepath_boo(boo_part4,[0.0:0.1:round(path_mhwithinpdmp4.ts[end]-1,digits=0);])

fbar = mean(samples4,dims=2)
scatter(vcat(x0,x1),fbar)
scatter(vcat(x0,x1),samples4[:,8000])
scatter(vcat(x0,x1),(exp.(-fbar).+1).^(-1))
scatter(x,Y)

theta1bar = mean(path_mhwithinpdmp4.ys[1,:],dims=1)
theta2bar = mean(path_mhwithinpdmp4.ys[2,:],dims=1)


## True distribution
prob(x) = (1-pdf(Normal(0,0.8),x))*(pdf(Normal(-6,0.8),x)+pdf(Normal(2,0.8),x))
xpred = [0.0:0.01:round(path_mhwithinpdmp2.ts[end]-1,digits=0);]
scatter(x,prob)


## 2-step Bayesian Inference

# Marginal distribution
marginaldist = Vector{Float64}()
theta_vector = unique(path_mhwithinpdmp3.ys,dims=2)

theta_vector = reshape(theta_vector, (2,Int(length(theta_vector)/2)))

for yaux in theta_vector[2,:]
    #yaux = reshape(yaux[:,1],(2,1))
    idx = findall(x->x==yaux, path_mhwithinpdmp3.ys[2,:])
    if length(idx)==1
        subset_sample=path_mhwithinpdmp3.xs[:,idx]
    else
        subset = Path(path_mhwithinpdmp3.xs[:,idx], path_mhwithinpdmp3.ts[idx])
        subset_sample = samplepath_boo(subset,[subset.ts[1]:0.02:subset.ts[end];])
    end
    N = size(subset_sample)[2]
    marginals= logistic.(subset_sample.*Y)
    prob=0
    for aux in 1:N
        prob+=prod(marginals[:,aux])
    end
    push!(marginaldist, prob/N)
end

#using Plots; plot()
#plot(size=(600,600))
scatter(theta_vector[1,:],theta_vector[2,:],size=(600,600))
scatter(theta_vector[1,:],marginaldist,size=(600,600))
scatter(theta_vector[2,:],marginaldist,size=(600,600))
histogram(theta_vector[1,:])
histogram(theta_vector[2,:])

marginaldist = Vector{Float64}()
theta_vector = unique(path_mhwithinpdmp3.ys)

for yaux in theta_vector
    idx = findall(x->x==yaux, path_mhwithinpdmp3.ys)
    if length(idx)==1
        subset_sample=path_mhwithinpdmp3.xs[:,idx]
    else
        subset = Path(path_mhwithinpdmp3.xs[:,idx], path_mhwithinpdmp3.ts[idx])
        subset_sample = samplepath_boo(subset,[subset.ts[1]:0.02:subset.ts[end];])
    end
    N = size(subset_sample)[2]
    marginals= logistic.(subset_sample.*y)
    prob=0
    for aux in 1:N
        prob+=prod(marginals[:,aux])
    end
    push!(marginaldist, prob/N)
end

plot(theta_vector,marginaldist)

## Laplace Approximation

using RDatasets
using GaussianProcesses
import Distributions:Normal
using Random

Random.seed!(113355)

crabs = dataset("MASS","crabs");              # load the data
crabs = crabs[shuffle(1:size(crabs)[1]), :];  # shuffle the data

train = crabs[1:div(end,2),:];

y = Array{Bool}(undef,size(train)[1]);       # response
y[train[:,:Sp].=="B"].=0;                      # convert characters to booleans
y[train[:,:Sp].=="O"].=1;

X = convert(Matrix,train[:,4:end]);

x0=rand(Normal(0,0.8), 30)
x1=vcat(rand(Normal(-6,0.8),20), rand(Normal(2,0.8),10))
X=vcat(x0,x1)
y=vcat(repeat([0.];outer=[30]),repeat([1.];outer=[30]))
y=Array{Bool}(y)


mZero = MeanZero();                # Zero mean function
Matern()
#kern = SqExponentialKernel Matern(3/2,zeros(5),0.0)
kern = SEIso(2., 2.)#SqExponentialKernel(1.0)
#KernelFunctions.SqExponentialKernel(1.0) #Matern(3/2,zeros(5),0.0);   # Matern 3/2 ARD kernel (note that hyperparameters are on the log scale)
lik = BernLik();


gp = GaussianProcesses.GP(X',y,mZero,kern,lik)

set_priors!(gp.kernel,[Normal(1.0,8.0) for i in 1:2])

aux = Vector()
aux = mcmc(gp; nIter=1000, burn=1000, thin=10)#Array{Float64,2}()
samples = Vector()#aux[61]
for i in 1:10
    samples_aux = mcmc(gp; nIter=1000, burn=1000, thin=10)
    print(samples_aux[61])
    push!(samples,samples_aux[61])
end
samples = mcmc(gp; nIter=1000, burn=1000, thin=10)



# ----
using Stheno
using KernelFunctions

# Choose the length-scale and variance of the process.
l = 2.7
σ² = 7.1

# Construct a kernel with this variance and length scale.
k = σ² * (SqExponentialKernel())

# Specify a zero-mean GP with this kernel. Don't worry about the GPC object.
f = GP(k, GPC())

using DelimitedFiles
using AugmentedGaussianProcesses
using KernelFunctions;
using Distributions
using AugmentedGaussianProcesses
using KernelFunctions

@info "Running full model"



X = readdlm("/Users/gusfmagalhaes/.julia/packages/AugmentedGaussianProcesses/FfXUz/examples/data/banana_X_train")
Y2 = readdlm("/Users/gusfmagalhaes/.julia/packages/AugmentedGaussianProcesses/FfXUz/examples/data/banana_Y_train")

Ms = [4, 8, 16, 32]
models = Vector{AbstractGP}(undef,length(Ms)+1)
kernel = KernelFunctions.SqExponentialKernel(1.0)

for (index, num_inducing) in enumerate(Ms)
    @info "Training with $(num_inducing) points"
    m = SVGP(X, Y2, kernel,LogisticLikelihood(),AnalyticVI(),num_inducing)
    @time train!(m,20)
    models[index]=m;
end


N = 1000
X = reshape((sort(rand(N)).-0.5).*40.0,N,1)
function latent(x)
    5.0.*sin.(x)./x
end
Y = (latent(X)+randn(N))[:];
scatter(X,Y,lab="")
Ms = [4, 8, 16, 32, 64]
models = Vector{AbstractGP}(undef,length(Ms)+1)
kernel = SqExponentialKernel(1.0)
for (index, num_inducing) in enumerate(Ms)
    @info "Training with $(num_inducing) points"
    m = SVGP(X, vec(Y), kernel,GaussianLikelihood(),AnalyticVI(),num_inducing)
    @time train!(m,100)
    models[index]=m;
end


kernel = SqExponentialKernel(1.0)
mfull = VGP(X,Y2, kernel,LogisticLikelihood(),AnalyticVI())
@time train!(mfull,5);
models[end] = mfull;


# Marginal distribution
marginaldist = Vector{Float64}()
theta_vector = unique(path_mhwithinpdmp3.ys,dims=2)

theta_vector = reshape(theta_vector, (2,Int(length(theta_vector)/2)))

for yaux in theta_vector[2,:]
    #yaux = reshape(yaux[:,1],(2,1))
    idx = findall(x->x==yaux, path_mhwithinpdmp3.ys[2,:])
    if length(idx)==1
        subset_sample=path_mhwithinpdmp3.xs[:,idx]
    else
        subset = Path(path_mhwithinpdmp3.xs[:,idx], path_mhwithinpdmp3.ts[idx])
        subset_sample = samplepath_boo(subset,[subset.ts[1]:0.02:subset.ts[end];])
    end
    N = size(subset_sample)[2]
    marginals= logistic.(subset_sample.*Y)
    prob=0
    for aux in 1:N
        prob+=prod(marginals[:,aux])
    end
    push!(marginaldist, prob/N)
end

#using Plots; plot()
#plot(size=(600,600))
scatter(theta_vector[1,:],theta_vector[2,:],size=(600,600))
scatter(theta_vector[1,:],marginaldist,size=(600,600))
scatter(theta_vector[2,:],marginaldist,size=(600,600))

using3D()
n = Int(length(theta_vector)/2)
x = theta_vector[1,:]
y = theta_vector[2,:]

xgrid = repeat(x',n,1)
ygrid = repeat(y,1,n)

z = zeros(n,n)

for i in 1:n
    for j in 1:n
        if i==j
            z[i:i,j:j] .= marginaldist[i]
        else
            z[i:i,j:j] .= NaN
        end
    end
end

fig = figure("pyplot_surfaceplot",figsize=(10,10))
ax = fig.add_subplot(2,1,1,projection="3d")
plot_surface(x,y,z, rstride=2,edgecolors="k", cstride=2, cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)

xlabel("sigma")
ylabel("l")
PyPlot.title("Surface Plot")

py, bestindex = findmax(marginaldist)
theta_vector[:,bestindex]

using Plots; pyplot()
plot_surface(theta_vector[1,:],theta_vector[2,:],marginaldist,rstride=2,edgecolors="k", cstride=2,
   cmap=ColorMap("gray"), alpha=0.8, linewidth=0.25)

# Example 3 Factorized Boomerang Sampler
nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_factor(gradll, f, v)
sim_mhwithinpdmp = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 20000)
(path_mhwithinpdmp, details_mhwithinpdmp) = simulate(sim_mhwithinpdmp)

boo_part=Path(path_mhwithinpdmp.xs, path_mhwithinpdmp.ts)
samples=samplepath_boo(boo_part,[0.0:0.01:round(path_mhwithinpdmp.ts[end]-1,digits=0);])

plot(samples[40,:],samples[57,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (61-dimensional GPC)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part.xs[40,:],boo_part.xs[57,:],legend=false)



scatter(x,f0)

scatter(x,samples[:,340])
    scatter(x,samples[:,3140])
#    scatter(x,samples[:,13340])
    #scatter(vcat(x0,x1),samples[:,18040])

fbar = mean(samples,dims=2)
scatter(vcat(x0,x1),fbar)
scatter(vcat(x0,x1),(exp.(-fbar).+1).^(-1))


## 2-step Bayesian Inference
#marginaldist = Vector{Float64}()
#theta_vector = unique(path_mhwithinpdmp.ys)

for yaux in theta_vector
    idx = findall(x->x==yaux, path_mhwithinpdmp3.ys)
    if length(idx)==1
        subset_sample=path_mhwithinpdmp3.xs[:,idx]
    else
        subset = Path(path_mhwithinpdmp3.xs[:,idx], path_mhwithinpdmp3.ts[idx])
        subset_sample = samplepath_boo(subset,[subset.ts[1]:0.02:subset.ts[end];])
    end
    N = size(subset_sample)[2]
    marginals= logistic.(subset_sample.*y)
    prob=0
    for aux in 1:N
        prob+=prod(marginals[:,aux])
    end
    push!(marginaldist, prob/N)
end

plot(theta_vector,marginaldist)


#MHsampler=false, y0=f0, ytarget=ytarget,
#epsilon=epsilon, nmh=nmh,lambdamh=lmh,




## EXAMPLE 1 Banana Distribution
using PDSampler
using LinearAlgebra         #For Cholesky
import Random               #For generating random vectors               #For plotting
using Plots
using DelimitedFiles
using Distributions


theta0 = [0.75 0.5]
p=length(theta0)
# Boundaries
ns, a = Matrix{Float64}(I, p, p), repeat([-Inf];outer=[p])#zeros(p)

geom = Polygonal(ns,a)
nextbd(x,v) = nextboundary(geom, x, v)

y = rand(Normal(theta0[1]+theta0[2]^2,1), 15)
prior1 = Normal(0,0.5)
prior2 = Normal(0,0.5)

C = [[1/4 0];[0 1/4]]
P = inv(C)
massmatrix = P#Matrix{Float64}([1./4 0 0 1./4], p, p)

gradll(theta) = [sum(y.-theta[1].-theta[2]^2), 2*sum(theta[2].*(y.-theta[1].-theta[2]))]

T    = 5000  # length of path generated
lref = .7#0.5#0.15#0.1    # rate of refreshment

#x0   = mu+L1.L*randn(p) # sensible starting point
prior = MvNormal(zeros(p),massmatrix)
theta0 = Distributions.rand(prior,1)[:,1]
v0 = Distributions.rand(prior,1)[:,1]

algname = "BOOMERANG"

nextev_mhwithinpdmp(f, v) = nextevent_boo(gradll, MvGaussianStandard(zeros(p),Matrix{Float64}(I, p, p)), f, v)
# Define a simulation
sim_mhwithinpdmp1 = Simulation(theta0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 200000)
(path_mhwithinpdmp1, details_mhwithinpdmp1) = simulate(sim_mhwithinpdmp1)

boo_part1=Path(path_mhwithinpdmp1.xs, path_mhwithinpdmp1.ts)
samples1=samplepath_boo(boo_part1,[0.0:0.01:round(path_mhwithinpdmp1.ts[end]-1,digits=0);])

plot(samples1[1,:],samples1[2,:],
    legend=false,
    size=(600,600),
    title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
    xlabel = "X_1(t)",
    ylabel = "X_2(t)",
    ylims = (-3.5,4),
    #fmt = :png,
    color=:black)
    scatter!(boo_part1.xs[1,:],boo_part1.xs[2,:],legend=false)


## Benchmarking MH-Boomerang vs MH-BPS

function nextevent_bps(lb::LinearBound,
                       x::Vector{<:Real}, v::Vector{<:Real})
    a = lb.a(x, v)
    b = lb.b
    @assert a >= 0.0 && b > 0.0 "<ippsampler/nextevent_bps/linearbound>"
    tau = -a / b + sqrt((a / b)^2 + 2randexp() / b)
    lambdabar = a + b * tau
    return NextEvent(tau, dobounce=(g,v)->(rand()<-dot(g, v)/lambdabar))
end

gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)
T    = 1000  # length of path generated
lref = 0.1#0.5#0.15#0.1    # rate of refreshment

theta0 = [7., 2.5]
massmatrix = Kernelmatrix(theta0)

#x0   = mu+L1.L*randn(p) # sensible starting point
prior = MvNormal(zeros(p),massmatrix)
f0 = Distributions.rand(prior,1)[:,1]
v0 = Distributions.rand(prior,1)[:,1]

algname = "BOOMERANG"
MH = false

## EXAMPLE 1, BOOMERANG GPC CTE BOUND

nextev_mhwithinpdmp(f, v) = nextevent_boo_gpc_cte(gradll, f, v)
# Define a simulation
sim_mhwithinpdmp1 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname;
                  mass=massmatrix, maxgradeval = 20000000)
(path_mhwithinpdmp1, details_mhwithinpdmp1) = simulate(sim_mhwithinpdmp1)
