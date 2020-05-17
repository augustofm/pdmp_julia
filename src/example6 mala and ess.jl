using Distributions

function mala(logdensity,gradient,h,M,niter,θinit)
        function gradientStep(θ,t)
                θ-t*M*gradient(θ)
        end
        θtrace=zeros(length(θinit),niter)#, length(θinit))Array{Float64}(length(θinit),niter)

        θ=θinit
        θtrace[:,1]=θinit
        for i=2:niter
                θold=θ
                θ=rand(MvNormal(gradientStep(θ,0.5*h),h*M))
                d=logdensity(θ) - logdensity(θold) + logpdf(MvNormal(gradientStep(θ,0.5*h),h*M),θold) - logpdf(MvNormal(gradientStep(θold,0.5*h),h*M),θ)
                if(!(log(rand(Uniform(0,1)))<d))
                        θ=θold
                end
                θtrace[:,i]=θ
        end
        θtrace
        end

using Arpack
ρ²=0.8
Σ=[1 ρ²;ρ² 1]

function logdensity(θ)
        logpdf(MvNormal(Σ),θ)
end

function gradient(θ)
        Σ\θ
end

niter=1000
h=1/eigs(inv(Σ),nev=1)[1][1]
draws=mala(logdensity,gradient,h,Matrix{Float64}(I,2,2),niter,[5.,50.])
pdraws=mala(logdensity,gradient,h,Σ,niter,[5,50]);


using PyPlot
function logdensity2d(x,y)
        logdensity([x,y])
end
x = -30:0.1:30
y = -30:0.1:50
X = repmat(x',length(y),1)
Y = repmat(y,1,length(x))
Z = map(logdensity2d,Y,X)
p1 = contour(x,y,Z,200)
plot(vec(draws[1,:]),vec(draws[2,:]))
plot(vec(pdraws[1,:]),vec(pdraws[2,:]))

using RCall
aux = draws'
@rput aux
R"eff = effectiveSize(aux)"
@rget eff

aux2 = pdraws'
@rput aux2
R"eff2 = effectiveSize(aux2)"
@rget eff2


R"min(effectiveSize($(draws’)))"
R> library(coda)

R> min(effectiveSize($(draws’)))




#theme(:ggplot2)

# Define the experiment
n_iter = 10000
n_name = 5
n_chain = 1

# experiment results
#val = rand(500,5, 2)
val = randn(n_iter, n_name, n_chain)# .+ [1, 2, 3]'
val = cat(aux1, dims=3)
val = hcat(val, rand(1:1, n_iter, 1, n_chain))

valaux = cat(aux1, dims=3)
val = valaux
val = cat(val,valaux,dims=3)
val = cat(aux1, dims=3)
val = hcat(val, rand(1:1, 10000, 1, 1))

val = cat(aux1, dims=3)
for id in 2:10
        valaux = cat(aux1, dims=3)
        val = cat(val,valaux, dims=3)
end

using MCMCChains
#using StatsPlots

algname="BPS"
gradll(f,invmatrix) = (Y.+1)./2-(exp.(-f).+1).^(-1)-invmatrix*f#gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)
nextev_mhwithinpdmp(f, theta, v, invmatrix) = nextevent_bps_gpc_affine(gradll, f, theta, v, invmatrix)
ess_vector_bps = Vector{Float64}()

for j in 1:10
        time=0.
        for id in 1:10
                f0 = Distributions.rand(prior,1)[:,1]
                v0 = Distributions.rand(prior,1)[:,1]
                # Define a simulation
                sim_mhwithinpdmp4 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                  nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                  mass=massmatrix, maxsegments=5000)
                (path_mhwithinpdmp4, details_mhwithinpdmp4) = simulate(sim_mhwithinpdmp4)
                time+=details_mhwithinpdmp4["clocktime"]
                #plot(path_mhwithinpdmp4.ys[1,:],path_mhwithinpdmp4.ys[2,:])
                boo_part4=Path(path_mhwithinpdmp4.xs, path_mhwithinpdmp4.ts)
                aux = Matrix(Transpose(samplepath(boo_part4, range(0, stop=0.999 * path_mhwithinpdmp4.ts[end], length=1000))))
                val = cat(aux, dims=3)
                if id==1
                        println("Here!")
                        valtotal=copy(val)
                        global valtotal
                else
                        valtotal=cat(valtotal,val,dims=3)
                        global valtotal
                end
        end
        chnbps = Chains(valtotal)
        ess=mean(summarystats(chnbps)[:ess])
        push!(ess_vector_bps,ess/time)
end


algname="BOOMERANG"
gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
nextev_mhwithinpdmp(f, theta, v) = nextevent_boo_gpc_cte(gradll,f, theta, v)
ess_vector_boo_cte = Vector{Float64}()

for j in 1:10
        time=0.
        for id in 1:10
                f0 = Distributions.rand(prior,1)[:,1]
                v0 = Distributions.rand(prior,1)[:,1]
                sim_mhwithinpdmp1 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                              nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                              Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                              mass=massmatrix, maxsegments=10000)
                (path_mhwithinpdmp1, details_mhwithinpdmp1) = simulate(sim_mhwithinpdmp1)
                time+=details_mhwithinpdmp1["clocktime"]
                #plot(path_mhwithinpdmp2.ys[1,:],path_mhwithinpdmp2.ys[2,:])
                boo_part1=Path(path_mhwithinpdmp1.xs, path_mhwithinpdmp1.ts)
                aux = Matrix(Transpose(samplepath(boo_part1, range(0, stop=0.999 * path_mhwithinpdmp1.ts[end], length=1000))))
                val = cat(aux, dims=3)
                if id==1
                        println("Here!")
                        valtotal=copy(val)
                        global valtotal
                else
                        valtotal=cat(valtotal,val,dims=3)
                        global valtotal
                end
        end
        chnboo2 = Chains(valtotal)
        ess=mean(summarystats(chnboo2)[:ess])
        println(ess)
        println(time)
        push!(ess_vector_boo_cte,ess/time)
end


algname="BOOMERANG"
gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
nextev_mhwithinpdmp(f, theta, v) = nextevent_boo_gpc_affine(gradll,f, theta, v)
ess_vector_boo = Vector{Float64}()

for j in 1:10
        time=0.
        for id in 1:10
                f0 = Distributions.rand(prior,1)[:,1]
                v0 = Distributions.rand(prior,1)[:,1]
                sim_mhwithinpdmp2 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                          nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                          Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                          mass=massmatrix, maxsegments= 5000)
                (path_mhwithinpdmp2, details_mhwithinpdmp2) = simulate(sim_mhwithinpdmp2)
                time+=details_mhwithinpdmp2["clocktime"]
                #plot(path_mhwithinpdmp2.ys[1,:],path_mhwithinpdmp2.ys[2,:])
                boo_part2=Path(path_mhwithinpdmp2.xs, path_mhwithinpdmp2.ts)
                aux = Matrix(Transpose(samplepath(boo_part2, range(0, stop=0.999 * path_mhwithinpdmp2.ts[end], length=1000))))
                val = cat(aux, dims=3)
                if id==1
                        println("Here!")
                        valtotal=copy(val)
                        global valtotal
                else
                        valtotal=cat(valtotal,val,dims=3)
                        global valtotal
                end
        end
        chnboo2 = Chains(valtotal)
        ess=mean(summarystats(chnboo2)[:ess])
        println(ess)
        println(time)
        push!(ess_vector_boo,ess/time)
end

algname="BOOMERANG"
gradll(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
nextev_mhwithinpdmp(f, theta, v) = nextevent_boo_gpc_numeric(lambda!,f, theta, v)

function lambda!(lambdavec::AbstractVector{T}, i::Int, t::T,
        x::Vector{<:Real},y::Vector{<:Real},v::Vector{<:Real}) where T
        lambdavec[i] = max(0,dot(-gradll(cos(t).*x+sin(t).*v,y),-sin(t).*x+cos(t).*v))#/t
        nothing
end;
ess_vector_boo2 = Vector{Float64}()

for j in 1:10
        time=0.
        for id in 1:10
                f0 = Distributions.rand(prior,1)[:,1]
                v0 = Distributions.rand(prior,1)[:,1]
                sim_mhwithinpdmp3 = Simulation(f0, v0, T, nextev_mhwithinpdmp, gradll,
                      nextbd, lref, algname, MHsampler=true, y0=theta0, ytarget=ytarget,
                      Sigmay = Kernelmatrix, actuallambda=lambda!,epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                      mass=massmatrix, maxsegments=5000)
                (path_mhwithinpdmp3, details_mhwithinpdmp3) = simulate(sim_mhwithinpdmp3)
                time+=details_mhwithinpdmp3["clocktime"]
                #plot(path_mhwithinpdmp3.ys[1,:],path_mhwithinpdmp3.ys[2,:])
                boo_part3=Path(path_mhwithinpdmp3.xs, path_mhwithinpdmp3.ts)
                aux = Matrix(Transpose(samplepath_boo(boo_part3, range(0, stop=0.999 * path_mhwithinpdmp3.ts[end], length=1000))))
                val = cat(aux, dims=3)
                if id==1
                        println("Here!")
                        valtotal=copy(val)
                        global valtotal
                else
                        valtotal=cat(valtotal,val,dims=3)
                        global valtotal
                end
        end
        chnboo2 = Chains(valtotal)
        ess=mean(summarystats(chnboo2)[:ess])
        println(ess)
        push!(ess_vector_boo2,ess/time)
end


using StatsPlots
df = DataFrame(Algorithm =vcat(repeat(["BPS"];outer=[10]),
                    repeat(["BOOMERANG \n(Constant bound)"];outer=[10]),
                    repeat(["BOOMERANG  \n(Affine bound)"];outer=[10]),
#                    repeat(["BOOMERANG \n(Combined bound)"];outer=[10])),
                    repeat(["BOOMERANG \n(Combined bound)"];outer=[10])),
                ESS_sec= vcat(ess_vector_bps,
                ess_vector_boo_cte,
                ess_vector_boo,
                ess_vector_boo2))
@df df boxplot(:Algorithm,:ESS_sec, alpha=0.5,legend=false,
    ylabel = "ESS/sec")
    savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/box_gpc_ess_sec")





val2 = cat(aux3, dims=3)
val4 = cat(aux4, dims=3)
val = cat(val1,val2,dims=3)
val = hcat(val, rand(1:2, 10000, 1, n_chain))

chn = Chains(val)
mean(summarystats(chn)[:ess])









val[1:10000,:].=1
val = hcat(val,vcat(repeat([1];outer=[10000]),repeat([2];outer=[10000])))
val = hcat(aux,repeat([1];outer=[10000]))
# construct a Chains object
n_iter = 500
n_name = 3
n_chain = 2
val = randn(n_iter, n_name, n_chain) .+ [1, 2, 3]'
val = hcat(val, rand(1:2, 10000, 1, n_chain))

# construct a Chains object
chn = Chains(val)

chn = Chains(val)

# visualize the MCMC simulation results
p1 = plot(chn)
p2 = plot(chn, colordim = :parameter)
