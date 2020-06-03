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


MH = true

using MCMCChains
#using StatsPlots

u1 = Gamma(1,3)
u2 = InverseGamma(2,10)

theta0 = [rand(u1,1)[1], rand(u2,1)[1]]
#theta0 = [7., 2.5]
#theta0 = [10, 5]
#theta0 = [rand(u1,1)[1], rand(u2,1)[1]]
massmatrix = Kernelmatrix(theta0)
prior = MvNormal(zeros(p),massmatrix)

ytarget(x,y) = (y[2]>0.0) ? pdf(MvNormal(zeros(p),Kernelmatrix(y)),x)*pdf(u1,y[1])*pdf(u2,y[2]) : 0


#algname="BPS"
#gradll(f,invmatrix) = (Y.+1)./2-(exp.(-f).+1).^(-1)-invmatrix*f#gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)
#nextev_mhwithinpdmp(f, theta, v, invmatrix) = nextevent_bps_gpc_affine(gradll, f, theta, v, invmatrix)
ess_vector_bps = Vector{Float64}()
ess_vector_boo1 = Vector{Float64}()
ess_vector_boo2 = Vector{Float64}()

ess_vector_bps_min = Vector{Float64}()
ess_vector_boo1_min = Vector{Float64}()
ess_vector_boo2_min = Vector{Float64}()
time_vector_bps = Vector{Float64}()
time_vector_boo1 = Vector{Float64}()
time_vector_boo2 = Vector{Float64}()

rhat_vector_bps = Vector{Float64}()
rhat_vector_boo1 = Vector{Float64}()
rhat_vector_boo2 = Vector{Float64}()

#using JLD
#@save "data.jld"
#save("ess_dataframe.jld",df)

n_chains = 4
n_experiments = 10

lref=0.1#1.5#0.05

epsilon=0.6
nmh=20
#theta0 = [rand(u1,1)[1], rand(u2,1)[1]]
theta0=[7.,2.5]
massmatrix = Kernelmatrix(theta0)
prior = MvNormal(zeros(p),massmatrix)
#f0 = Distributions.rand(prior,1)[:,1]

T=10000
maxseg = 10000000

lmh_vec = [0.1,0.01,0.001,0.]#,0.01,0.]
#N_vec = [200000,200000,100000,100000]
#, 0.01, 0.001, 0.]#0., 0.001, 0.01, 0.1]
#idx=1
#global idx
for lmh in lmh_vec[4]

        println("SCENARIO lmh: ", lmh)

        for j in 1:n_experiments
                #println(idx)
                #theta0 = [rand(u1,1)[1], rand(u2,1)[1]]
                #theta0 = path_mhwithinpdmp2.ysfull[:,j]
                #theta0 = vcat(Distributions.rand(theta_distsigma,1)[:,1], Distributions.rand(theta_distlength,1)[:,1])
                #theta0 = [7, 2.5]
                println("-------------------------")
                println("Experiment ", j)
        #        println(theta0)
                ## Boomerang Affine

                algname1="BOOMERANG"
                println(algname1," Affine")
                gradll1(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
                nextev_mhwithinpdmp1(f, theta, v) = nextevent_boo_gpc_affine(gradll1,f, theta, v)

                runtime=0.
                for id in 1:n_chains
#                        print("Chain : ",id)
                        f0 = Distributions.rand(prior,1)[:,1]
                        v0 = Distributions.rand(prior,1)[:,1]
                        sim_mhwithinpdmp1 = Simulation(f0, v0, T, nextev_mhwithinpdmp1, gradll1,
                                  nextbd, lref, algname1, MHsampler=true, y0=theta0, ytarget=ytarget,
                                  Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                                  mass=massmatrix, blocksize=100000, maxsegments=maxseg)
                        (path_mhwithinpdmp1, details_mhwithinpdmp1) = simulate(sim_mhwithinpdmp1)
                        runtime+=details_mhwithinpdmp1["clocktime"]
                        #println(runtime)#plot(path_mhwithinpdmp2.ys[1,:],path_mhwithinpdmp2.ys[2,:])
                        boo_part1=Path(path_mhwithinpdmp1.xs, path_mhwithinpdmp1.ts)
                        aux = Matrix(Transpose(samplepath(boo_part1, range(0, stop=0.999 * path_mhwithinpdmp1.ts[end], length=50000))))
                        val = cat(aux, dims=3)
                        #valtotal=copy(val)
                        #valtotal=cat(valtotal,val,dims=3)
                        if id==1
                                valtotal=copy(val)
                                global valtotal
                        else
                                valtotal=cat(valtotal,val,dims=3)
                                global valtotal
                        end
                end
                chnboo1 = Chains(valtotal)
                ess=mean(summarystats(chnboo1)[:ess])
                #min=minimum(summarystats(chnboo1)[:ess])
                rhat=maximum(summarystats(chnboo1)[:r_hat])
        #        println(ess)
        #        println(time)
                push!(ess_vector_boo1,ess/runtime)
                push!(time_vector_boo1,runtime)
                #push!(ess_vector_boo1_min,min/runtime)
                push!(rhat_vector_boo1,rhat)
                println("ESS/sec: ",ess/runtime)
                println("r_hat: ",rhat)
                ## Boomerang Numerical
                # Numerical
        #        algname2="BOOMERANG"
        #        println(algname2," Combined")
        #        global lambda!
        #        gradll2(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
        #        function lambda!(lambdavec::AbstractVector{T}, i::Int, t::T,
        #                x::Vector{<:Real},y::Vector{<:Real},v::Vector{<:Real}) where T
        #                lambdavec[i] = max(0,dot(-gradll2(cos(t).*x+sin(t).*v,y),-sin(t).*x+cos(t).*v))#/t
        #                nothing
        #        end;
        #        nextev_mhwithinpdmp2(f, theta, v) = nextevent_boo_gpc_numeric(lambda!,f, theta, v)

        #        runtime=0.
        #        for id in 1:n_chains
        #                #f0 = Distributions.rand(prior,1)[:,1]
        #                v0 = Distributions.rand(prior,1)[:,1]
        #                sim_mhwithinpdmp2 = Simulation(f0, v0, T, nextev_mhwithinpdmp2, gradll2,
        #                      nextbd, lref, algname2, MHsampler=true, y0=theta0, ytarget=ytarget,
        #                      Sigmay = Kernelmatrix, actuallambda=lambda!,epsilon=epsilon, nmh=nmh,lambdamh=lmh;
        #                      mass=massmatrix, maxsegments=maxseg)
        #                (path_mhwithinpdmp2, details_mhwithinpdmp2) = simulate(sim_mhwithinpdmp2)
        #                runtime+=details_mhwithinpdmp2["clocktime"]
                        #plot(path_mhwithinpdmp3.ys[1,:],path_mhwithinpdmp3.ys[2,:])
        #                boo_part2=Path(path_mhwithinpdmp2.xs, path_mhwithinpdmp2.ts)
        #                aux = Matrix(Transpose(samplepath_boo(boo_part2, range(0, stop=0.999 * path_mhwithinpdmp2.ts[end], length=10000))))
        #                val = cat(aux, dims=3)
        #                if id==1
        #                        valtotal=copy(val)
        #                        global valtotal
        #                else
        #                        valtotal=cat(valtotal,val,dims=3)
        #                        global valtotal
        #                end
        #        end
        #        chnboo2 = Chains(valtotal)
        #        ess=mean(summarystats(chnboo2)[:ess])
        #        min=minimum(summarystats(chnboo2)[:ess])
        #        rhat=maximum(summarystats(chnboo2)[:r_hat])
                #        println(ess)
        #        push!(ess_vector_boo2,ess/runtime)
        #        push!(time_vector_boo2,runtime)
        #        push!(ess_vector_boo2_min,min/runtime)
        #        push!(rhat_vector_boo2,rhat)
        #        println("ESS/sec: ",ess/runtime)

                algname3="BPS"
                println(algname3)
                gradll3(f,invmatrix) = (Y.+1)./2-(exp.(-f).+1).^(-1)-invmatrix*f#gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)
                nextev_mhwithinpdmp3(f, theta, v, invmatrix) = nextevent_bps_gpc_affine(gradll3, f, theta, v, invmatrix)

                runtime=0.
                for id in 1:n_chains
                        #println(runtime)
                        f0 = Distributions.rand(prior,1)[:,1]
                        v0 = Distributions.rand(prior,1)[:,1]
                        # Define a simulation
                        sim_mhwithinpdmp3 = Simulation(f0, v0, T, nextev_mhwithinpdmp3, gradll3,
                          nextbd, lref, algname3, MHsampler=true, y0=theta0, ytarget=ytarget,
                          Sigmay = Kernelmatrix, epsilon=epsilon, nmh=nmh,lambdamh=lmh;
                          mass=massmatrix, maxsegments=maxseg)
                        (path_mhwithinpdmp3, details_mhwithinpdmp3) = simulate(sim_mhwithinpdmp3)
                        runtime+=details_mhwithinpdmp3["clocktime"]
                        #plot(path_mhwithinpdmp4.ys[1,:],path_mhwithinpdmp4.ys[2,:])
                        boo_part3=Path(path_mhwithinpdmp3.xs, path_mhwithinpdmp3.ts)
                        aux = Matrix(Transpose(samplepath(boo_part3, range(0, stop=0.999 * path_mhwithinpdmp3.ts[end], length=50000))))
                        val = cat(aux, dims=3)
                        #valtotal=copy(val)
                        #valtotal=cat(valtotal,val,dims=3)
                        if id==1
                                valtotal=copy(val)
                                global valtotal
                        else
                                valtotal=cat(valtotal,val,dims=3)
                                global valtotal
                        end
                end
                chnbps = Chains(valtotal)
                ess=mean(summarystats(chnbps)[:ess])
                #min=minimum(summarystats(chnbps)[:ess])
                rhat=maximum(summarystats(chnbps)[:r_hat])

                push!(ess_vector_bps,ess/runtime)
                push!(time_vector_bps,runtime)
                #push!(ess_vector_bps_min,min/runtime)
                push!(rhat_vector_bps,rhat)

                println("ESS/sec: ",ess/runtime)
                println("r_hat: ",rhat)

                #idx+=1
#                println("Cummulative: bps, boo1, boo2 ", mean(ess_vector_bps)," , ", mean(ess_vector_boo1)," , ", mean(ess_vector_boo2))
#                println("Cummulative min: bps, boo1, boo2 ", mean(ess_vector_bps_min)," , ", mean(ess_vector_boo1_min)," , ", mean(ess_vector_boo2_min))
#                println("min rhat: ", rhat_vector_bps[j]," , ", rhat_vector_boo1[j]," , ", rhat_vector_boo2[j])
                println("Cummulative: boo, bps ", mean(ess_vector_boo1)," , ", mean(ess_vector_bps))
                #println("Cummulative min: boo, bps ", mean(ess_vector_boo1_min)," , ", mean(ess_vector_bps_min))
                #println("min rhat: ", rhat_vector_boo1[idx]," , ", rhat_vector_bps[idx])
        end
end


# Numerical
algname2="BOOMERANG"
println(algname2," Combined")
global lambda!
gradll2(f,theta) = (Y.+1)./2-(exp.(-f).+1).^(-1)#+gradloglik(MvGaussianStandard(zeros(length(f)),Kernelmatrix(theta)),f)+massmatrix*f#inv(massmatrix)*x
function lambda!(lambdavec::AbstractVector{T}, i::Int, t::T,
        x::Vector{<:Real},y::Vector{<:Real},v::Vector{<:Real}) where T
        lambdavec[i] = max(0,dot(-gradll2(cos(t).*x+sin(t).*v,y),-sin(t).*x+cos(t).*v))#/t
        nothing
end;
nextev_mhwithinpdmp2(f, theta, v) = nextevent_boo_gpc_numeric(lambda!,f, theta, v)

runtime=0.
for id in 1:n_chains
        f0 = Distributions.rand(prior,1)[:,1]
        v0 = Distributions.rand(prior,1)[:,1]
        sim_mhwithinpdmp2 = Simulation(f0, v0, T, nextev_mhwithinpdmp2, gradll2,
              nextbd, lref, algname2, MHsampler=true, y0=theta0, ytarget=ytarget,
              Sigmay = Kernelmatrix, actuallambda=lambda!,epsilon=epsilon, nmh=nmh,lambdamh=lmh;
              mass=massmatrix, maxsegments=5000000)
        (path_mhwithinpdmp2, details_mhwithinpdmp2) = simulate(sim_mhwithinpdmp2)
        runtime+=details_mhwithinpdmp2["clocktime"]
        #plot(path_mhwithinpdmp3.ys[1,:],path_mhwithinpdmp3.ys[2,:])
        boo_part2=Path(path_mhwithinpdmp2.xs, path_mhwithinpdmp2.ts)
        aux = Matrix(Transpose(samplepath_boo(boo_part2, range(0, stop=0.999 * path_mhwithinpdmp2.ts[end], length=100000))))
        val = cat(aux, dims=3)
        if id==1
                valtotal=copy(val)
                global valtotal
        else
                valtotal=cat(valtotal,val,dims=3)
                global valtotal
        end
end
chnboo2 = Chains(valtotal)
ess=mean(summarystats(chnboo2)[:ess])
min=minimum(summarystats(chnboo2)[:ess])
#        println(ess)
push!(ess_vector_boo2,ess/runtime)
push!(time_vector_boo2,runtime)
push!(ess_vector_boo2_min,min/runtime)
println("ESS/sec: ",ess/runtime)


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
                aux = Matrix(Transpose(samplepath(boo_part1, range(0, stop=0.999 * path_mhwithinpdmp1.ts[end], length=10000))))
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
#using LatexStrings

exp_vec = vcat(repeat(["0.0"];outer=[10]),
        repeat(["0.001"];outer=[10]),
#        repeat(["0.01"];outer=[10]),
        repeat(["0.01"];outer=[10]),
#        repeat(lmh_vec[4];outer=[10]),
        repeat(["0.1"];outer=[10]))
df = DataFrame(Experiment=repeat(exp_vec;outer=[2]),
                Algorithm =vcat(repeat(["Metropolis-within-BPS"];outer=[40]),
#                    repeat(["BOOMERANG \n(Constant bound)"];outer=[4]),
#                    repeat(["BOOMERANG  \n(Combined bound)"];outer=[40]),
#                    repeat(["BOOMERANG \n(Combined bound)"];outer=[10])),
                    repeat(["Metropolis-within-Boomerang"];outer=[40])),
                ESS_sec= vcat(reverse(log.(ess_vector_bps)),
#                ess_vector_boo_cte,
#                log.(ess_vector_boo2),
                reverse(log.(ess_vector_boo1))))
#Plots.scalefontsizes(1.2)
@df df groupedboxplot(:Experiment,:ESS_sec,outliers=false,alpha=0.5,
        legend=:topright,
        #title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
        xlabel = "lambda MH",
        group=:Algorithm,ylabel = "log ESS/sec")
        #savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/box_gpc_ess_sec_bps_vs_boo1_box_full_mcmc_lref_15_logONLY3")

df = DataFrame(Experiment=repeat(exp_vec;outer=[2]),
                Algorithm =vcat(repeat(["Metropolis-within-BPS"];outer=[30]),
#                    repeat(["BOOMERANG \n(Constant bound)"];outer=[4]),
#                    repeat(["BOOMERANG  \n(Combined bound)"];outer=[40]),
#                    repeat(["BOOMERANG \n(Combined bound)"];outer=[10])),
                    repeat(["Metropolis-within-Boomerang"];outer=[30])),
                ESS_sec= vcat(reverse(ess_vector_bps[vcat(5:24,35:44)]),
#                ess_vector_boo_cte,
#                log.(ess_vector_boo2),
                reverse(ess_vector_boo1[vcat(5:24,35:44)])))
@df df groupedboxplot(:Experiment,:ESS_sec,outliers=false,alpha=0.5,
        #title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
        xlabel = "lambda MH",
        group=:Algorithm,ylabel = "ESS/sec")
        #savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/box_gpc_ess_sec_bps_vs_boo1_box_full_mcmc_lref_15ONLY3")

df = DataFrame(Experiment=repeat(exp_vec;outer=[2]),
                Algorithm =vcat(repeat(["MH-within-BPS"];outer=[30]),
#                    repeat(["BOOMERANG \n(Constant bound)"];outer=[4]),
#                    repeat(["BOOMERANG  \n(Combined bound)"];outer=[40]),
#                    repeat(["BOOMERANG \n(Combined bound)"];outer=[10])),
                    repeat(["Metropolis-within-Boomerang"];outer=[30])),
                ESS_sec= vcat(ess_vector_bps_min[vcat(5:24,35:44)],
#                ess_vector_boo_cte,
#                log.(ess_vector_boo2),
                ess_vector_boo1_min[vcat(5:24,35:44)]))
@df df groupedboxplot(:Experiment,:ESS_sec,outliers=false,alpha=0.5,
        #title="MH-Boomerang Sampler (62-dimensional GPC Affine Bound)",
        xlabel = "lambda MH",
        group=:Algorithm,ylabel = "ESS/sec")
        #savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/box_gpc_ess_sec_bps_vs_boo1_box_full_mcmc_lref_005_min")

writedlm( "ess_vector_bps15lref10000TwoRef.csv",  ess_vector_bps, ',')
writedlm( "ess_vector_boo115lref10000TwoRef.csv",  ess_vector_boo1, ',')
writedlm( "rhat_vector_bps15lref10000TwoRef.csv",  rhat_vector_bps, ',')
writedlm( "rhat_vector_boo115lref10000TwoRef.csv",  rhat_vector_boo1, ',')


idxboo1 = findall(x->x<1.05, rhat_vector_boo1)
idxbps= findall(x->x<1.05, rhat_vector_bps)
idxboo2 = findall(x->x<1.05, rhat_vector_boo2)


plot(1:40,ess_vector_bps[1:40],label="BPS")
    plot!(1:40,ess_vector_boo1[1:40],label="Boo Affine")
 #   plot!(1:50,ess_vector_boo2,label="Boo Combined")
plot(1:40,time_vector_bps[5:44],label="BPS")
        plot!(1:40,time_vector_boo1[5:44],label="Boo Affine")
        #plot!(1:50,time_vector_boo2,label="Boo Combined")
plot(1:40,reverse(rhat_vector_bps), label="BPS")
        plot!(1:40,reverse(rhat_vector_boo1), label="Boo Affine")
        #plot!(1:50,rhat_vector_boo2, label="Boo Combined")
        #savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/gpc_min_rhat_lref_15")


push!(ess_vector_boo2,ess/runtime)
push!(time_vector_boo2,runtime)
push!(ess_vector_boo2_min,min/runtime)
push!(rhat_vector_boo2,rhat)


exp_vec = vcat(repeat(["0.0"];outer=[1]),
        repeat(["0.001"];outer=[1]),
        repeat(["0.01"];outer=[1]),
#        repeat(lmh_vec[4];outer=[10]),
        repeat(["0.1"];outer=[1]))

alg_vec = vcat(repeat(["BPS"];outer=[1]),
        repeat(["BOO1"];outer=[1]),
        repeat(["BOO2"];outer=[1]))
# RAW ESS
df = DataFrame(Experiment=repeat(exp_vec;outer=[30]),
                Algorithm =repeat(alg_vec;outer=[40]),
            ESS_sec= reshape(hcat(ess_vector_bps,ess_vector_boo1,ess_vector_boo2)',(120,1))[:,1])

@df df groupedboxplot(:Experiment,:ESS_sec, outliers=false,
     group=:Algorithm,alpha=0.4)

function rand_outlier(magnitude::Int=3)
 v = rand()
 return rand([-magnitude, magnitude]) * v
end

# Make `ng` groups of `n` points for `nx` x values, with `ol` fraction of outliers
function grouped_df(n::Int, nx::Int, ng::Int, ol::Float64)
 return DataFrame(x=repeat(collect(1:nx), ng*n),
            y=[rand() < ol ? rand_outlier() : rand() for _ in 1:n*ng*nx],
            g=vcat([fill(randstring(), nx*n) for _ in 1:ng]...))
end
# simple case with no groups
@df grouped_df(100, 3, 1, 0.1) groupedboxplot(:x, :y, legend=false)
@df grouped_df(100, 4, 4, 0.1) groupedboxplot(:x, :y, group=:g, legend=false)
dfbps = DataFrame(Experiment = exp_vec,
        Algorithm =repeat(["BPS"];outer=[40]),
            ESS_sec= log.(ess_vector_bps))
dfboo1 = DataFrame(Experiment = exp_vec,
    Algorithm =repeat(["BOO"];outer=[40]),
        ESS_sec= log.(ess_vector_boo1))

@df dfbps boxplot(:Experiment,:ESS_sec, outliers=false,
         alpha=0.4,label="BPS")
@df dfboo1 boxplot!(:Experiment,:ESS_sec, outliers=false,
                alpha=0.4,label="BOO")


@df dfbps violin(:Experiment,:ESS_sec,
         side=:left,alpha=0.4,mode = :uniform,label="BPS")
@df dfboo1 violin!(:Experiment,:ESS_sec,
                side=:right,alpha=0.4,label="BOO")

#savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/box_gpc_runtime_sec_bps_vs_boo1_box_fixed_theta")





plot(1:40,ess_vector_bps_min)
        plot!(1:40,ess_vector_boo1_min)
        plot!(1:40,ess_vector_boo2_min)
        #savefig("/Users/gusfmagalhaes/Documents/dev/PDSampler.jl/plots/gpc_ess_sec_bps_vs_boo1")
plot(1:40,time_vector_bps)
        plot!(1:40,time_vector_boo1)
        plot!(1:40,time_vector_boo2)
plot(1:40,rhat_vector_bps)
        plot!(1:40,rhat_vector_boo1)
        plot!(1:40,rhat_vector_boo2)


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
