export
    Path,
    samplepath,
    samplepath_boo,
    quadpathpoly,
    pathmean,
    pathmean_boo,
    pathvar_boo,
    esspath

"""
    AllowedTimeType

Syntactic sugar for union type of Vector{Real} and LinSpace{Real} (types
accepted for the `samplelocalpath` function).
"""
const AllowedTimeType = Union{Vector{<:Real}, AbstractRange{<:Real}}

"""
    Path

Type to store a path simulated via PDSampling. It stores the corners or trajectory and the
times.
"""
mutable struct Path
    xs::Matrix{Real}  # corners/trajectories (dim: nfeatures * nsteps)
    ts::Vector{Real}  # times (dim: nsteps)
    #is_jump::BitArray # if corner or halmitonian trajectory
    # -- implicit
    p::Int      # nfeatures
    nseg::Int   # number of segments
    Path(xs, ts) = new(xs, ts, size(xs, 1), length(ts) - 1)
end

"""
    Segment

Type to store the information relative to a single segment within a path in the
global BPS.
"""
struct Segment
    ta::Real
    tb::Real
    xa::Vector{Real}
    xb::Vector{Real}
    # -- implicit
    tau::Real
    v::Vector{Real}
    function Segment(ta, tb, xa, xb)
        @assert tb > ta "times of subsequent events should be distinct"
        new(ta, tb, xa, xb, tb - ta, (xb - xa) / (tb - ta))
    end
end

"""
    Segment Boomerang

Type to store the information relative to a single segment within a path in the
Boomerang simulation.
"""
struct SegmentBoo
    ta::Real
    tb::Real
    xa::Vector{Real}
    xb::Vector{Real}
    # -- implicit
    tau::Real
    va::Vector{Real}
    function SegmentBoo(ta, tb, xa, xb)
        @assert tb > ta "times of subsequent events should be distinct"
        new(ta, tb, xa, xb, (tb - ta) ,(xb-cos(tb-ta)*xa) / sin(tb - ta))
    end
end



"""
    getsegment(p, j)

Retrieves segment j where j=1 corresponds to the first segment from the initial
time to the first corner.
"""
getsegment(p::Path, j::Int) =
    Segment(p.ts[j], p.ts[j+1], p.xs[:,j], p.xs[:,j+1])

"""
    getsegment_boo(p, j)

Retrieves segment j where j=1 corresponds to the first segment from the initial
time to the first corner and the initial velocity
"""
getsegment_boo(p::Path, j::Int) =
    SegmentBoo(p.ts[j], p.ts[j+1], p.xs[:,j], p.xs[:,j+1])


"""
    samplepath(p, t)

Sample the piecewise linear trajectory defined by the corners `xs` and the
times `ts` at given time `t`. The matrix returned has dimensions `p*length(t)`.
"""
function samplepath(p::Path, t::AllowedTimeType)
    @assert t[1] >= p.ts[1] && t[end] <= p.ts[end]
    samples = zeros(p.p, length(t))
    #
    ti = t[1]
    j = searchsortedlast(p.ts, ti)
    seg = getsegment(p, j)
    i = 1
    while i <= length(t)
        ti = t[i]
        if ti <= seg.tb
            samples[:,i] = seg.xa + (ti - seg.ta) * seg.v
            # increment
            i += 1
        else
            j += searchsortedlast(p.ts[j+1:end], ti)
            # safeguard for out of bound: point to last segment
            j = (j+1 >= length(p.ts)) ? length(p.ts) - 1 : j
            seg = getsegment(p,j)
        end
    end
    samples
end
samplepath(p::Path, t::Real) = samplepath(p, [t])[:, 1]

"""
    samplepath_boo(p, t)

Sample the piecewise hamiltonian trajectory defined by the corners `xs` and the
times `ts` at given time `t`. The matrix returned has dimensions `p*length(t)`.
"""
function samplepath_boo(p::Path, t::AllowedTimeType)
    # Check if the allowed time is within the simulation time frame
    @assert t[1] >= p.ts[1] && t[end] <= p.ts[end]
    samples = zeros(p.p, length(t))
    #
    ti = t[1]
    j = searchsortedlast(p.ts, ti)
# STOPPED HERE
    seg = getsegment_boo(p, j)
    i = 1
    while i <= length(t)
        ti = t[i]
        if ti <= seg.tb
            samples[:,i] = cos(ti-seg.ta)*seg.xa + sin(ti - seg.ta) * seg.va
            # increment
            i += 1
        else
            j += searchsortedlast(p.ts[j+1:end], ti)
            # safeguard for out of bound: point to last segment
            j = (j+1 >= length(p.ts)) ? length(p.ts) - 1 : j
            seg = getsegment_boo(p,j)
        end
    end
#    i = 1
#    while i <= length(t)
#        ti = t[i]
#        if ti <= seg[2]
#            samples[:,i] = cos(ti-seg[1])*seg[3] + sin(ti - seg[1]) * seg[5]
#            # increment
#            i += 1
#        else
#            j += searchsortedlast(p.ts[j+1:end], ti)
#            # safeguard for out of bound: point to last segment
#            j = (j+1 >= length(p.ts)) ? length(p.ts) - 1 : j
#            seg = SegmentBoo(p.ts[j], p.ts[j+1], p.xs[:,j], p.xs[:,j+1])
#        end
#    end
    samples
end
samplepath_boo(p::Path, t::Real) = samplepath(p, [t])[:, 1]


"""
    quadpathpoly(path, poly)

Integrate a polynomial on the elements along a path. So if we are sampling in Rd
and have phi(X)=Poly(xk), then we can integrate the polynomial along the path.
Note that we can't do the same thing if phi(X) mixes 2 or more components.
"""
function quadpathpoly(path::Path, pol::Poly)
    dim = length(path.xs[:,1])
    res = zeros(dim)
    nseg = length(path.ts)-1
    T = path.ts[end]
    for segidx = 1:nseg-1
        # for each segment in the path
        segment = getsegment(path, segidx)
        # get the characteristics
        tau = segment.tau
        xa = segment.xa
        v = segment.v
        # integrate along that segment
        for d = 1:dim
            res[d] += polyint(polyval(pol, Poly([xa[d], v[d]])))(tau)
        end
    end
    # last segment
    lastsegment = getsegment(path, nseg)
    # characteristics
    tau = T - lastsegment.ta
    xa = lastsegment.xa
    v = lastsegment.v
    # integrate along that segment
    for d = 1:dim
        res[d] += polyint(polyval(pol, Poly([xa[d], v[d]])))(tau)
    end
    res/T
end

function quadpathboo_1cm(path::Path)
    dim = length(path.xs[:,1])
    res = zeros(dim)
    nseg = length(path.ts)-1
    T = path.ts[end]
    for segidx = 1:nseg-1
        # for each segment in the path
        segment = getsegment_boo(path, segidx)
        # get the characteristics
        tau = segment.tau
        xa = segment.xa
        va = segment.va
        # integrate along that segment
        for d = 1:dim
#            res[d] += polyint(polyval(pol, Poly([xa[d], v[d]])))(tau)
            res[d] += sin(tau)*xa[d]-cos(tau)*va[d]#+va[d]
        end
    end
    # last segment
    lastsegment = getsegment_boo(path, nseg)
    # characteristics
    tau = T - lastsegment.ta
    xa = lastsegment.xa
    va = lastsegment.va
    # integrate along that segment
    for d = 1:dim
        #res[d] += polyint(polyval(pol, Poly([xa[d], v[d]])))(tau)
        res[d] += sin(tau)*xa[d]-cos(tau)*va[d]#+va[d]
    end
    res/T
end
function quadpathboo_2cm(path::Path)
    dim = length(path.xs[:,1])
    res = zeros(dim)
    nseg = length(path.ts)-1
    T = path.ts[end]
    for segidx = 1:nseg-1
        # for each segment in the path
        segment = getsegment_boo(path, segidx)
        # get the characteristics
        tau = segment.tau
        xa = segment.xa
        va = segment.va
        # integrate along that segment
        for d = 1:dim
#            res[d] += polyint(polyval(pol, Poly([xa[d], v[d]])))(tau)
            res[d] += (0.5+0.25*sin(2*tau))*xa[d]^2+(0.5-0.25*sin(2*tau))*va[d]^2
        end
    end
    # last segment
    lastsegment = getsegment_boo(path, nseg)
    # characteristics
    tau = T - lastsegment.ta
    xa = lastsegment.xa
    va = lastsegment.va
    # integrate along that segment
    for d = 1:dim
        #res[d] += polyint(polyval(pol, Poly([xa[d], v[d]])))(tau)
        res[d] += (0.5+0.25*sin(2*tau))*xa[d]^2+(0.5-0.25*sin(2*tau))*va[d]^2
    end
    res/T
end

pathmean(path::Path) = quadpathpoly(path, Poly([0.0, 1.0]))
pathmean_boo(path::Path) = quadpathboo_1cm(path)
pathvar_boo(path::Path) = quadpathboo_2cm(path)


"""
    esspath(path, ns, rtol, limitfrac)

Increase the number of sample points on the path. The hope is that the ESS for
each dimension divided by the number of samples becomes larger than `limfrac`.
"""
function esspath(path::Path; ns::Int=1000, rtol::Real=0.1,
                 limfrac::Real=0.2, maxns::Int=100000)

    Tp = 0.999 * path.ts[end]
    gg = range(0, stop=Tp, length=ns)

    spath  = samplepath(path, gg)
    old_ess = [ess(spath[i, :]) for i in 1:path.p]
    flag = any(old_ess ./ ns .> limfrac)
    cur_ess = flag ? similar(old_ess) : old_ess

    while flag && (ns < maxns / 2)
        ns *= 2
        gg = range(0, stop=Tp, length=ns)
        spath = samplepath(path, gg)
        cur_ess = [ess(spath[i, :]) for i in 1:path.p]

        # cond1 : check whether the ESS/NS is big enough
        # cond2 : check whether the ESS changed significantly
        flag   = any(cur_ess ./ ns .> limfrac) ||
                 any((abs.(cur_ess - old_ess) ./ old_ess) .> rtol)
        old_ess = cur_ess
    end
    (cur_ess, ns)
end

"""
    ess(v)

Compute the ESS of a vector of samples. This code is adapted from Klara.jl.
"""
function ess(v::AbstractVector{T}) where T <: Real
    length_v = length(v)
    mcvar_iid = var(v) / length_v

    # = IMSE ============================
    maxlag = length_v - 1
    k = Int(floor((maxlag - 1) / 2))
    m = k + 1
    g = zeros(T, m)
    # empirical autocovariance
    acv = autocov(v, 0:maxlag)
    # calculate Ĝ_{n, m} from the acv
    for j ∈ 0:k
        g[j+1] = acv[2j + 1] + acv[2j + 2]
        if g[j+1] <= 0
            m = j
            break
        end
    end
    # create monotone sequence of g
    if m > 1
        for j ∈ 2:m
            if g[j] > g[j - 1]
                g[j] = g[j - 1]
            end
        end
    end
    mcvar_imse = (-acv[1] + 2sum(g[1:m])) / length_v
    # ===================================

    return length_v * mcvar_iid / mcvar_imse
end
