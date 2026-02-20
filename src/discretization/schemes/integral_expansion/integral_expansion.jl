# Iterative trapezoidal rule (uniform grid)
function _euler_integral(II, s, jx, u, ufunc, dx::Number)
    j, x = jx
    if II[j] == 1
        return Num(0)
    end
    I1 = unitindex(ndims(u, s), j)
    # Iterative cumulative trapezoid from index 1 to II[j]
    result = Num(0)
    Icur = II - I1 * (II[j] - 1)  # start at index 1
    for k in 2:II[j]
        Iprev = Icur
        Icur = Iprev + I1
        result = result + (dx / 2) * (ufunc(u, [Iprev], x)[1] + ufunc(u, [Icur], x)[1])
    end
    return result
end

# Iterative trapezoidal rule (nonuniform grid)
function _euler_integral(II, s, jx, u, ufunc, dx::AbstractVector)
    j, x = jx
    if II[j] == 1
        return Num(0)
    end
    I1 = unitindex(ndims(u, s), j)
    result = Num(0)
    Icur = II - I1 * (II[j] - 1)  # start at index 1
    for k in 2:II[j]
        Iprev = Icur
        Icur = Iprev + I1
        dxk = dx[k - 1]
        result = result + (dxk / 2) * (ufunc(u, [Iprev], x)[1] + ufunc(u, [Icur], x)[1])
    end
    return result
end

function euler_integral(II, s, jx, u, ufunc)
    j, x = jx
    dx = s.dxs[x]
    return _euler_integral(II, s, jx, u, ufunc, dx)
end

"""
    euler_integral_array(s, jx, u, ufunc)

Compute the cumulative trapezoidal integral along dimension `j` for all grid points
at once, returning a vector of symbolic expressions. This avoids redundant work
compared to calling `euler_integral` per point.
"""
function euler_integral_array(s, jx, u, ufunc)
    j, x = jx
    dx = s.dxs[x]
    n = length(s, x)
    I1 = unitindex(ndims(u, s), j)

    # Build array of u values at each grid point along dimension j
    # For 1D this is straightforward; for multi-D we'd need to handle slices
    result = Vector{Num}(undef, n)
    result[1] = Num(0)
    for k in 2:n
        dxk = dx isa Number ? dx : dx[k - 1]
        # Construct indices for points k-1 and k
        # Use a reference CartesianIndex with just the j-th component varying
        Iprev = CartesianIndex(ntuple(d -> d == j ? k - 1 : 1, ndims(u, s)))
        Icur = CartesianIndex(ntuple(d -> d == j ? k : 1, ndims(u, s)))
        result[k] = result[k - 1] + (dxk / 2) * (
            ufunc(u, [Iprev], x)[1] + ufunc(u, [Icur], x)[1]
        )
    end
    return result
end

# An integral across the whole domain (xmin .. xmax)
function whole_domain_integral(II, s, jx, u, ufunc)
    j, x = jx
    dx = s.dxs[x]
    if II[j] == length(s, x)
        return _euler_integral(II, s, jx, u, ufunc, dx)
    end

    dist2max = length(s, x) - II[j]
    I1 = unitindex(ndims(u, s), j)
    Imax = II + dist2max * I1
    return _euler_integral(Imax, s, jx, u, ufunc, dx)
end

@inline function generate_euler_integration_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms
    )
    ufunc(u, I, x) = s.discvars[u][I]

    eulerrules = reduce(
        safe_vcat,
        [
            [
                    Integral(
                        x in DomainSets.ClosedInterval(
                            s.vars.intervals[x][1],
                            Num(x)
                        )
                    )(u) => euler_integral(
                        Idx(II, s, u, indexmap), s, (x2i(s, u, x), x), u, ufunc
                    )
                    for x in ivs(u, s)
                ]
                for u in depvars
        ],
        init = []
    )
    return eulerrules
end

function wd_integral_Idx(II::CartesianIndex, s::DiscreteSpace, u, x, indexmap)
    # We need to construct a new index as indices may be of different size
    length(ivs(u, s)) == 0 && return CartesianIndex()
    # A hack using the boundary value re-indexing function to get an index that will work
    u_ = mol_substitute(u, [x => s.axies[x][end]])
    II = newindex(u_, II, s, indexmap)
    return II
end

@inline function generate_whole_domain_integration_rules(
        II::CartesianIndex, s::DiscreteSpace, depvars, indexmap, terms, bvar = nothing
    )
    ufunc(u, I, x) = s.discvars[u][I]
    wholedomainrules = reduce(
        safe_vcat,
        [
            [
                    Integral(
                        x in DomainSets.ClosedInterval(
                            s.vars.intervals[x][1],
                            s.vars.intervals[x][2]
                        )
                    )(u) => whole_domain_integral(
                        wd_integral_Idx(II, s, u, x, indexmap), s, (x2i(s, u, x), x), u, ufunc
                    )
                    for x in filter(x -> (!haskey(indexmap, x) | isequal(x, bvar)), ivs(u, s))
                ]
                for u in depvars
        ],
        init = []
    )
    return wholedomainrules
end
