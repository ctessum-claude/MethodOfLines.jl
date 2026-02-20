"""
    build_centered_stencil_matrix(D, gridlen, bs, x)

Build a sparse stencil matrix `L` of size `gridlen × gridlen` such that `L * u`
applies the centered finite difference approximation for the derivative operator `D`
at every grid point. Boundary stencils are used near domain edges; interior stencils
are used elsewhere. Interface (periodic) boundaries are handled via index wrapping.

# Arguments
- `D::DerivativeOperator`: Contains stencil coefficients and boundary coefficients
- `gridlen::Int`: Number of grid points along the dimension
- `bs`: Interface boundary list (from `filter_interfaces`)
- `x`: The independent variable for this dimension
"""
function build_centered_stencil_matrix(
        D::DerivativeOperator{T, N, Wind, DX}, gridlen, bs, x
    ) where {T, N, Wind, DX <: Number}
    haslower, hasupper = haslowerupper(bs, x)
    half = div(D.stencil_length, 2)

    L = spzeros(T, gridlen, gridlen)

    for i in 1:gridlen
        if (i <= D.boundary_point_count) & !haslower
            # Lower boundary one-sided stencil
            weights = D.low_boundary_coefs[i]
            offset = 1 - i
            for k in 1:D.boundary_stencil_length
                col = i + k - 1 + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        elseif (i > gridlen - D.boundary_point_count) & !hasupper
            # Upper boundary one-sided stencil
            weights = D.high_boundary_coefs[gridlen - i + 1]
            offset = gridlen - i
            for k in 1:D.boundary_stencil_length
                col = i + k - D.boundary_stencil_length + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        else
            # Interior centered stencil (with periodic wrapping if needed)
            weights = D.stencil_coefs
            for (k, offset) in enumerate(half_range(D.stencil_length))
                col = i + offset
                if length(bs) > 0
                    # Periodic wrapping
                    col = mod1(col, gridlen)
                end
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        end
    end

    return L
end

"""
    build_centered_stencil_matrix(D, gridlen, bs, x)

Non-uniform grid variant: stencil coefficients vary per grid point.
"""
function build_centered_stencil_matrix(
        D::DerivativeOperator{T, N, Wind, DX}, gridlen, bs, x
    ) where {T, N, Wind, DX <: AbstractVector}
    @assert length(bs) == 0 "Interface boundary conditions are not yet supported for nonuniform dx dimensions."
    half = div(D.stencil_length, 2)

    L = spzeros(T, gridlen, gridlen)

    for i in 1:gridlen
        if i <= D.boundary_point_count
            # Lower boundary one-sided stencil
            weights = D.low_boundary_coefs[i]
            offset = 1 - i
            for k in 1:D.boundary_stencil_length
                col = i + k - 1 + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        elseif i > gridlen - D.boundary_point_count
            # Upper boundary one-sided stencil
            weights = D.high_boundary_coefs[gridlen - i + 1]
            offset = gridlen - i
            for k in 1:D.boundary_stencil_length
                col = i + k - D.boundary_stencil_length + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        else
            # Interior stencil (coefficients vary per point for non-uniform grids)
            weights = D.stencil_coefs[i - D.boundary_point_count]
            for (k, offset) in enumerate(half_range(D.stencil_length))
                col = i + offset
                if 1 <= col <= gridlen
                    L[i, col] += weights[k]
                end
            end
        end
    end

    return L
end

"""
    build_upwind_stencil_matrix(D, gridlen, bs, x, forward::Bool)

Build a sparse stencil matrix for an upwind derivative operator.

The `forward` parameter indicates the stencil direction:
- `forward=true`: forward-biased stencil (range `0:len-1`), boundary at upper end.
  Used with `windmap[1]` (applied when wind is negative, i.e. `ispositive=false`).
- `forward=false`: backward-biased stencil (range `-(len-1):0`), boundary at lower end.
  Used with `windmap[2]` (applied when wind is positive, i.e. `ispositive=true`).

In `CompleteUpwindDifference`, `D.boundary_point_count` stores the high boundary count
(`stencil_length - 1 - offside`) and `D.offside` stores the low boundary count.
"""
function build_upwind_stencil_matrix(
        D::DerivativeOperator{T, N, Wind, DX}, gridlen, bs, x, forward::Bool
    ) where {T, N, Wind, DX <: Number}
    haslower, hasupper = haslowerupper(bs, x)

    # D.offside = low_boundary_point_count
    # D.boundary_point_count = high_boundary_point_count
    low_bpc = D.offside
    high_bpc = D.boundary_point_count

    L = spzeros(T, gridlen, gridlen)

    for i in 1:gridlen
        if forward
            # Forward-biased stencil: interior range is 0:(stencil_length-1)
            # Boundary handling at upper end only
            if (i > gridlen - high_bpc) & !hasupper
                # Upper boundary one-sided stencil
                weights = D.high_boundary_coefs[gridlen - i + 1]
                offset = gridlen - i
                for k in 1:D.boundary_stencil_length
                    col = i + k - D.boundary_stencil_length + offset
                    if 1 <= col <= gridlen
                        L[i, col] += weights[k]
                    end
                end
            else
                # Interior forward stencil
                weights = D.stencil_coefs
                for (k, offset) in enumerate(0:(D.stencil_length - 1))
                    col = i + offset
                    if length(bs) > 0
                        col = mod1(col, gridlen)
                    end
                    if 1 <= col <= gridlen
                        L[i, col] += weights[k]
                    end
                end
            end
        else
            # Backward-biased stencil: interior range is -(stencil_length-1):0
            # Boundary handling at lower end only
            if (i <= low_bpc) & !haslower
                # Lower boundary one-sided stencil
                weights = D.low_boundary_coefs[i]
                offset = 1 - i
                for k in 1:D.boundary_stencil_length
                    col = i + k - 1 + offset
                    if 1 <= col <= gridlen
                        L[i, col] += weights[k]
                    end
                end
            else
                # Interior backward stencil
                weights = D.stencil_coefs
                for (k, offset) in enumerate((-(D.stencil_length - 1)):0)
                    col = i + offset
                    if length(bs) > 0
                        col = mod1(col, gridlen)
                    end
                    if 1 <= col <= gridlen
                        L[i, col] += weights[k]
                    end
                end
            end
        end
    end

    return L
end

"""
    apply_stencil_along_dim(L, u_scalarized, j, ndim)

Apply a stencil matrix `L` along dimension `j` of a multi-dimensional symbolic array.

For 1D: `L * u_vec` (standard matrix-vector product)
For 2D, dim 1: `L * u_mat` (applies to each column)
For 2D, dim 2: `u_mat * transpose(L)` (applies to each row)
For 3D+: reshape to 2D, apply along the target dimension, reshape back.

The general strategy for ndim ≥ 3 is to permute dimension `j` to the front,
reshape into a 2D matrix (first dim = j, second dim = product of all others),
apply `L * reshaped`, and undo the permutation.
"""
function apply_stencil_along_dim(L, u_scalarized, j, ndim)
    if ndim <= 1
        return L * u_scalarized
    elseif ndim == 2
        if j == 1
            return L * u_scalarized
        else  # j == 2
            return u_scalarized * transpose(L)
        end
    else
        # General N-dimensional case:
        # Move dimension j to the front, flatten remaining dims, apply L, unflatten, permute back.
        sz = size(u_scalarized)
        perm = vcat(j, setdiff(1:ndim, j))
        iperm = invperm(perm)

        u_perm = permutedims(u_scalarized, perm)
        nj = sz[j]
        nrest = prod(sz[k] for k in 1:ndim if k != j)
        u_2d = reshape(u_perm, nj, nrest)
        result_2d = L * u_2d
        result_perm = reshape(result_2d, size(L, 1), size(u_perm)[2:end]...)
        return permutedims(result_perm, iperm)
    end
end

"""
    compute_derivative_vectors(stencil_matrices, s, depvars)

Compute the derivative arrays for all dependent variables and derivative orders
by multiplying stencil matrices with scalarized symbolic array variables.

Returns a nested dictionary:
  `deriv_vecs[u][Differential(x)^d]` → array of `Num` (for centered derivatives)
  `deriv_vecs[u][Differential(x)^d]` → `(array, array)` (for upwind: fwd, bwd)

Each element at index `(i,j,...)` contains the symbolic expression for the derivative
at that grid point. The arrays have the same shape as the discretized variable.
"""
function compute_derivative_vectors(stencil_matrices, s, depvars)
    deriv_vecs = Dict()
    for u in depvars
        uop = operation(u)
        haskey(stencil_matrices, uop) || continue
        u_matrices = stencil_matrices[uop]
        u_dvecs = Dict()
        u_dep = depvar(u, s)
        haskey(s.discvars, u_dep) || continue
        u_scalarized_raw = s.discvars[u_dep]
        ndims(u_scalarized_raw) == 0 && continue  # skip ODE-only variables
        u_scalarized = u_scalarized_raw
        ndim = ndims(u, s)

        for (diff_op, mat_with_dim) in u_matrices
            if mat_with_dim isa Tuple && mat_with_dim[1] === :mixed
                # Mixed derivative: (:mixed, Lx, jx, Ly, jy)
                _, Lx, jx, Ly, jy = mat_with_dim
                # Apply Ly along dim jy first, then Lx along dim jx: Lx * (Ly * u) = Dxy(u)
                tmp = apply_stencil_along_dim(Ly, u_scalarized, jy, ndim)
                u_dvecs[diff_op] = apply_stencil_along_dim(Lx, tmp, jx, ndim)
            elseif mat_with_dim isa Tuple && mat_with_dim[1] isa Tuple
                # Upwind: ((L_fwd, L_bwd), j)
                (L_fwd, L_bwd), j = mat_with_dim
                dvec_fwd = apply_stencil_along_dim(L_fwd, u_scalarized, j, ndim)
                dvec_bwd = apply_stencil_along_dim(L_bwd, u_scalarized, j, ndim)
                u_dvecs[diff_op] = (dvec_fwd, dvec_bwd)
            else
                # Centered: (L, j)
                L, j = mat_with_dim
                u_dvecs[diff_op] = apply_stencil_along_dim(L, u_scalarized, j, ndim)
            end
        end
        deriv_vecs[u] = u_dvecs
    end
    return deriv_vecs
end

"""
    build_stencil_matrices(s, depvars, derivweights, bcmap)

Build all stencil matrices for all dependent variables and derivative orders.
Returns a nested dictionary:
  `matrices[uop][Differential(x)^d] => (L, j)` for centered derivatives
  `matrices[uop][Differential(x)^d] => ((L_fwd, L_bwd), j)` for upwind derivatives
  `matrices[uop][Differential(x)*Differential(y)] => (:mixed, Lx, jx, Ly, jy)` for mixed derivatives

where `j` is the spatial dimension index that the stencil operates on.
"""
function build_stencil_matrices(s, depvars, derivweights, bcmap)
    matrices = Dict()

    for u in depvars
        uop = operation(u)
        u_matrices = Dict()

        for x in ivs(u, s)
            gridlen = length(s, x)
            bs = filter_interfaces(bcmap[uop][x])
            j = x2i(s, u, x)

            # Centered (even order) derivatives
            for d in derivweights.orders[x]
                if iseven(d)
                    D_op = derivweights.map[Differential(x)^d]
                    L = build_centered_stencil_matrix(D_op, gridlen, bs, x)
                    u_matrices[Differential(x)^d] = (L, j)
                end
            end

            # Upwind (odd order) derivatives
            for d in derivweights.orders[x]
                if isodd(d) && haskey(derivweights.windmap[1], Differential(x)^d)
                    # windmap[1]: used when ispositive=false → forward-biased stencil
                    D_fwd = derivweights.windmap[1][Differential(x)^d]
                    L_fwd = build_upwind_stencil_matrix(D_fwd, gridlen, bs, x, true)

                    # windmap[2]: used when ispositive=true → backward-biased stencil
                    D_bwd = derivweights.windmap[2][Differential(x)^d]
                    L_bwd = build_upwind_stencil_matrix(D_bwd, gridlen, bs, x, false)

                    u_matrices[Differential(x)^d] = ((L_fwd, L_bwd), j)
                end
            end
        end

        # Mixed derivatives: Dx*Dy(u) for all pairs of spatial variables
        # Uses first-order stencil matrices applied sequentially: Lx along dim x, then Ly along dim y
        spatial_vars = collect(ivs(u, s))
        for (ix, x) in enumerate(spatial_vars)
            for (iy, y) in enumerate(spatial_vars)
                isequal(x, y) && continue
                # Only build for x < y to avoid duplicate pairs (Dxy = Dyx)
                ix >= iy && continue

                mixed_op = Differential(x) * Differential(y)

                # Build first-order centered stencil matrices for x and y
                # Use the centered difference operator for first derivative (map stores Differential(x)^1)
                if haskey(derivweights.map, Differential(x)) && haskey(derivweights.map, Differential(y))
                    D_x = derivweights.map[Differential(x)]
                    D_y = derivweights.map[Differential(y)]
                    gridlen_x = length(s, x)
                    gridlen_y = length(s, y)
                    bs_x = filter_interfaces(bcmap[uop][x])
                    bs_y = filter_interfaces(bcmap[uop][y])
                    jx = x2i(s, u, x)
                    jy = x2i(s, u, y)
                    Lx = build_centered_stencil_matrix(D_x, gridlen_x, bs_x, x)
                    Ly = build_centered_stencil_matrix(D_y, gridlen_y, bs_y, y)
                    u_matrices[mixed_op] = (:mixed, Lx, jx, Ly, jy)
                    # Also store the reverse order for Dy*Dx lookups
                    u_matrices[Differential(y) * Differential(x)] = (:mixed, Ly, jy, Lx, jx)
                end
            end
        end

        matrices[uop] = u_matrices
    end

    return matrices
end
