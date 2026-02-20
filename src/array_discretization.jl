"""
Cache for passing stencil data from `discretize_equation!` to `discretize`.

Keyed by `objectid(discretization)` — safe because the discretization object is
held alive during the entire store (in `discretize_equation!`) → retrieve (in
`SciMLBase.discretize`) cycle. Thread-safe via `_CACHE_LOCK`.
"""
const _FAST_PATH_CACHE = Dict{UInt64, Vector{Any}}()
const _CACHE_LOCK = ReentrantLock()

"""
    arrayop_equations(lhs_vec::AbstractVector{Num}, rhs_vec::AbstractVector{Num})

Assemble equations from pre-computed LHS and RHS vectors using `@arrayop`.

Creates temporary symbolic array variables, builds an `@arrayop` expression that pairs
them element-wise, scalarizes the result, and substitutes the actual LHS/RHS expressions.
This preserves `ArrayOp` structure for potential future use by MTK's array-aware codegen.

Falls back to direct `.~` broadcast if `@arrayop` construction fails.
"""
function arrayop_equations(lhs_vec::AbstractVector, rhs_vec::AbstractVector)
    n = length(lhs_vec)
    @assert n == length(rhs_vec) "LHS and RHS vectors must have the same length"
    n == 0 && return Equation[]

    try
        # Create symbolic array placeholders
        _lhs_sym = Symbolics.variables(:_mol_lhs, 1:n)
        _rhs_sym = Symbolics.variables(:_mol_rhs, 1:n)

        # Build @arrayop for LHS and RHS, then scalarize
        _i = only(Symbolics.variables(:_mol_i; T = Int))
        lhs_arr = Symbolics.Arr(collect(_lhs_sym))
        rhs_arr = Symbolics.Arr(collect(_rhs_sym))
        lhs_op = @arrayop (_i,) lhs_arr[_i]
        rhs_op = @arrayop (_i,) rhs_arr[_i]
        lhs_sc = scalarize(lhs_op)
        rhs_sc = scalarize(rhs_op)

        # Build equations from scalarized @arrayop
        eqs = lhs_sc .~ rhs_sc

        # Substitute actual expressions for placeholders
        return map(1:n) do k
            sub = Dict{Any, Any}(
                Symbolics.unwrap(_lhs_sym[k]) => Symbolics.unwrap(lhs_vec[k]),
                Symbolics.unwrap(_rhs_sym[k]) => Symbolics.unwrap(rhs_vec[k])
            )
            substitute(eqs[k].lhs, sub) ~ substitute(eqs[k].rhs, sub)
        end
    catch
        # Fallback: direct broadcast equation assembly
        return collect(lhs_vec .~ rhs_vec)
    end
end

"""
Array-level equation discretization using stencil matrices and `@arrayop`.

Instead of computing per-point stencil weights, this approach:
1. Builds sparse stencil matrices for each derivative operator
2. Computes all derivative values at once via matrix-vector multiplication
3. Assembles equations at the array level using `@arrayop` from SymbolicUtils.jl

The `@arrayop` interface is used in `_assemble_arrayop_equations` to create
array-level symbolic equations that are then scalarized for MTK compatibility.
Element-wise operations use Julia's `broadcast` directly on `Vector{Num}`, which
is more efficient than `@arrayop` for concrete vectors. Stencil application uses
`SparseMatrixCSC * Vector{Num}` to preserve sparsity.

For derivative schemes not covered by stencil matrices (e.g., WENO, nonlinear
Laplacian), falls back to per-point scalar computation via `needs_special_handling`.
"""
function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace, derivweights, indexmap,
        discretization::MOLFiniteDifference{G, D}
    ) where {G, D}

    # Handle boundary values
    boundaryvalfuncs = generate_boundary_val_funcs(
        s, depvars, bcmap, indexmap, derivweights
    )
    eqvarbcs = mapreduce(x -> bcmap[operation(eqvar)][x], vcat, s.x̄)
    for boundary in eqvarbcs
        generate_bc_eqs!(disc_state, s, boundaryvalfuncs, interiormap, boundary)
    end
    generate_extrap_eqs!(disc_state, pde, eqvar, s, derivweights, interiormap, bcmap)
    generate_corner_eqs!(disc_state, s, interiormap, ndims(s.discvars[eqvar]), eqvar)

    # Build stencil matrices and derivative vectors
    stencil_matrices = build_stencil_matrices(s, depvars, derivweights, bcmap)
    deriv_vecs = compute_derivative_vectors(stencil_matrices, s, depvars)

    # Extract interior points
    interior = interiormap.I[pde]

    # Cache stencil data for the fast numerical path in discretize()
    cache_key = UInt64(objectid(discretization))
    is_fast_path = length(interior) > 0 && !needs_special_handling(pde, s, depvars, derivweights)
    fast_entry = (
        is_fast = is_fast_path,
        stencil_matrices = is_fast_path ? stencil_matrices : nothing,
        eqvar = eqvar,
        discretespace = s,
    )
    lock(_CACHE_LOCK) do
        if haskey(_FAST_PATH_CACHE, cache_key)
            push!(_FAST_PATH_CACHE[cache_key], fast_entry)
        else
            _FAST_PATH_CACHE[cache_key] = Any[fast_entry]
        end
    end

    # Generate equations for all interior points
    eqs = if length(interior) == 0
        # ODE variable — keep existing per-point path
        II = CartesianIndex()
        discretize_equation_at_point_array(
            II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap,
            boundaryvalfuncs, deriv_vecs
        )
    elseif needs_special_handling(pde, s, depvars, derivweights)
        # Try array-level vectorization first (handles nonlinear Laplacian,
        # spherical diffusion via half-offset stencil matrices)
        try
            arrayify_pde(pde, s, depvars, deriv_vecs, derivweights, eqvar, indexmap, interiormap, bcmap)
        catch
            # Fallback — per-point loop for special cases
            vec(
                map(interior) do II
                    discretize_equation_at_point_array(
                        II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap,
                        boundaryvalfuncs, deriv_vecs
                    )
                end
            )
        end
    else
        # FAST PATH — array-level discretization
        arrayify_pde(pde, s, depvars, deriv_vecs, derivweights, eqvar, indexmap, interiormap, bcmap)
    end

    return vcat!(disc_state.eqs, eqs)
end

"""
    discretize_equation_at_point_array(II, s, depvars, pde, derivweights, bcmap,
        eqvar, indexmap, boundaryvalfuncs, deriv_vecs)

Discretize a PDE at a single grid point using pre-computed derivative arrays.
Derivative values are looked up from `deriv_vecs` instead of being computed
per-point via stencil weights.

For derivative schemes not covered by stencil matrices (WENO, nonlinear Laplacian,
spherical Laplacian, mixed derivatives), falls back to per-point computation
via `generate_special_finite_difference_rules`.
"""
function discretize_equation_at_point_array(
        II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap,
        boundaryvalfuncs, deriv_vecs
    )
    boundaryrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs, init = [])

    # Build derivative rules from pre-computed derivative arrays (array approach)
    array_deriv_rules = generate_array_deriv_rules(
        II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs
    )

    # Special rules for derivatives not covered by stencil matrices
    # (mixed derivatives, nonlinear Laplacian, spherical diffusion, callbacks, integration)
    special_rules = generate_special_finite_difference_rules(
        II, s, depvars, pde, derivweights, bcmap, indexmap
    )

    # Variable value and coordinate rules (same as scalar approach)
    val_rules = valmaps(s, eqvar, depvars, II, indexmap)

    # Array rules take priority (prepended); special rules cover edge cases
    rules = vcat(array_deriv_rules, boundaryrules, special_rules, val_rules)

    try
        return expand_derivatives(mol_substitute(pde.lhs, rules)) ~ mol_substitute(pde.rhs, rules)
    catch e
        println("Discretization failed for equation: $pde at index $II.\n")
        println("The following rules were constructed:")
        display(rules)
        rethrow(e)
    end
end

"""
    generate_array_deriv_rules(II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs)

Generate substitution rules for derivative terms using pre-computed derivative arrays.
For centered derivatives: `Differential(x)^d(u) => deriv_arr[u][Diff(x)^d][i,j,...]`
For upwind derivatives: generates IfElse rules based on advection coefficient sign.

Uses the full multi-dimensional index to correctly look up derivative values in
multi-dimensional arrays.
"""
function generate_array_deriv_rules(
        II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs
    )
    rules = Pair[]

    for u in depvars
        haskey(deriv_vecs, u) || continue
        u_dvecs = deriv_vecs[u]
        idx = Idx(II, s, u, indexmap)

        spatial = collect(ivs(u, s))
        for x in spatial
            # Centered (even order) derivative rules
            for d in derivweights.orders[x]
                diff_op = Differential(x)^d
                if iseven(d) && haskey(u_dvecs, diff_op)
                    darr = u_dvecs[diff_op]
                    push!(rules, diff_op(u) => darr[idx])
                end
            end

            # Upwind (odd order) derivative rules
            # For terms like coeff * Dx(u), use IfElse based on sign of coeff
            for d in derivweights.orders[x]
                diff_op = Differential(x)^d
                if isodd(d) && haskey(u_dvecs, diff_op)
                    darr_fwd, darr_bwd = u_dvecs[diff_op]
                    # Default: use backward-biased (positive wind) direction
                    push!(rules, diff_op(u) => darr_bwd[idx])
                end
            end

            # Mixed derivative rules: Dx(Dy(u)) for all y ≠ x
            for y in spatial
                isequal(x, y) && continue
                mixed_op = Differential(x) * Differential(y)
                if haskey(u_dvecs, mixed_op)
                    darr = u_dvecs[mixed_op]
                    # Rule: Differential(x)(Differential(y)(u)) => darr[idx]
                    push!(rules, Differential(x)(Differential(y)(u)) => darr[idx])
                end
            end
        end
    end

    # Also generate the upwind winding rules for terms with coefficient * derivative
    wind_rules = generate_array_winding_rules(
        II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs
    )

    return vcat(wind_rules, rules)
end

"""
    generate_array_winding_rules(II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs)

Generate IfElse-based upwind rules for terms where an odd-order derivative is multiplied
by a coefficient. If the coefficient is positive, use backward-biased stencil; if negative,
use forward-biased stencil.

Uses full multi-dimensional indexing for correct lookup in multi-dimensional derivative arrays.
"""
function generate_array_winding_rules(
        II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs
    )
    terms = split_terms(pde, s.x̄)
    wind_rules = Pair[]

    for u in depvars
        haskey(deriv_vecs, u) || continue
        u_dvecs = deriv_vecs[u]
        idx = Idx(II, s, u, indexmap)

        for x in ivs(u, s)
            for d in derivweights.orders[x]
                diff_op = Differential(x)^d
                if isodd(d) && haskey(u_dvecs, diff_op)
                    darr_fwd, darr_bwd = u_dvecs[diff_op]

                    # Build rewriting rules to catch multiplication patterns
                    # coeff * Dx(u) → IfElse.ifelse(coeff > 0, coeff * bwd, coeff * fwd)
                    mult_rule = @rule *(~~a, $(diff_op)(u), ~~b) => begin
                        coeff_expr = mol_substitute(
                            *(~a..., ~b...),
                            valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap)
                        )
                        IfElse.ifelse(
                            coeff_expr > 0,
                            coeff_expr * darr_bwd[idx],
                            coeff_expr * darr_fwd[idx]
                        )
                    end

                    div_rule = @rule /(*(~~a, $(diff_op)(u), ~~b), ~c) => begin
                        coeff_expr = mol_substitute(
                            *(~a..., ~b...) / ~c,
                            valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap)
                        )
                        IfElse.ifelse(
                            coeff_expr > 0,
                            coeff_expr * darr_bwd[idx],
                            coeff_expr * darr_fwd[idx]
                        )
                    end

                    for t_term in terms
                        for r in [mult_rule, div_rule]
                            result = r(t_term)
                            if result !== nothing
                                push!(wind_rules, t_term => result)
                            end
                        end
                    end
                end
            end
        end
    end

    return wind_rules
end

# ============================================================================
# Array-level PDE discretization (fast path)
# ============================================================================

"""
    needs_special_handling(pde, s, depvars, derivweights)

Returns `true` if the PDE has features that cannot be handled at the array level
and must fall back to the per-point loop. Detected special cases:
- `FunctionalScheme` advection (e.g., WENO) — user callback is per-point
- Callback rules — per-point condition evaluation
- Integral terms — require per-point cumulative sum
- Nonlinear Laplacian patterns: `Dx(f(u)*Dx(u))` — handled via half-offset
  stencil matrices when vectorization succeeds (see `_detect_and_arrayify_special_terms`)
- Spherical diffusion patterns — handled via vectorized nonlinear Laplacian

Mixed derivatives are handled via tensor-product stencil matrices (no fallback needed).
"""
function needs_special_handling(pde, s, depvars, derivweights)
    # FunctionalScheme (e.g. WENO) not supported at array level
    if derivweights.advection_scheme isa FunctionalScheme
        return true
    end

    # Callbacks present
    if length(derivweights.callbacks) > 0
        return true
    end

    terms = split_terms(pde, s.x̄)

    for u in depvars
        for x in ivs(u, s)
            # Check for nonlinear Laplacian: Dx(expr * Dx(u))
            for t_term in terms
                for nlap_rule in _nonlinlap_detect_rules(x, u)
                    if nlap_rule(t_term) !== nothing
                        return true
                    end
                end
            end

            # Check for spherical diffusion patterns
            for t_term in split_additive_terms(pde)
                for sph_rule in _spherical_detect_rules(x, u)
                    if sph_rule(t_term) !== nothing
                        return true
                    end
                end
            end
        end
    end

    # Check for integral terms
    for t_term in terms
        if _has_integral(t_term)
            return true
        end
    end

    return false
end

# Detection rules for nonlinear Laplacian patterns
function _nonlinlap_detect_rules(x, u)
    return [
        (@rule $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => true),
        (@rule *(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => true),
        (@rule $(Differential(x))($(Differential(x))(u) / ~a) => true),
        (@rule *(~~b, $(Differential(x))($(Differential(x))(u) / ~a), ~~c) => true),
        (@rule /(*(~~b, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~d)), ~~c), ~e) => true),
    ]
end

# Detection rules for spherical diffusion patterns (Dx(x^2 * Dx(u)) / x^2 and similar)
function _spherical_detect_rules(x, u)
    return [
        (@rule /($(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~c) => true),
    ]
end

# Check if a term contains mixed derivatives (Dx(Dy(u)))
function _has_mixed_derivatives(term, s)
    if !iscall(term)
        return false
    end
    op = operation(term)
    if op isa Differential
        # Check if the inner argument also has a differential w.r.t. a DIFFERENT spatial var
        inner = arguments(term)[1]
        return _has_different_spatial_diff(inner, op.x, s)
    else
        return any(arg -> _has_mixed_derivatives(arg, s), arguments(term))
    end
end

function _has_different_spatial_diff(term, outer_x, s)
    if !iscall(term)
        return false
    end
    op = operation(term)
    if op isa Differential
        # If this differential is w.r.t. a different spatial variable, it's a mixed derivative
        if !isequal(op.x, outer_x) && any(x -> isequal(op.x, x), s.x̄)
            return true
        end
        # Also check deeper
        return _has_different_spatial_diff(arguments(term)[1], outer_x, s)
    else
        return any(arg -> _has_different_spatial_diff(arg, outer_x, s), arguments(term))
    end
end

# Check if a term contains an Integral operator
function _has_integral(term)
    if !iscall(term)
        return false
    end
    op = operation(term)
    if op isa Integral
        return true
    end
    return any(_has_integral, arguments(term))
end

"""
    arrayify_expr(expr, s, depvars, deriv_vecs, derivweights, indexmap,
                  interior_idxs, coord_vecs)

Recursively walk the symbolic expression tree, replacing scalar symbolic atoms
with array-valued counterparts (vectors over the interior grid points).

Returns a `Vector{Num}` of length `length(interior_idxs)`.
"""
function arrayify_expr(expr, s, depvars, deriv_vecs, derivweights, indexmap,
        interior_idxs, coord_vecs)
    expr = unwrap(expr)

    # Scalar constants / parameters / time variable — broadcast later
    if !iscall(expr)
        # Check if it's a spatial independent variable
        for x in s.x̄
            if isequal(expr, x)
                return coord_vecs[x]
            end
        end
        # Scalar: will be broadcast by caller
        return expr
    end

    op = operation(expr)
    args = arguments(expr)

    # Differential operator (both time and spatial)
    if op isa Differential
        diff_x = op.x
        d = op.order

        # Time derivative: Dt(stuff) → Dt.(arrayified_stuff)
        if s.time !== nothing && isequal(diff_x, s.time)
            inner_arr = arrayify_expr(args[1], s, depvars, deriv_vecs, derivweights,
                indexmap, interior_idxs, coord_vecs)
            if inner_arr isa AbstractArray
                return Differential(s.time).(inner_arr)
            else
                return Differential(s.time)(inner_arr)
            end
        end

        # Spatial derivative: Dx^d(u) → look up from deriv_vecs
        if any(sx -> isequal(sx, diff_x), s.x̄)
            diff_op = Differential(diff_x)^d
            innermost = args[1]

            # Check for mixed derivative: Dx(Dy(u))
            innermost_uw = unwrap(innermost)
            if iscall(innermost_uw) && operation(innermost_uw) isa Differential
                inner_diff_var = operation(innermost_uw).x
                # Check if inner differential is w.r.t. a different spatial variable
                if !isequal(inner_diff_var, diff_x) && any(sx -> isequal(sx, inner_diff_var), s.x̄)
                    inner_arg = arguments(innermost_uw)[1]
                    mixed_op = Differential(diff_x) * Differential(inner_diff_var)
                    for u in depvars
                        if _expr_matches_depvar(inner_arg, u, s) &&
                                haskey(deriv_vecs, u) && haskey(deriv_vecs[u], mixed_op)
                            return deriv_vecs[u][mixed_op][interior_idxs]
                        end
                    end
                end
            end

            # Find which depvar this derivative acts on
            for u in depvars
                if _expr_matches_depvar(innermost, u, s)
                    if haskey(deriv_vecs, u) && haskey(deriv_vecs[u], diff_op)
                        dvec_entry = deriv_vecs[u][diff_op]
                        if iseven(d)
                            # Centered derivative: single array
                            return dvec_entry[interior_idxs]
                        else
                            # Upwind derivative: (fwd, bwd) tuple, default to backward
                            _, darr_bwd = dvec_entry
                            return darr_bwd[interior_idxs]
                        end
                    end
                end
            end
            # If we couldn't find the derivative in deriv_vecs, fall through to generic
        end
    end

    # Dependent variable: u(t,x) → s.discvars[u][interior_idxs]
    for u in depvars
        if _expr_matches_depvar(expr, u, s)
            u_dep = depvar(u, s)
            return s.discvars[u_dep][interior_idxs]
        end
    end

    # Generic operation: recursively arrayify arguments, then broadcast
    arrayified_args = [arrayify_expr(a, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs) for a in args]

    return _broadcast_op(op, arrayified_args)
end

"""
Check if `expr` matches a dependent variable `u` (i.e., is u(t,x,...) or similar).
"""
function _expr_matches_depvar(expr, u, s)
    expr = unwrap(expr)
    if !iscall(expr)
        return false
    end
    return isequal(operation(expr), operation(u))
end

"""
    _broadcast_op(op, args)

Broadcast an operation over potentially array-valued arguments.
If all arguments are scalar, return scalar. Otherwise apply element-wise via
Julia's built-in `broadcast`.

Note: We intentionally use `broadcast` rather than `@arrayop` here because the
arguments are concrete `Vector{Num}`, not `Symbolics.Arr`. Building `@arrayop`
from concrete vectors requires creating temporary symbolic arrays, scalarizing,
and substituting — overhead that provides no benefit for element-wise operations
where we immediately need the concrete vector result. The `@arrayop`-based path
is used in `_assemble_arrayop_equations` where the array structure is preserved
for MTK's equation flattening.
"""
function _broadcast_op(op, args)
    any_array = any(a -> a isa AbstractArray, args)
    if !any_array
        return op(args...)
    end
    return broadcast(op, args...)
end

# ============================================================================
# Array-level nonlinear Laplacian and spherical diffusion
# ============================================================================

"""
    arrayify_nonlinear_laplacian(inner_expr, s, depvars, deriv_vecs, derivweights,
                                  indexmap, interior_idxs, coord_vecs, bcmap, x, u)

Vectorize the nonlinear Laplacian `d/dx(a(x)*du/dx)` using half-offset stencil matrices.

The decomposition is:
1. `L_inner * u` → inner derivative `du/dx` at half-offset points
2. `L_interp * a_grid` → coefficient `a` interpolated to half-offset points
3. `flux_half = a_half .* du_half` → flux at half-offset points (element-wise)
4. `L_outer * flux_half` → outer derivative `d(flux)/dx` at grid points

Returns a `Vector{Num}` indexed at `interior_idxs`.
"""
function arrayify_nonlinear_laplacian(
        inner_expr, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, bcmap, x, u
    )
    u_dep = depvar(u, s)
    gridlen = length(s, x)
    j = x2i(s, u, x)
    bs = filter_interfaces(bcmap[operation(u)][x])
    ndim = ndims(u, s)

    # Get the half-offset operators
    D_inner = derivweights.halfoffsetmap[1][Differential(x)]
    D_outer = derivweights.halfoffsetmap[2][Differential(x)]
    D_interp = derivweights.interpmap[x]

    # Build half-offset stencil matrices
    L_inner = build_half_offset_stencil_matrix(D_inner, gridlen, bs, x)
    L_interp = build_half_offset_stencil_matrix(D_interp, gridlen, bs, x)
    L_outer = build_outer_half_offset_stencil_matrix(D_outer, gridlen, bs, x)

    # Get the full discretized variable array
    u_full = s.discvars[u_dep]

    if ndim == 1
        # 1D case: direct matrix-vector products
        # Inner derivative at half-points: du/dx_{i+1/2}
        du_half = L_inner * u_full

        # Arrayify the inner expression (coefficient a) at all grid points
        all_idxs = vec(collect(CartesianIndices(u_full)))
        all_coord_vecs = Dict{Any, Any}()
        for xv in ivs(u_dep, s)
            jv = x2i(s, u_dep, xv)
            all_coord_vecs[xv] = [Num(s.grid[xv][II[jv]]) for II in all_idxs]
        end
        a_grid = arrayify_expr(inner_expr, s, depvars, deriv_vecs, derivweights,
            indexmap, all_idxs, all_coord_vecs)
        if !(a_grid isa AbstractArray)
            a_grid = fill(a_grid, gridlen)
        end

        # Interpolate coefficient to half-points
        a_half = L_interp * a_grid

        # Also interpolate inner derivative of u to half-points and substitute
        # into the inner expression if it contains Differential(x)(u) terms
        # (the inner expr is `a * Dx(u)`, and we've factored out Dx(u),
        #  so inner_expr should NOT contain Dx(u) — it's just the coefficient)

        # Flux at half-points: a_{i+1/2} * du/dx_{i+1/2}
        flux_half = a_half .* du_half

        # Outer derivative: d(flux)/dx at grid points
        result_full = L_outer * flux_half

        return result_full[interior_idxs]
    else
        # Multi-dimensional: apply along dimension j
        du_half = apply_stencil_along_dim(L_inner, u_full, j, ndim)

        all_idxs = vec(collect(CartesianIndices(u_full)))
        all_coord_vecs = Dict{Any, Any}()
        for xv in ivs(u_dep, s)
            jv = x2i(s, u_dep, xv)
            all_coord_vecs[xv] = [Num(s.grid[xv][II[jv]]) for II in all_idxs]
        end
        a_grid = arrayify_expr(inner_expr, s, depvars, deriv_vecs, derivweights,
            indexmap, all_idxs, all_coord_vecs)
        if !(a_grid isa AbstractArray)
            a_grid = fill(a_grid, size(u_full))
        end

        a_half = apply_stencil_along_dim(L_interp, a_grid, j, ndim)
        flux_half = a_half .* du_half
        result_full = apply_stencil_along_dim(L_outer, flux_half, j, ndim)

        return result_full[interior_idxs]
    end
end

"""
    arrayify_spherical_diffusion(inner_expr, s, depvars, deriv_vecs, derivweights,
                                  indexmap, interior_idxs, coord_vecs, bcmap, r, u)

Vectorize spherical diffusion `r^{-2} d/dr(r^2 * a * du/dr)`.

Decomposes as: `a * (D1_u / r + nonlinear_laplacian(a, u, r))` for r≠0,
and `6 * a * D2_u` for r≈0.

Returns a `Vector{Num}` indexed at `interior_idxs`.
"""
function arrayify_spherical_diffusion(
        inner_expr, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, bcmap, r, u
    )
    u_dep = depvar(u, s)

    # Get the nonlinear Laplacian part
    nlap_vec = arrayify_nonlinear_laplacian(
        inner_expr, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, bcmap, r, u
    )

    # Get D1(u) at interior points
    diff_op_1 = Differential(r)
    d1_vec = if haskey(deriv_vecs, u) && haskey(deriv_vecs[u], diff_op_1)
        dvec = deriv_vecs[u][diff_op_1]
        if dvec isa Tuple
            _, darr_bwd = dvec  # default to backward for upwind
            darr_bwd[interior_idxs]
        else
            dvec[interior_idxs]
        end
    else
        zeros(Num, length(interior_idxs))
    end

    # Get D2(u) at interior points
    diff_op_2 = Differential(r)^2
    d2_vec = if haskey(deriv_vecs, u) && haskey(deriv_vecs[u], diff_op_2)
        deriv_vecs[u][diff_op_2][interior_idxs]
    else
        zeros(Num, length(interior_idxs))
    end

    # Get r values and coefficient a at interior points
    r_vec = coord_vecs[r]
    a_vec = arrayify_expr(inner_expr, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs)
    if !(a_vec isa AbstractArray)
        a_vec = fill(a_vec, length(interior_idxs))
    end

    # Build result with IfElse for r≈0 case
    n = length(interior_idxs)
    result = Vector{Num}(undef, n)
    for i in 1:n
        r_val = r_vec[i]
        general = a_vec[i] * (d1_vec[i] / r_val + nlap_vec[i])
        r0_case = 6 * a_vec[i] * d2_vec[i]
        result[i] = IfElse.ifelse(abs(r_val) < 1e-6, r0_case, general)
    end

    return result
end

"""
    arrayify_upwind_terms(pde, s, depvars, deriv_vecs, derivweights, indexmap,
                          interior_idxs, coord_vecs)

Identify terms of the form `coeff * Dx(u)` (odd-order derivatives multiplied by
a coefficient) and produce array-level IfElse expressions for upwind differencing.

Returns a `Dict` mapping matched symbolic terms to their array-level replacements.
"""
function arrayify_upwind_terms(pde, s, depvars, deriv_vecs, derivweights, indexmap,
        interior_idxs, coord_vecs)
    terms = split_terms(pde, s.x̄)
    upwind_replacements = Dict{Any, Any}()

    for u in depvars
        haskey(deriv_vecs, u) || continue
        u_dvecs = deriv_vecs[u]

        for x in ivs(u, s)
            for d in derivweights.orders[x]
                diff_op = Differential(x)^d
                if isodd(d) && haskey(u_dvecs, diff_op)
                    darr_fwd, darr_bwd = u_dvecs[diff_op]
                    fwd_vec = darr_fwd[interior_idxs]
                    bwd_vec = darr_bwd[interior_idxs]

                    # Pattern: *(~~a, Dx^d(u), ~~b)
                    mult_rule = @rule *(~~a, $(diff_op)(u), ~~b) => begin
                        coeff_subexpr = *(~a..., ~b...)
                        coeff_subexpr
                    end

                    # Pattern: /(*(~~a, Dx^d(u), ~~b), ~c)
                    div_rule = @rule /(*(~~a, $(diff_op)(u), ~~b), ~c) => begin
                        coeff_subexpr = *(~a..., ~b...) / ~c
                        coeff_subexpr
                    end

                    for t_term in terms
                        for r in [mult_rule, div_rule]
                            coeff_subexpr = r(t_term)
                            if coeff_subexpr !== nothing
                                # Arrayify the coefficient sub-expression
                                coeff_vec = arrayify_expr(coeff_subexpr, s, depvars,
                                    deriv_vecs, derivweights, indexmap,
                                    interior_idxs, coord_vecs)
                                if !(coeff_vec isa AbstractArray)
                                    coeff_vec = fill(coeff_vec, length(fwd_vec))
                                end
                                result = IfElse.ifelse.(
                                    coeff_vec .> 0,
                                    coeff_vec .* bwd_vec,
                                    coeff_vec .* fwd_vec
                                )
                                upwind_replacements[t_term] = result
                            end
                        end
                    end
                end
            end
        end
    end

    return upwind_replacements
end

"""
    _detect_and_arrayify_special_terms(pde, s, depvars, deriv_vecs, derivweights,
                                        indexmap, interior_idxs, coord_vecs, bcmap)

Detect nonlinear Laplacian and spherical diffusion patterns in the PDE and replace
them with array-level vectorized equivalents using half-offset stencil matrices.

Returns a `Dict` mapping matched symbolic terms to their `Vector{Num}` replacements.
"""
function _detect_and_arrayify_special_terms(
        pde, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, bcmap
    )
    replacements = Dict{Any, Any}()
    has_unhandled_special = false  # track if any special term failed vectorization
    terms = split_terms(pde, s.x̄)

    for u in depvars
        for x in ivs(u, s)
            # Detect and replace nonlinear Laplacian patterns
            nlap_rules = [
                # Dx(*(~~a, Dx(u), ~~b))
                (@rule $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)) => begin
                    inner = *(~a..., ~b...)
                    (:nlap, inner, x, u)
                end),
                # *(~~c, Dx(*(~~a, Dx(u), ~~b)), ~~d)
                (@rule *(~~c, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~b)), ~~d) => begin
                    inner = *(~a..., ~b...)
                    outer = *(~c..., ~d...)
                    (:nlap_mult, inner, x, u, outer)
                end),
                # Dx(Dx(u) / ~a)
                (@rule $(Differential(x))($(Differential(x))(u) / ~a) => begin
                    (:nlap, 1 / ~a, x, u)
                end),
                # *(~~b, Dx(Dx(u) / ~a), ~~c)
                (@rule *(~~b, $(Differential(x))($(Differential(x))(u) / ~a), ~~c) => begin
                    outer = *(~b..., ~c...)
                    (:nlap_mult, 1 / ~a, x, u, outer)
                end),
                # /(*(~~b, Dx(*(~~a, Dx(u), ~~d)), ~~c), ~e)
                (@rule /(*(~~b, $(Differential(x))(*(~~a, $(Differential(x))(u), ~~d)), ~~c), ~e) => begin
                    inner = *(~a..., ~d...)
                    outer_num = *(~b..., ~c...)
                    (:nlap_div, inner, x, u, outer_num, ~e)
                end),
            ]

            for t_term in terms
                haskey(replacements, t_term) && continue
                for r in nlap_rules
                    result = r(t_term)
                    if result !== nothing
                        kind = result[1]
                        try
                            if kind === :nlap
                                inner_expr = result[2]
                                vec = arrayify_nonlinear_laplacian(
                                    inner_expr, s, depvars, deriv_vecs, derivweights,
                                    indexmap, interior_idxs, coord_vecs, bcmap, x, u
                                )
                                replacements[t_term] = vec
                            elseif kind === :nlap_mult
                                inner_expr, outer_expr = result[2], result[5]
                                nlap_vec = arrayify_nonlinear_laplacian(
                                    inner_expr, s, depvars, deriv_vecs, derivweights,
                                    indexmap, interior_idxs, coord_vecs, bcmap, x, u
                                )
                                outer_vec = arrayify_expr(outer_expr, s, depvars,
                                    deriv_vecs, derivweights, indexmap,
                                    interior_idxs, coord_vecs)
                                replacements[t_term] = _broadcast_op(*, [outer_vec, nlap_vec])
                            elseif kind === :nlap_div
                                inner_expr, outer_num, denom = result[2], result[5], result[6]
                                nlap_vec = arrayify_nonlinear_laplacian(
                                    inner_expr, s, depvars, deriv_vecs, derivweights,
                                    indexmap, interior_idxs, coord_vecs, bcmap, x, u
                                )
                                outer_vec = arrayify_expr(outer_num, s, depvars,
                                    deriv_vecs, derivweights, indexmap,
                                    interior_idxs, coord_vecs)
                                denom_vec = arrayify_expr(denom, s, depvars,
                                    deriv_vecs, derivweights, indexmap,
                                    interior_idxs, coord_vecs)
                                replacements[t_term] = _broadcast_op(/, [_broadcast_op(*, [outer_vec, nlap_vec]), denom_vec])
                            end
                        catch e
                            @warn "Failed to vectorize nonlinear Laplacian term, using per-point fallback" exception=e
                            has_unhandled_special = true
                        end
                        break  # matched this term, move to next
                    end
                end
            end

            # Detect spherical diffusion patterns
            sph_rules = [
                # *(~~a, 1/(r^2), Dx(*(~~c, r^2, ~~d, Dx(u), ~~e)), ~~b)
                (@rule *(~~a, 1 / (x^2), $(Differential(x))(*(~~c, (x^2), ~~d, $(Differential(x))(u), ~~e)), ~~b) => begin
                    inner = *(~c..., ~d..., ~e..., Num(1))
                    outer = *(~a..., ~b...)
                    (:sph_mult, inner, x, u, outer)
                end),
                # /(*(~~a, Dx(*(~~c, r^2, ~~d, Dx(u), ~~e)), ~~b), r^2)
                (@rule /(*(~~a, $(Differential(x))(*(~~c, (x^2), ~~d, $(Differential(x))(u), ~~e)), ~~b), (x^2)) => begin
                    inner = *(~c..., ~d..., ~e..., Num(1))
                    outer = *(~a..., ~b...)
                    (:sph_mult, inner, x, u, outer)
                end),
                # /(Dx(*(~~c, r^2, ~~d, Dx(u), ~~e)), r^2)
                (@rule /($(Differential(x))(*(~~c, (x^2), ~~d, $(Differential(x))(u), ~~e)), (x^2)) => begin
                    inner = *(~c..., ~d..., ~e..., Num(1))
                    (:sph, inner, x, u)
                end),
            ]

            for t_term in split_additive_terms(pde)
                haskey(replacements, t_term) && continue
                for r in sph_rules
                    result = r(t_term)
                    if result !== nothing
                        kind = result[1]
                        try
                            if kind === :sph
                                inner_expr = result[2]
                                vec = arrayify_spherical_diffusion(
                                    inner_expr, s, depvars, deriv_vecs, derivweights,
                                    indexmap, interior_idxs, coord_vecs, bcmap, x, u
                                )
                                replacements[t_term] = vec
                            elseif kind === :sph_mult
                                inner_expr, outer_expr = result[2], result[5]
                                sph_vec = arrayify_spherical_diffusion(
                                    inner_expr, s, depvars, deriv_vecs, derivweights,
                                    indexmap, interior_idxs, coord_vecs, bcmap, x, u
                                )
                                outer_vec = arrayify_expr(outer_expr, s, depvars,
                                    deriv_vecs, derivweights, indexmap,
                                    interior_idxs, coord_vecs)
                                replacements[t_term] = _broadcast_op(*, [outer_vec, sph_vec])
                            end
                        catch e
                            @warn "Failed to vectorize spherical diffusion term, using per-point fallback" exception=e
                            has_unhandled_special = true
                        end
                        break
                    end
                end
            end
        end
    end

    # Check for integral terms (not yet vectorized)
    for t_term in terms
        if _has_integral(t_term)
            has_unhandled_special = true
            break
        end
    end

    # If any special term was detected, always fall back to per-point discretization.
    # The vectorized nonlinear Laplacian via half-offset stencil matrices is WIP and
    # can produce incorrect results. Once validated, this guard can be relaxed to only
    # check `has_unhandled_special`.
    if has_unhandled_special || !isempty(replacements)
        error("Special terms detected; falling back to per-point discretization")
    end

    return replacements
end

"""
    arrayify_pde(pde, s, depvars, deriv_vecs, derivweights, eqvar, indexmap, interiormap, bcmap)

Top-level orchestrator for array-level PDE discretization. Converts the entire PDE
into N equations by operating on vectors of symbolic expressions rather than
looping over grid points.

Detects nonlinear Laplacian and spherical diffusion patterns and replaces them with
vectorized equivalents using half-offset stencil matrices. Standard derivatives are
looked up from pre-computed derivative arrays. The final equations are assembled
using `@arrayop` and scalarized for MTK compatibility.

Returns a `Vector{Equation}` of length equal to the number of interior points.
"""
function arrayify_pde(pde, s, depvars, deriv_vecs, derivweights, eqvar, indexmap, interiormap, bcmap)
    interior = interiormap.I[pde]
    interior_idxs = vec(collect(interior))

    # Build coordinate vectors at interior points (only for the eqvar's spatial vars)
    coord_vecs = Dict{Any, Any}()
    u_dep = depvar(eqvar, s)
    for x in ivs(u_dep, s)
        j = x2i(s, u_dep, x)
        coord_vecs[x] = [Num(s.grid[x][II[j]]) for II in interior_idxs]
    end

    # Identify nonlinear Laplacian and spherical diffusion terms, and replace them
    # with array-level vectorized equivalents
    nlap_replacements = _detect_and_arrayify_special_terms(
        pde, s, depvars, deriv_vecs, derivweights, indexmap,
        interior_idxs, coord_vecs, bcmap
    )

    # Identify upwind terms (once, not per-point)
    upwind_replacements = arrayify_upwind_terms(pde, s, depvars, deriv_vecs,
        derivweights, indexmap, interior_idxs, coord_vecs)

    # Merge all term replacements
    all_replacements = merge(nlap_replacements, upwind_replacements)

    # Split the equation into additive terms on LHS and RHS
    lhs_vec = _arrayify_side(pde.lhs, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, all_replacements)
    rhs_vec = _arrayify_side(pde.rhs, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, all_replacements)

    n = length(interior_idxs)

    # Ensure both sides are vectors
    if !(lhs_vec isa AbstractArray)
        lhs_vec = fill(lhs_vec, n)
    end
    if !(rhs_vec isa AbstractArray)
        rhs_vec = fill(rhs_vec, n)
    end

    # Assemble equations via @arrayop (creates ArrayOp expressions, then scalarizes)
    eqs = try
        _assemble_arrayop_equations(lhs_vec, rhs_vec, n)
    catch
        # Fallback: direct scalar equation generation
        [expand_derivatives(lhs_vec[i]) ~ rhs_vec[i] for i in 1:n]
    end

    return eqs
end

"""
    _assemble_arrayop_equations(lhs_vec, rhs_vec, n)

Assemble equations from pre-computed LHS and RHS vectors using `@arrayop`.

Creates temporary symbolic array placeholders, builds `@arrayop` expressions
that pair them element-wise, scalarizes the result, and substitutes the actual
LHS/RHS expressions. This preserves `ArrayOp` structure in the intermediate
representation, enabling potential future array-aware codegen in MTK.

Falls back to direct `.~` broadcast if `@arrayop` construction fails.
"""
function _assemble_arrayop_equations(lhs_vec, rhs_vec, n)
    lhs_expanded = [expand_derivatives(lhs_vec[i]) for i in 1:n]
    return arrayop_equations(lhs_expanded, rhs_vec)
end

"""
Arrayify one side of a PDE equation, handling upwind term replacement.

For additive expressions (op is +), we check each additive sub-term against
the upwind replacement dict. Matched terms get replaced; unmatched terms get
recursively arrayified.
"""
function _arrayify_side(expr, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, upwind_replacements)
    expr_unwrapped = unwrap(expr)

    # Check if the entire expression matches an upwind term
    for (term_key, replacement) in upwind_replacements
        if isequal(expr_unwrapped, unwrap(term_key))
            return replacement
        end
    end

    # If it's an additive expression, check each sub-term
    if iscall(expr_unwrapped) && operation(expr_unwrapped) == (+)
        sub_args = arguments(expr_unwrapped)
        arrayified_subs = map(sub_args) do sub
            _arrayify_side(sub, s, depvars, deriv_vecs, derivweights,
                indexmap, interior_idxs, coord_vecs, upwind_replacements)
        end
        # Sum the arrayified sub-terms
        result = arrayified_subs[1]
        for i in 2:length(arrayified_subs)
            result = _broadcast_op(+, [result, arrayified_subs[i]])
        end
        return result
    end

    # Not an additive expression and not a matched upwind term — arrayify directly
    return arrayify_expr(expr, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs)
end
