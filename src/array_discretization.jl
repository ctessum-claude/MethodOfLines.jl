"""Cache for passing stencil data from `discretize_equation!` to `discretize`."""
const _FAST_PATH_CACHE = Dict{UInt, Vector{Any}}()

"""
Array-level equation discretization using stencil matrices.

Instead of computing per-point stencil weights, this approach:
1. Builds sparse stencil matrices for each derivative operator
2. Computes all derivative values at once via matrix-vector multiplication
3. Assembles per-point equations by indexing into the pre-computed arrays

This is mathematically equivalent to the scalar approach but avoids
redundant per-point stencil weight computation. For derivative schemes
not covered by stencil matrices (e.g., WENO, nonlinear Laplacian),
falls back to per-point scalar computation.
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
    cache_key = objectid(discretization)
    is_fast_path = length(interior) > 0 && !needs_special_handling(pde, s, depvars, derivweights)
    fast_entry = (
        is_fast = is_fast_path,
        stencil_matrices = is_fast_path ? stencil_matrices : nothing,
        eqvar = eqvar,
        discretespace = s,
    )
    if haskey(_FAST_PATH_CACHE, cache_key)
        push!(_FAST_PATH_CACHE[cache_key], fast_entry)
    else
        _FAST_PATH_CACHE[cache_key] = Any[fast_entry]
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
        # Fallback — keep existing per-point loop for special cases
        vec(
            map(interior) do II
                discretize_equation_at_point_array(
                    II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap,
                    boundaryvalfuncs, deriv_vecs
                )
            end
        )
    else
        # FAST PATH — array-level discretization
        arrayify_pde(pde, s, depvars, deriv_vecs, derivweights, eqvar, indexmap, interiormap)
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

        for x in ivs(u, s)
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
- Nonlinear Laplacian patterns: `Dx(f(u)*Dx(u))`
- Spherical diffusion patterns
- Mixed derivatives: `Dxy(u)`
- `FunctionalScheme` advection (e.g., WENO)
- Integral terms
- Callback rules
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

        # Check for mixed derivatives (two different spatial vars in one derivative chain)
        for t_term in terms
            if _has_mixed_derivatives(t_term, s)
                return true
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
Broadcast an operation over potentially array-valued arguments.
If all arguments are scalar, return scalar. Otherwise broadcast.
"""
function _broadcast_op(op, args)
    any_array = any(a -> a isa AbstractArray, args)
    if !any_array
        # All scalar
        return op(args...)
    end
    # Broadcast
    return broadcast(op, args...)
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
    arrayify_pde(pde, s, depvars, deriv_vecs, derivweights, eqvar, indexmap, interiormap)

Top-level orchestrator for array-level PDE discretization. Converts the entire PDE
into N scalar equations by operating on vectors of symbolic expressions rather than
looping over grid points.

Returns a `Vector{Equation}` of length equal to the number of interior points.
"""
function arrayify_pde(pde, s, depvars, deriv_vecs, derivweights, eqvar, indexmap, interiormap)
    interior = interiormap.I[pde]
    interior_idxs = vec(collect(interior))

    # Build coordinate vectors at interior points (only for the eqvar's spatial vars)
    coord_vecs = Dict{Any, Any}()
    u_dep = depvar(eqvar, s)
    for x in ivs(u_dep, s)
        j = x2i(s, u_dep, x)
        coord_vecs[x] = [Num(s.grid[x][II[j]]) for II in interior_idxs]
    end

    # Identify upwind terms (once, not per-point)
    upwind_replacements = arrayify_upwind_terms(pde, s, depvars, deriv_vecs,
        derivweights, indexmap, interior_idxs, coord_vecs)

    # Split the equation into additive terms on LHS and RHS
    lhs_vec = _arrayify_side(pde.lhs, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, upwind_replacements)
    rhs_vec = _arrayify_side(pde.rhs, s, depvars, deriv_vecs, derivweights,
        indexmap, interior_idxs, coord_vecs, upwind_replacements)

    n = length(interior_idxs)

    # Ensure both sides are vectors
    if !(lhs_vec isa AbstractArray)
        lhs_vec = fill(lhs_vec, n)
    end
    if !(rhs_vec isa AbstractArray)
        rhs_vec = fill(rhs_vec, n)
    end

    # Generate N scalar equations
    return [expand_derivatives(lhs_vec[i]) ~ rhs_vec[i] for i in 1:n]
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
