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

    # Generate equations for all interior points
    eqs = if length(interior) == 0
        II = CartesianIndex()
        discretize_equation_at_point_array(
            II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap,
            boundaryvalfuncs, deriv_vecs
        )
    else
        vec(
            map(interior) do II
                discretize_equation_at_point_array(
                    II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap,
                    boundaryvalfuncs, deriv_vecs
                )
            end
        )
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
