"""
Array-level equation discretization using stencil matrices.

Instead of computing per-point stencil weights, this approach:
1. Builds sparse stencil matrices for each derivative operator
2. Computes all derivative values at once via matrix-vector multiplication
3. Assembles per-point equations by indexing into the pre-computed vectors

This is mathematically equivalent to the scalar approach but avoids
redundant per-point stencil weight computation.
"""
function PDEBase.discretize_equation!(
        disc_state::PDEBase.EquationState, pde::Equation, interiormap,
        eqvar, bcmap, depvars, s::DiscreteSpace, derivweights, indexmap,
        discretization::MOLFiniteDifference{G, D}
    ) where {G, D <: ArrayDiscretization}

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

Discretize a PDE at a single grid point using pre-computed derivative vectors.
Derivative values are looked up from `deriv_vecs` instead of being computed
per-point via stencil weights.
"""
function discretize_equation_at_point_array(
        II, s, depvars, pde, derivweights, bcmap, eqvar, indexmap,
        boundaryvalfuncs, deriv_vecs
    )
    boundaryrules = mapreduce(f -> f(II), vcat, boundaryvalfuncs, init = [])

    # Build derivative rules from pre-computed derivative vectors
    deriv_rules = generate_array_deriv_rules(
        II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs
    )

    # Variable value and coordinate rules (same as scalar approach)
    val_rules = valmaps(s, eqvar, depvars, II, indexmap)

    rules = vcat(deriv_rules, boundaryrules, val_rules)

    try
        return expand_derivatives(mol_substitute(pde.lhs, rules)) ~ mol_substitute(pde.rhs, rules)
    catch e
        println("Array discretization failed for equation: $pde at index $II.\n")
        println("The following rules were constructed:")
        display(rules)
        rethrow(e)
    end
end

"""
    generate_array_deriv_rules(II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs)

Generate substitution rules for derivative terms using pre-computed derivative vectors.
For centered derivatives: `Differential(x)^d(u) => deriv_vecs[u][Diff(x)^d][i]`
For upwind derivatives: generates IfElse rules based on advection coefficient sign.
"""
function generate_array_deriv_rules(
        II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs
    )
    rules = Pair[]

    for u in depvars
        haskey(deriv_vecs, u) || continue
        u_dvecs = deriv_vecs[u]

        for x in ivs(u, s)
            j = x2i(s, u, x)
            idx = Idx(II, s, u, indexmap)

            # Centered (even order) derivative rules
            for d in derivweights.orders[x]
                diff_op = Differential(x)^d
                if iseven(d) && haskey(u_dvecs, diff_op)
                    dvec = u_dvecs[diff_op]
                    push!(rules, diff_op(u) => dvec[idx[j]])
                end
            end

            # Upwind (odd order) derivative rules
            # For terms like coeff * Dx(u), use IfElse based on sign of coeff
            for d in derivweights.orders[x]
                diff_op = Differential(x)^d
                if isodd(d) && haskey(u_dvecs, diff_op)
                    dvec_fwd, dvec_bwd = u_dvecs[diff_op]
                    # Default: use backward-biased (positive wind) direction
                    push!(rules, diff_op(u) => dvec_bwd[idx[j]])
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
"""
function generate_array_winding_rules(
        II, s, depvars, derivweights, bcmap, indexmap, pde, deriv_vecs
    )
    terms = split_terms(pde, s.x̄)
    wind_rules = Pair[]

    for u in depvars
        haskey(deriv_vecs, u) || continue
        u_dvecs = deriv_vecs[u]

        for x in ivs(u, s)
            j = x2i(s, u, x)
            idx = Idx(II, s, u, indexmap)

            for d in derivweights.orders[x]
                diff_op = Differential(x)^d
                if isodd(d) && haskey(u_dvecs, diff_op)
                    dvec_fwd, dvec_bwd = u_dvecs[diff_op]
                    i_grid = idx[j]

                    # Build rewriting rules to catch multiplication patterns
                    # coeff * Dx(u) → IfElse.ifelse(coeff > 0, coeff * bwd, coeff * fwd)
                    mult_rule = @rule *(~~a, $(diff_op)(u), ~~b) => begin
                        coeff_expr = mol_substitute(
                            *(~a..., ~b...),
                            valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap)
                        )
                        IfElse.ifelse(
                            coeff_expr > 0,
                            coeff_expr * dvec_bwd[i_grid],
                            coeff_expr * dvec_fwd[i_grid]
                        )
                    end

                    div_rule = @rule /(*(~~a, $(diff_op)(u), ~~b), ~c) => begin
                        coeff_expr = mol_substitute(
                            *(~a..., ~b...) / ~c,
                            valmaps(s, u, depvars, Idx(II, s, depvar(u, s), indexmap), indexmap)
                        )
                        IfElse.ifelse(
                            coeff_expr > 0,
                            coeff_expr * dvec_bwd[i_grid],
                            coeff_expr * dvec_fwd[i_grid]
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
