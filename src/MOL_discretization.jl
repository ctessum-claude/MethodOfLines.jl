# Method of lines discretization scheme

function PDEBase.interface_errors(
        pdesys::PDESystem, v::PDEBase.VariableMap, discretization::MOLFiniteDifference
    )
    depvars = v.ū
    indvars = v.x̄
    for x in indvars
        @assert haskey(discretization.dxs, Num(x))||haskey(discretization.dxs, x) "Variable $x has no step size"
    end
    if !any(s -> discretization.advection_scheme isa s, [UpwindScheme, FunctionalScheme])
        throw(ArgumentError("Only `UpwindScheme()` and `FunctionalScheme()` are supported advection schemes. Got $(typeof(discretization.advection_scheme))."))
    end
    return if !(discretization.disc_strategy isa AbstractDiscretizationStrategy)
        throw(ArgumentError("Discretization strategy must be an `AbstractDiscretizationStrategy`, got $(typeof(discretization.disc_strategy))."))
    end
end

function PDEBase.check_boundarymap(boundarymap, discretization::MOLFiniteDifference)
    bs = filter_interfaces(flatten_vardict(boundarymap))
    for b in bs
        dx1 = discretization.dxs[Num(b.x)]
        dx2 = discretization.dxs[Num(b.x2)]
        if dx1 != dx2
            throw(ArgumentError("The step size of the connected variables $(b.x) and $(b.x2) must be the same. If you need nonuniform interface boundaries please post an issue on GitHub."))
        end
    end
    return
end

function get_discrete(pdesys, discretization)
    t = get_time(discretization)
    PDEBase.cardinalize_eqs!(pdesys)

    ############################
    # System Parsing and Transformation
    ############################
    # Parse the variables in to the right form and store useful information about the system
    v = VariableMap(pdesys, discretization)
    # Check for basic interface errors
    PDEBase.interface_errors(pdesys, v, discretization)
    # Extract tspan
    tspan = t !== nothing ? v.intervals[t] : nothing
    # Find the derivative orders in the bcs
    bcorders = Dict(map(x -> x => d_orders(x, get_bcs(pdesys)), all_ivs(v)))
    # Create a map of each variable to their boundary conditions including initial conditions
    boundarymap = PDEBase.parse_bcs(get_bcs(pdesys), v, bcorders)
    # Check that the boundary map is valid
    PDEBase.check_boundarymap(boundarymap, discretization)

    # Transform system so that it is compatible with the discretization
    if should_transform(pdesys, discretization, boundarymap)
        pdesys = PDEBase.transform_pde_system!(v, boundarymap, pdesys, discretization)
    end

    pdeeqs = get_eqs(pdesys)
    bcs = get_bcs(pdesys)

    ############################
    # Discretization of system
    ############################
    disc_state = PDEBase.construct_disc_state(discretization)

    # Create discretized space and variables, this is called `s` throughout
    s = PDEBase.construct_discrete_space(v, discretization)

    return Dict(
        vcat(
            [Num(x) => s.grid[x] for x in s.x̄], [Num(u) => s.discvars[u] for u in s.ū]
        )
    )
end

function ODEFunctionExpr(
        pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference
    )
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    return try
        if tspan === nothing
            @assert true "Codegen for NonlinearSystems is not yet implemented."
        else
            simpsys = mtkcompile(sys)
            return ODEFunction(simpsys; expression = Val{true})
        end
    catch e
        println("The system of equations is:")
        println(get_eqs(sys))
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end

function SciMLBase.ODEFunction(
        pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference;
        analytic = nothing, kwargs...
    )
    sys, tspan = SciMLBase.symbolic_discretize(pdesys, discretization)
    return try
        if tspan === nothing
            @assert true "Codegen for NonlinearSystems is not yet implemented."
        else
            simpsys = mtkcompile(sys)
            if analytic !== nothing
                analytic = analytic isa Dict ? analytic : Dict(analytic)
                s = getmetadata(sys, ModelingToolkit.ProblemTypeCtx, nothing).discretespace
                us = get_unknowns(simpsys)
                gridlocs = get_gridloc.(us, (s,))
                f_analytic = generate_function_from_gridlocs(analytic, gridlocs, s)
            end
            return ODEFunction(
                simpsys; analytic = f_analytic, eval_module = @__MODULE__,
                discretization.kwargs..., kwargs...
            )
        end
    catch e
        println("The system of equations is:")
        println(get_eqs(sys))
        println()
        println("Discretization failed, please post an issue on https://github.com/SciML/MethodOfLines.jl with the failing code and system at low point count.")
        println()
        rethrow(e)
    end
end

function generate_code(
        pdesys::PDESystem, discretization::MethodOfLines.MOLFiniteDifference,
        filename = "generated_code_of_pdesys.jl"
    )
    code = ODEFunctionExpr(pdesys, discretization)
    rm(filename; force = true)
    return open(filename, "a") do io
        println(io, code)
    end
end

"""
    SciMLBase.discretize(pdesys, discretization::MOLFiniteDifference; kwargs...)

Discretize a PDESystem using the method of lines. For linear PDEs on the fast path
(no special handling needed), this replaces the mtkcompile-generated ODE function with
a direct sparse matrix-vector multiplication using the pre-computed stencil matrix,
yielding ~100x faster ODE function evaluation for large grids.
"""
function SciMLBase.discretize(
        pdesys::PDESystem,
        discretization::MOLFiniteDifference;
        analytic = nothing, kwargs...
    )
    # StaggeredGrid has its own discretize override (positional arg signature)
    if discretization.grid_align isa StaggeredGrid
        return SciMLBase.discretize(pdesys, discretization, analytic)
    end

    cache_key = UInt64(objectid(discretization))

    # Use PDEBase's default discretize pipeline (handles symbolic_discretize, mtkcompile,
    # ODEProblem creation, metadata, analytic functions — all correctly)
    prob = invoke(SciMLBase.discretize,
                  Tuple{PDESystem, PDEBase.AbstractEquationSystemDiscretization},
                  pdesys, discretization; analytic = analytic, kwargs...)

    # Try to replace the ODE function with a fast sparse matvec
    fast_path_data = lock(_CACHE_LOCK) do
        pop!(_FAST_PATH_CACHE, cache_key, nothing)
    end
    if fast_path_data !== nothing && prob isa ODEProblem && all(d -> d.is_fast, fast_path_data)
        fast_prob = _try_build_fast_problem(prob, fast_path_data)
        if fast_prob !== nothing
            return fast_prob
        end
    end

    return prob
end

"""
    _try_build_fast_problem(prob, fast_path_data)

Attempt to build a fast ODEProblem using direct sparse matrix-vector multiplication
instead of the symbolically-generated ODE function from mtkcompile.

For linear PDEs with constant coefficients and boundary conditions, this replaces
the ODE function with `mul!(du, L, u) .+ bc_correction`, yielding ~100x faster
evaluation compared to the mtkcompile-generated function for large grids.

The boundary correction is computed numerically by comparing `L_ii * u0` against
`prob.f(du, u0, p, 0)`, avoiding the need to extract symbolic boundary values.

Returns the fast ODEProblem, or `nothing` if the fast path is not applicable.
"""
function _try_build_fast_problem(prob, fast_path_data)
    all(d -> d.is_fast, fast_path_data) || return nothing
    n_total = length(prob.u0)
    n_total == 0 && return nothing

    # Build block-diagonal stencil matrix for multi-variable systems
    L_blocks = SparseMatrixCSC{Float64, Int}[]
    total_interior = 0

    for data in fast_path_data
        s = data.discretespace
        eqvar = data.eqvar
        stencil_matrices = data.stencil_matrices

        u_dep = depvar(eqvar, s)
        haskey(s.discvars, u_dep) || return nothing
        discvars = s.discvars[u_dep]

        # Only handle 1D for now (discvars is a Vector, not a Matrix)
        discvars isa AbstractVector || return nothing
        gridlen = length(discvars)

        # Get stencil matrices for this variable
        u_op = operation(u_dep)
        haskey(stencil_matrices, u_op) || return nothing
        u_matrices = stencil_matrices[u_op]

        # Build combined stencil matrix (sum of all derivative operator matrices)
        L_combined = spzeros(Float64, gridlen, gridlen)
        for (_, mat_with_dim) in u_matrices
            if mat_with_dim isa Tuple && mat_with_dim[1] isa Tuple
                # Upwind derivatives: not supported in fast path yet
                return nothing
            elseif mat_with_dim isa Tuple && mat_with_dim[1] === :mixed
                # Mixed derivatives: not supported in fast path yet
                return nothing
            end
            L, _ = mat_with_dim
            L_combined .+= L
        end

        push!(L_blocks, L_combined)
        total_interior += gridlen
    end

    # For multi-variable: the problem u0 has all interior DOFs concatenated
    # Determine interior indices for each block
    interior_blocks = Vector{Int}[]
    offset = 0
    for (i, data) in enumerate(fast_path_data)
        s = data.discretespace
        eqvar = data.eqvar
        u_dep = depvar(eqvar, s)
        discvars = s.discvars[u_dep]
        gridlen = length(discvars)
        n_block = gridlen  # full gridlen; we'll determine interior from prob.u0 size

        push!(interior_blocks, collect(1:gridlen))
    end

    # For single-variable systems, use original logic to determine interior
    if length(fast_path_data) == 1
        gridlen = size(L_blocks[1], 1)
        n = n_total

        interior_indices = if n == gridlen - 2
            collect(2:gridlen-1)
        elseif n == gridlen
            collect(1:gridlen)
        else
            return nothing
        end

        L_ii = L_blocks[1][interior_indices, interior_indices]
    else
        # Multi-variable: build block-diagonal interior matrix
        # Determine per-variable DOF count from ordering
        # Heuristic: assume equal split or try common BC patterns
        n_vars = length(fast_path_data)
        gridlens = [size(L, 1) for L in L_blocks]

        # Try: each variable has (gridlen - 2) interior points (Dirichlet both ends)
        n_interior_per_var = [gl - 2 for gl in gridlens]
        if sum(n_interior_per_var) == n_total
            L_ii = blockdiag([L[2:end-1, 2:end-1] for L in L_blocks]...)
        elseif sum(gridlens) == n_total
            # No BCs stripped: periodic or all-Neumann
            L_ii = blockdiag(L_blocks...)
        else
            return nothing  # can't determine interior structure
        end
    end

    # Compute boundary correction numerically:
    # For f(du, u, p, t) = L_full * u_full, we have
    #   du_interior = L_ii * u_interior + L_ib * u_boundary
    # So bc_correction = du_standard - L_ii * u_interior
    du_standard = similar(prob.u0)
    prob.f(du_standard, prob.u0, prob.p, 0.0)
    du_from_Lii = L_ii * prob.u0
    bc_correction = du_standard - du_from_Lii
    has_bc_correction = any(x -> abs(x) > eps(), bc_correction)

    # Verify at a different time to ensure time-independence (linear, constant BCs)
    du_standard2 = similar(prob.u0)
    prob.f(du_standard2, prob.u0, prob.p, 0.5)
    if !isapprox(du_standard, du_standard2; rtol=1e-10)
        return nothing  # time-dependent — fast path not applicable
    end

    # Verify at a different u to ensure linearity (L*u + b form)
    u_perturbed = prob.u0 .* 1.5 .+ 0.1
    du_perturbed_std = similar(prob.u0)
    prob.f(du_perturbed_std, u_perturbed, prob.p, 0.0)
    du_perturbed_fast = L_ii * u_perturbed
    if has_bc_correction
        du_perturbed_fast .+= bc_correction
    end
    if !isapprox(du_perturbed_fast, du_perturbed_std; rtol=1e-6)
        return nothing  # nonlinear or coefficient mismatch — fast path not applicable
    end

    # Build fast ODE function closure
    L_fast = copy(L_ii)
    fast_rhs! = if has_bc_correction
        bc_corr = copy(bc_correction)
        (du, u, p, t) -> begin
            mul!(du, L_fast, u)
            du .+= bc_corr
            nothing
        end
    else
        (du, u, p, t) -> begin
            mul!(du, L_fast, u)
            nothing
        end
    end

    # Preserve the sys, observed, and other symbolic fields from the original ODEFunction
    # so that the solution wrapper can unpack variables correctly
    orig_f = prob.f
    fast_f = ODEFunction{true, SciMLBase.FullSpecialize}(
        fast_rhs!;
        sys = orig_f.sys,
        observed = orig_f.observed,
        jac_prototype = orig_f.jac_prototype,
    )
    return remake(prob; f=fast_f)
end
