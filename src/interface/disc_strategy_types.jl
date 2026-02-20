# Discretization strategies
# -------------------------
abstract type AbstractDiscretizationStrategy end

# Array discretization
# ~~~~~~~~~~~~~~~~~~~~~
# Builds sparse stencil matrices for each derivative operator and computes all
# derivatives via matrix-vector multiplication (`SparseMatrixCSC * Vector{Num}`).
# Equations are assembled using `@arrayop` from SymbolicUtils.jl for array-level
# structure, then scalarized for MTK compatibility. Falls back to per-point
# computation for special cases (nonlinear Laplacian, spherical diffusion, WENO, etc.).
struct ArrayDiscretization <: AbstractDiscretizationStrategy end
