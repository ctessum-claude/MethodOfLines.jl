# Discretization strategies
# -------------------------
abstract type AbstractDiscretizationStrategy end

# Array discretization
# ~~~~~~~~~~~~~~~~~~~~~
# This discretization strategy builds sparse stencil matrices for each derivative
# operator and computes all derivatives at once via matrix-vector multiplication.
# Equations are assembled at the array level using @arrayop where possible,
# falling back to per-point computation for special cases (nonlinear Laplacian,
# spherical diffusion, WENO, etc.).
struct ArrayDiscretization <: AbstractDiscretizationStrategy end
