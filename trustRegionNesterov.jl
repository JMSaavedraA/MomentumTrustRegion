#Jose Miguel Saavedra Aguilar
#CIMAT Master in Applied Mathematics

using LinearAlgebra

include("s2mpjlib.jl")

"""
trustRegionNesterov.jl

High-level implementations of trust-region algorithms used in the paper
"A Momentum Trust-Region Algorithm for Unconstrained Optimization".

Provides:
- `trustRegionDoglegBFGS`  : Dogleg trust-region with BFGS updates
- `trustRegionDoglegHessian`: Dogleg trust-region using exact Hessian
- `trustRegionNesterovDoglegBFGS`: Momentum (Nesterov) + Dogleg + BFGS
- `trustRegionNesterovDoglegHessian`: Momentum (Nesterov) + Dogleg + Hessian

Typical internal defaults: `w1=1e-4`, `c0=0.1`, `c1=1/4`, `c2=1/2`, `η=2`,
`λ=0.8`, `Δ0=1`, `ΔMin=0.01*sqrt(n)`, `ΔMax=10*sqrt(n)`, `tol=1e-4`.

Example:
```julia
include("s2mpjlib.jl")
include("trustRegionNesterov.jl")
pb, pbm, prob = initProblem("ROSENBR")
x, Ng, k, tol, G = trustRegionNesterovDoglegBFGS(pb, pbm, prob)
```
"""

"""
updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)

Compute the updated trust-region radius according to the agreement ratio `ρ`.

Arguments
- `Δ` : current trust-region radius
- `ρ` : ratio of actual to predicted reduction (model agreement)
- `c1`, `c2` : thresholds for contraction / expansion
- `η` : expansion/contraction factor (>1)
- `ΔMin`, `ΔMax` : minimum and maximum allowed radius
- `d` : proposed step (kept for API compatibility)
- `c` : code from `dogleg` indicating which region was chosen ("N","D","C")

Returns
- new `Δ` (Number). Note: the caller should assign the returned value if they want
  to update the radius in-place: `Δ = updateΔ(Δ, ρ, ...)`.
"""
function updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
    if (ρ < c1)
        # Contraction: poor model agreement -> reduce radius
        Δ = max(Δ / η , ΔMin)
    elseif (ρ > c2 && c != "N")
        # Expansion: very good agreement and not a Newton step -> increase radius
        Δ = min(η * Δ, ΔMax)
    #else
        # keep Δ unchanged (alternative: set Δ = norm(d))
    end
    return Δ
end

"""
dogleg(Δ, g, B, pU, pN, d; tol=1e-6, dim=length(g))

Solve the trust-region subproblem approximately using the Dogleg method.

This routine computes (in-place) a step `d` following the dogleg path defined by
the Cauchy point `pU` and the (approximate) Newton step `pN`. The chosen step is
written into `d` and a code is returned indicating which region was used:
- "C" : Cauchy (steepest-descent) point clipped to the boundary
- "N" : Full Newton step (inside the trust region)
- "D" : Dogleg interpolation between Cauchy and Newton points

Arguments
- `Δ` : trust-region radius
- `g` : gradient vector
- `B` : (approximate) Hessian matrix
- `pU`, `pN` : preallocated vectors for Cauchy and Newton candidates
- `d` : output step (written in-place)

Returns
- `c` : String code describing which region was selected ("C","N","D").
"""
function dogleg(Δ::Number,g::AbstractVector,B::AbstractMatrix, pU::AbstractVector, pN::AbstractVector, d::AbstractVector, tol::Number=1e-6,dim=length(g))
    # Compute the scaled steepest-descent (Cauchy) point pU
    @inbounds pU .= - (g⋅g) / (g⋅(B*g)) * g
    NpU = norm(pU)
    if NpU > Δ
        # Cauchy point lies outside the trust region -> scale to the boundary
        @inbounds d .= Δ*pU/NpU
        c = "C"
    else
        # Compute (approximate) Newton step pN by solving B*pN = -g
        @inbounds pN .= - (B \ g)
        if norm(pN) <= Δ
            # Newton step is inside trust region -> accept it
            @inbounds d .= pN
            c = "N"
        else
            # Interpolate along the dogleg path between pU and pN
            w = pN-pU
            p1 = w⋅w
            p2 = pU⋅w
            p3 = (Δ + NpU) * (Δ - NpU)
            t = (-p2 + sqrt(p2^2 + p1*p3)) / p1
            @inbounds d .= pU + t*w
            c = "D"
        end
    end
    return c
end

"""
trustRegionDoglegBFGS(pb, pbm, f)

Trust-region method using the Dogleg subproblem solver and a BFGS Hessian update.

Arguments
- `pb`  : `PB` problem descriptor (from `s2mpjlib`)
- `pbm` : `PBM` problem metadata
- `f`   : problem evaluation function (e.g. `probfunc` returning `f`, `g`, `H`)

Returns
- `x`  : final iterate
- `Ng` : final gradient norm
- `k`  : number of iterations performed
- `tol`: gradient tolerance used
- `G`  : history of gradient norms per iteration
"""
function trustRegionDoglegBFGS(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters: Armijo / radius thresholds / update factors
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    # Evaluate initial f, g, H/B
    f0, g, B = f("fgHx", pbm, x)
    try
        # Ensure B is a dense matrix and regularize if ill-conditioned
        B = Matrix(B)
        if cond(B) * tol > 1
            B = 0.8 * B + 0.2 * I
        end
        S = bunchkaufman(Symmetric(collect(B))) # factorization to inspect inertia
        minD = minimum(diag(S.D))
        if minD < tol
            # Force positive-definiteness via diagonal modification
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P
        end
    catch
        # Fallback to identity if Hessian is unavailable or factorization fails
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    Δ = 1
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        if cond(B) * tol > 1 || any(isnan.(B)) || any(isinf.(B))
            try
                # Recompute Hessian at current iterate `x` (no momentum `v` available here)
                _, _, B = f("fgHx", pbm, x)
                B = Matrix(B)
                if any(isnan.(B)) || any(isinf.(B))
                    B = Matrix{Float64}(I, n, n)
                elseif cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
            catch
                B = Matrix{Float64}(I, n, n)
            end
        end
        c = dogleg(Δ, g, B, pU, pN, d, tol, n)
        @. xs = x + d
        try
            # Evaluate the objective at the candidate step and compute the trust ratio
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)  # predicted reduction (model)
            if !isfinite(denom) || abs(denom) < 1e-14
                # Avoid division by zero or non-finite model predictions
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                # Poor agreement -> shrink trust region and reject step
                Δ = max(Δ / η, ΔMin)
            else
                # Accept step: update iterate, gradient and perform BFGS update
                x .= xs
                y .= gs .- g
                # u = B * d (used in BFGS denominator)
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                # BFGS rank-two update: ensure denominators are safe before update
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du
                else
                    # Skip BFGS update to avoid division by zero / numerical issues
                end
                g .= gs
                f0 = f1
                Ng = norm(g)
                # Update trust radius (returns a new Δ; assign if desired)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            # Evaluation failed: conservatively reduce Δ
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]  # Truncate the gradient norm history
    return x, Ng, k, tol, G
end


"""
trustRegionDoglegHessian(pb, pbm, f)

Trust-region method using the Dogleg subproblem and an exact Hessian evaluation.

This variant queries the problem for the Hessian (via `f("fgHx", ...)`) and
regularizes it if necessary. The overall structure is: initialize, perform a
small line search for an initial radius, then iterate: solve the dogleg
subproblem, evaluate the model, compute the trust ratio `ρ`, accept or reject
the step, and update the trust radius.

Returns `(x, Ng, k, tol, G)` as in the BFGS variant.
"""
function trustRegionDoglegHessian(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    # Algorithm parameters
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    tol = 1e-4
    # Evaluate initial f, gradient and Hessian
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        if cond(B) * tol > 1
            B = 0.8 * B + 0.2 * I
        end
        S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
        end
    catch
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    Δ = 1
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        try
            if any(isnan.(B)) || any(isinf.(B))
                B = Matrix{Float64}(I, n, n)
            elseif cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        c = dogleg(Δ, g, B, pU, pN, d, tol, n)
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            if !isfinite(denom) || abs(denom) < 1e-14
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                f0, g, B = f("fgHx", pbm, x)
                B = Matrix(B)
                Ng = norm(g)
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng
    end
    G = G[1:k]  # Truncate the gradient norm history
    return x, Ng, k, tol, G
end

"""
trustRegionNesterovDoglegBFGS(pb, pbm, f)

Momentum (Nesterov-style) trust-region method with Dogleg subproblem and BFGS updates.

This variant maintains a momentum displacement `v` that is updated each iteration.
An Armijo-like line search is used to scale the momentum when necessary. The BFGS
update is performed at shifted points `(x + v)` to account for the momentum.

Returns `(x, Ng, k, tol, G)`.
"""
function trustRegionNesterovDoglegBFGS(pb::PB, pbm::PBM, f::Function)
    # Algorithm parameters and initialization
    maxIter = 100000  # Maximum iterations
    G = zeros(Float64, maxIter)  # Gradient norm history
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0  # Iteration counter
    tol = 1e-4
    # Initial evaluation at x (not shifted)
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        if cond(B) * tol > 1
            B = 0.5 * (B + I)
        end
        S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
        end
    catch
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    v = zeros(Float64, n)   # Initial momentum is zeros
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    Δ = 1
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end

        if cond(B) * tol > 1 || any(isnan.(B)) || any(isinf.(B))
            try
                _, _, B = f("fgHx", pbm, x + v) # Recompute B
                B = Matrix(B)
                if any(isnan.(B)) || any(isinf.(B))
                    B = Matrix{Float64}(I, n, n)
                elseif cond(B) * tol > 1
                    B .= 0.8 * B + 0.2 * I
                end
                S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
                minD = minimum(diag(S.D))
                if minD < tol
                    B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
                end
            catch
                B = Matrix{Float64}(I, n, n)
            end
        end
        
        c = dogleg(Δ,g,B,pU,pN,d,tol,n) # Dogleg approximation of the quadratic model
        @. xs = x + v + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            if !isfinite(denom) || abs(denom) < 1e-14
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end # The trust ratio ρ
            if ρ < c0 && Δ > ΔMin
                # Case 1. Reject due to bad trust ratio and reduce the trust radius
                Δ = max(Δ / η , ΔMin)
            else
                # Case 2. Accept the step and update the model
                x .= xs    # Update x_k
                Ng = norm(gs)   # Update ||∇f(x_k)||
                y .= gs - g  # Update y_k
                # Momentum update: add the current step to the momentum displacement `v`
                # (Nesterov-style accumulation). Ensure `v` is a descent direction.
                v .= v + d   # Update v_k
                if dot(gs,v) > 0
                    v .= -v # Make momentum a descent direction if it is not already
                end
                jter = 0;
                f0 = f("fx", pbm, x + v)
                goOn = f0 > f1 + w1 * dot(gs, v)
                while goOn && jter < 30
                    v .= λ * v
                    f0 = f("fx", pbm, x + v)
                    goOn = f0 > f1 + w1 * dot(gs, v)
                    jter += 1
                end
                evalgrsum!(x + v, pbm, g)  # Update ∇f(x_k + v_k)
                # BFGS Update (robustified)
                y += g - gs  # Update y_k -> final y = g_new - g_old
                mul!(u, B, (d + v))
                yd = dot(y, (d+v))
                du = dot(u, (d+v))
                if isfinite(yd) && isfinite(du) && abs(yd) > 1e-14 && abs(du) > 1e-14
                    B .+= (y * y') / yd - (u * u') / du # Update ∇²f(x_k)
                else
                    # Skip update if denominators are numerically unsafe
                end
                # Update the trust radius
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng    # Store the norm of ∇f(x_k)
    end
    G = G[1:k]  # Truncate the gradient norm history
    return x, Ng, k, tol, G
end


"""
trustRegionNesterovDoglegHessian(pb, pbm, f)

Momentum trust-region method using the Dogleg solver with exact Hessian evaluations
at shifted points `(x + v)`. This variant keeps the same momentum handling as the
BFGS version but queries the problem for the Hessian when updating the model.

Returns `(x, Ng, k, tol, G)`.
"""
function trustRegionNesterovDoglegHessian(pb::PB, pbm::PBM, f::Function)
    # Algorithm parameters and initialization
    maxIter = 100000  # Maximum iterations
    G = zeros(Float64, maxIter)  # Gradient norm history
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0  # Iteration counter
    tol = 1e-4
    # Initial evaluation at x
    f0, g, B = f("fgHx", pbm, x)
    try
        B = Matrix(B)
        if cond(B) * tol > 1
            B = 0.5 * (B + I)
        end
        S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
        minD = minimum(diag(S.D))
        if minD < tol
            B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
        end
    catch
        B = Matrix{Float64}(I, n, n)
    end
    Ng = norm(g)
    y = similar(g)
    pU = similar(g)
    pN = similar(g)
    gs = similar(g)
    v = zeros(Float64, n)   # Initial momentum is zeros
    c = "I"
    u = similar(g)
    d = -g
    s = dot(d, g)
    jter = 0
    Δ = 1
    # Line search for initial Δ
    try
        f1 = f("fx", pbm, x + Δ * d)
        goOn = f1 > f0 + w1 * Δ * dot(g, d)
        while goOn && jter < 30
            Δ .= λ * Δ
            f1 = f("fx", pbm, x + Δ * d)
            goOn = f1 > f0 + w1 * Δ * dot(g, d)
            jter += 1
        end
    catch
        Δ = norm(g)
    end

    t_start = time()  # Start timer
    while Ng > tol && k < maxIter
        # Timeout check
        if time() - t_start > 3600
            println("Timeout: Exceeded 1 hour in trustRegionDogleg.")
            break
        end
        
        if any(isnan.(B)) || any(isinf.(B))
            B = Matrix{Float64}(I, n, n)
        end
        
        try
            if any(isnan.(B)) || any(isinf.(B))
                B = Matrix{Float64}(I, n, n)
            elseif cond(B) * tol > 1
                B .= 0.8 * B + 0.2 * I
            end
            S = bunchkaufman(Symmetric(collect(B))) # B=P'UDU'P factorization (L'DL) with permutations
            minD = minimum(diag(S.D))
            if minD < tol
                B .= S.P'*S.U*abs.(S.D)*S.U'*S.P #B is P.D. iff D has positive entries
            end
        catch
            B = Matrix{Float64}(I, n, n)
        end
        
        c = dogleg(Δ,g,B,pU,pN,d,tol,n) # Dogleg approximation of the quadratic model
        
        @. xs = x + v + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            if !isfinite(denom) || abs(denom) < 1e-14
                ρ = -Inf
            else
                ρ = (f1 - f0) / denom
            end # The trust ratio ρ
            if ρ < c0 && Δ > ΔMin
                # Case 1. Reject due to bad trust ratio and reduce the trust radius
                Δ = max(Δ / η , ΔMin)
            else
                # Case 2. Accept the step and update the model
                x .= xs    # Update x_k
                Ng = norm(gs)   # Update ||∇f(x_k)||
                y .= gs - g  # Update y_k
                v .= v + d   # Update v_k
                if dot(gs,v) > 0
                    v .= -v # Make momentum a descent direction if it is not already
                end
                jter = 0;
                f0 = f("fx", pbm, x + v)
                goOn = f0 > f1 + w1 * dot(gs, v)
                while goOn && jter < 30
                    v .= λ * v
                    f0 = f("fx", pbm, x + v)
                    goOn = f0 > f1 + w1 * dot(gs, v)
                    jter += 1
                end
                f0, g, B = f("fgHx", pbm, x + v)
                B = Matrix(B)
                # Update the trust radius
                Δ = updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
            end
        catch
            Δ = max(Δ / η, ΔMin)
        end
        k += 1
        G[k] = Ng    # Store the norm of ∇f(x_k)
    end
    G = G[1:k]  # Truncate the gradient norm history
    return x, Ng, k, tol, G
end