#Jose Miguel Saavedra Aguilar
#CIMAT Master in Applied Mathematics

using LinearAlgebra

include("s2mpjlib.jl")
include("gradienteConjugado.jl")

function updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
    if (ρ < c1)
        # Reduce the trust radius
        Δ = max(Δ / η , ΔMin)
    elseif (ρ > c2 && c != "N")
        Δ = min(η * Δ, ΔMax) # Increase Δ
    #else
        #Δ = norm(d)
    end
end

function dogleg(Δ::Number,g::AbstractVector,B::AbstractMatrix, pU::AbstractVector, pN::AbstractVector, d::AbstractVector, tol::Number=1e-6,dim=length(g))
    # Dogleg method for the trust region problem
    @inbounds pU .= - (g⋅g) / (g⋅(B*g)) * g
    NpU = norm(pU)
    if NpU > Δ
        @inbounds d .= Δ*pU/NpU
        c = "C"
    else
        @inbounds pN .= - (B \ g)
        if norm(pN) <= Δ
            @inbounds d .= pN
            c = "N"
        else
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

function trustRegionDoglegBFGS(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    tol = 1e-4
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
        c = dogleg(Δ, g, B, pU, pN, d, tol, n)
        @. xs = x + d
        try
            f1 = evalgrsum!(xs, pbm, gs)
            denom = 0.5 * dot(d, B*d) + dot(g, d)
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                y .= gs .- g
                mul!(u, B, d)
                yd = dot(y, d)
                du = dot(d, u)
                B .+= (y * y') / yd - (u * u') / du
                g .= gs
                f0 = f1
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
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


function trustRegionDoglegHessian(pb::PB, pbm::PBM, f::Function)
    maxIter = 100000
    G = zeros(Float64, maxIter)
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0
    tol = 1e-4
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
            ρ = (f1 - f0) / denom
            if ρ < c0 && Δ > ΔMin
                Δ = max(Δ / η, ΔMin)
            else
                x .= xs
                f0, g, B = f("fgHx", pbm, x)
                B = Matrix(B)
                Ng = norm(g)
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
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

function trustRegionNesterovDoglegBFGS(pb::PB, pbm::PBM, f::Function)
    # Trust-Region method for noisy functions with dogleg approximation of the model and BFGS Update
    # Algorithm's parameters
    maxIter = 100000  # Maximum iterations
    G = zeros(Float64, maxIter)  # Gradient norm history
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0  # Iteration counter
    tol = 1e-4
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
            ρ = ((f1-f0)/(.5*d⋅(B*d) + g⋅d)) # The trust ratio ρ
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
                evalgrsum!(x + v, pbm, g)  # Update ∇f(x_k + v_k)
                # BFGS Update
                y += g - gs  # Update y_k
                mul!(u, B, (d + v))
                yd = dot(y, (d+v))
                du = dot(u, (d+v))
                B .+= (y * y') / yd - (u * u') / du # Update ∇²f(x_k)
                # Update the trust radius
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
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


function trustRegionNesterovDoglegHessian(pb::PB, pbm::PBM, f::Function)
    # Trust-Region method for noisy functions with dogleg approximation of the model and BFGS Update
    # Algorithm's parameters
    maxIter = 100000  # Maximum iterations
    G = zeros(Float64, maxIter)  # Gradient norm history
    x = copy(pb.x0)
    n = length(x)  # the length of x_k
    w1, c0, c1, c2, η, λ = 1e-4, 0.1, 1/4, 1/2, 2, 0.8
    ΔMax, ΔMin = 10 * sqrt(n), 0.01 * sqrt(n)
    xs = similar(x)
    k = 0  # Iteration counter
    tol = 1e-4
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
            ρ = ((f1-f0)/(.5*d⋅(B*d) + g⋅d)) # The trust ratio ρ
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
                updateΔ(Δ, ρ, c1, c2, η, ΔMin, ΔMax, d, c)
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