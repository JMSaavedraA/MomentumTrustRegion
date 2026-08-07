using Base.Threads, Plots


function initProblem(name::String)
    probfunc = getfield(Main, Symbol(name));
    pb, pbm = probfunc("setup")
    return pb, pbm, probfunc
end

function loadproblem(name::String)
    # Load the problem from the file
    filename = "julia_problems/$name.jl";
    include(filename);
end

function checkConvergence(results::AbstractMatrix, algos::Vector{String})
    # Check if all results are finite
    for i in 1:size(results, 1)
        for j in 1:length(algos)
            thisResult = results[i, 3*j-2]
            thisIter = results[i, 3*j-1]
            if thisResult < 0 || isnan(thisResult) || isinf(thisResult) || thisResult > 1e-4 || thisIter == 100000
                results[i, 3*j-2] = Inf  # Set corresponding g value to Inf
                results[i, 3*j-1] = Inf  # Set corresponding k value to Inf
                results[i, 3*j] = Inf   # Set corresponding t value to Inf
            end
        end
    end
end

function compute_performance_profile(data::AbstractMatrix, metric_indices)
    n_problems = size(data, 1)
    n_solvers = length(metric_indices)

    # Build metric matrix
    metric_matrix = zeros(n_problems, n_solvers)
    for (j, idx) in enumerate(metric_indices)
        for i in 1:n_problems
            value = data[i, idx]
            metric_matrix[i, j] = isnan(value) || value <= 0 ? Inf : value
        end
    end

    # Compute performance ratios
    ratios = similar(metric_matrix)
    for i in 1:n_problems
        row = metric_matrix[i, :]
        best = minimum(row)
        for j in 1:n_solvers
            ratios[i, j] = isfinite(row[j]) ? row[j] / best : Inf
        end
    end

    # Determine tau range
    finite_ratios = filter(isfinite, vec(ratios))
    tau_max = maximum(finite_ratios) * 1.5
    taus = sort!(unique!(finite_ratios))   # maybe also ensure 1.0 is included
    if !any(taus .== 1.0)
        pushfirst!(taus, 1.0)
    end


    # Compute profile for each solver
    profiles = zeros(length(taus), n_solvers)
    for (k, τ) in enumerate(taus)
        for j in 1:n_solvers
            profiles[k, j] = count(ratios[:, j] .<= τ) / n_problems
        end
    end

    return taus, profiles
end

function plot_and_save_profile(data::AbstractMatrix, algo_names::Vector{String}, output_dir::String = ".")
    T = 3 * (1:length(algo_names))  # Column indices per metric
    metrics = [
        ("Gradient Norm", T .- 2),
        ("Iterations",    T .- 1),
        ("Time",          T)
    ]

    for (label, indices) in metrics
        taus, profiles = compute_performance_profile(data, indices)
        logtaus = log.(taus)

        # Create fresh plot
        p = plot()
        for j in 1:length(algo_names)
            plot!(p, logtaus, profiles[:, j], label=algo_names[j], lw=2, palette = :Dark2_8, seriestype=:steppost)
        end

        # Add plot styling
        plot!(p,
            title = "Performance Profile: $label",
            xlabel = "log(τ)",
            ylabel = "ρ(τ)",
            legend = :bottomright,
            grid = :both,
            size = (700, 500),
            dpi = 1200,
            titlefont = font(16, "Computer Modern"),
            guidefont = font(12, "Computer Modern"),
            tickfont = font(10, "Computer Modern"),
            legendfont = font(10, "Computer Modern")
            )

        # Ensure directory exists
        if !isdir(output_dir)
            mkpath(output_dir)
        end

        filename = joinpath(output_dir, "performance_profile_$(lowercase(replace(label, ' ' => "_"))).pdf")
        savefig(p, filename)
    end
end

function compareMethods(functions::Vector{String},algos::Vector{String})
    # Atomic variables for thread safety
    n = length(functions);
    m = length(algos);
    # Preallocate results matrix: rows = functions, columns = [gD, kD, tD, gN, kN, tN, gP, kP, tP, gDD, kDD, tDD, gQ, kQ, tQ, gM, kM, tM]
    results = zeros(Float64, n, 3*m);

    @threads for i in 1:n
        name = functions[i];
        pb, pbm, probfunc = initProblem(name);
        for j in 1:m
            # Initialize all outputs as NaN
            g = k = t_local = NaN;
            algo_name = algos[j];
            algo = getfield(Main, Symbol(algo_name));
            try
                t_local = @elapsed _, g, k, _, _, _, _ = algo(pb, pbm, probfunc);
            catch e
                @info "$algo_name failed for $name: $e"
                g = k = t_local = NaN;
            end
            results[i, 3 * j - 2 : 3 * j] = [g, k, t_local];
        end
        println("Function $name completed successfully.")
    end
    return results
end

function compareMethodsOneFunc(functionName::String,algos::Vector{String})
    # Atomic variables for thread safety
    m = length(algos);
    pb, pbm, probfunc = initProblem(functionName);
    allGrads = zeros(Float64, 100000, m);
    allFVals = zeros(Float64, 100000, m);
    allAcc = zeros(Float64, m);
    allTimes = zeros(Float64, m);
    allIters = zeros(Int, m);
    @threads for j in 1:m
        # Initialize all outputs as NaN
        g = t = acc = 0.0;
        k = 0;
        G = zeros(Float64, 100000);
        F = zeros(Float64, 100000);
        algo_name = algos[j];
        algo = getfield(Main, Symbol(algo_name));
        try
            t = @elapsed _, _, k, _, G, F, acc = algo(pb, pbm, probfunc);
        catch e
            @info "$algo_name failed for $functionName: $e"
            g = k = t = acc = NaN;
            G[1:100000] = NaN;
            F[1:100000] = NaN;
        end
        allGrads[1:k, j] = G;
        allFVals[1:k, j] = F;
        allAcc[j] = acc;
        allTimes[j] = t;
        allIters[j] = k;
    end
    println("Function $functionName completed successfully.")
    return allGrads, allFVals, allAcc, allTimes, allIters
end
