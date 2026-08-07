using LinearAlgebra, Base.Threads, DelimitedFiles, StatsBase, Plots, CSV, DataFrames

include("s2mpjlib.jl")
include("trustRegionMomentum.jl")
include("processingFunctions.jl")

funcName = "VESUVIOULS"
plotting = false

algos = ["trustRegionDoglegHessian", "trustRegionDoglegBFGS", "trustRegionDoglegDampedBFGS", "trustRegionMomentumDoglegHessian", "trustRegionMomentumDoglegBFGS", "trustRegionMomentumDoglegDampedBFGS"]

loadproblem(funcName)
allGrads, allFVals, allAcc, allTimes, allIters = compareMethodsOneFunc(funcName, algos)

kMax = maximum(allIters)

for i in eachindex(algos)
    allFVals[allIters[i]:kMax, i] .= allFVals[allIters[i], i]
    allGrads[allIters[i]:kMax, i] .= allGrads[allIters[i], i]
end

algos_names = ["CH", "CB", "CD", "MH", "MB", "MD"]

if(plotting)
    # Plot gradient norm histories
    plt1 = plot()
    for (i, algo) in enumerate(algos_names)
        plot!(plt1,
            1:kMax,
            allGrads[1:kMax, i],
            label=algo,
            xlabel="Iteration",
            ylabel="Gradient Norm",
            title="Gradient Norm Histories for $funcName",
            size = (700, 500),
            palette = :Dark2_8,
            dpi = 1200
            )
    end
    # Optionally save the figure
    savefig(plt1, string(funcName, "_gradient_norms.pdf"))

    # Plot function value histories
    plt2 = plot()
    for (i, algo) in enumerate(algos_names)
        plot!(plt2,
            1:kMax,
            allFVals[1:kMax, i],
            label=algo,
            xlabel="Iteration",
            ylabel="Function Value",
            title="Function Value Histories for $funcName",
            size = (700, 500),
            palette = :Dark2_8,
            dpi = 1200
            )
    end
    # Optionally save the figure
    savefig(plt2, string(funcName, "_function_values.pdf"))
end

# Create CSV with acceptance percentage, CPU time, and iterations


results_df = DataFrame(
    "Algorithm" => algos,
    "Acceptance Percentage" => allAcc,
    "CPU Time" => allTimes,
    "Iterations" => allIters
)

CSV.write(string(funcName, "_summary.csv"), results_df)

println("Summary CSV file created: ", string(funcName, "_summary.csv"))
println("Done!")