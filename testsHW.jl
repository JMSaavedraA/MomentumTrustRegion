using LinearAlgebra, Base.Threads, DelimitedFiles, StatsBase, Plots, CSV, DataFrames

include("s2mpjlib.jl")
include("trustRegionNesterov.jl")
include("functions.jl")
include("processingFunctions.jl")


functionsToTest = names

algos = ["trustRegionDoglegHessian", "trustRegionDoglegBFGS", "trustRegionNesterovDoglegHessian", "trustRegionNesterovDoglegBFGS"]

results = compareMethods(functionsToTest, algos)

# Define header
header = ["Function", "gDH", "kDH", "tDH", "gDB", "kDB", "tDB", "gNH", "kNH", "tNH", "gNB", "kNB", "tNB"]

# Combine function names and results into a DataFrame
df = DataFrame([:Function => functionsToTest], copycols=false)
for (i, col) in enumerate(header[2:end])
    df[!, col] = results[:, i]
end

test_name = "nesterovHessianBFGS"

# Save to CSV
CSV.write("test_results_"*test_name*".csv", df)

println("Results saved to test_results_"*test_name*".csv")

algos_names = ["Usual Dogleg with Hessian", "Usual Dogleg with BFGS", "Nesterov Dogleg with Hessian", "Nesterov Dogleg with BFGS"]
checkConvergence(results, algos_names)
for (i, col) in enumerate(header[2:end])
    df[!, col] = results[:, i]
end
CSV.write("test_results_"*test_name*"_clean.csv", df)
plot_and_save_profile(results, algos_names, "Figures_"*test_name)