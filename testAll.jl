using LinearAlgebra, Base.Threads, DelimitedFiles, StatsBase, Plots, CSV, DataFrames

include("s2mpjlib.jl")
include("trustRegionMomentum.jl")
include("functions.jl")
include("processingFunctions.jl")

for f in functionsToTest
    loadproblem(f)
end

algos = ["trustRegionDoglegHessian", "trustRegionDoglegBFGS", "trustRegionDoglegDampedBFGS", "trustRegionMomentumDoglegHessian", "trustRegionMomentumDoglegBFGS", "trustRegionMomentumDoglegDampedBFGS"]

results = compareMethods(functionsToTest, algos)

# Define header
header = ["Function", "gDH", "kDH", "tDH", "gDB", "kDB", "tDB", "gDD", "kDD", "tDD", "gMH", "kMH", "tMH", "gMB", "kMB", "tMB", "gMD", "kMD", "tMD"]

# Combine function names and results into a DataFrame
df = DataFrame([:Function => functionsToTest], copycols=false)
for (i, col) in enumerate(header[2:end])
    df[!, col] = results[:, i]
end

test_name = "momentumHessianDampedBFGS"

# Save to CSV
CSV.write("test_results_"*test_name*".csv", df)

println("Results saved to test_results_"*test_name*".csv")

algos_names = ["CH", "CB", "CD", "MH", "MB", "MD"]
checkConvergence(results, algos_names)
for (i, col) in enumerate(header[2:end])
    df[!, col] = results[:, i]
end
CSV.write("test_results_"*test_name*"_clean.csv", df)
plot_and_save_profile(results, algos_names, "Figures_"*test_name)