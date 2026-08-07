using CSV, DataFrames, Plots
include("processingFunctions.jl")

test_name = "nesterovHessianBFGS_clean"


function loadResults(test_name, testname)
    df = CSV.read("test_results_"*test_name*"_clean.csv", DataFrame)
    results = Matrix(df[:, 2:end])
    functionsToTest = df[:, 1]
    return df, results, functionsToTest
end

df, results, functionsToTest = loadResults(test_name, test_name)