# Momentum Trust-Region — Julia code

Julia implementation and experiment code used for the paper “A Momentum Trust-Region Algorithm for Unconstrained Optimization” (Saavedra & Dalmau). This version is specially adapted to run the S2MPJ problem structure and includes the specific algorithms and variants tested in the paper.

## Repository structure
- `trustRegionNesterov.jl` — main algorithms (Nesterov/momentum + trust-region variants)
- `trustRegionNesterov.jl` (Hessian / BFGS variants included)
- `functions.jl` — list/loader of S2MPJ problems used in experiments
- `processingFunctions.jl` — helpers: `initProblem`, profiling, plotting, CSV export
- `testsHW.jl` — test harness that compares solvers and writes CSV/figures
- `README.md` — this file

## Requirements
- Julia (paper used Julia 1.12.5; any recent 1.12.x should work)
- Recommended packages: `Plots`, `CSV`, `DataFrames`, `StatsBase` (install below)
- Standard library: `LinearAlgebra`, `DelimitedFiles`, `Base.Threads`
- S2MPJ problem dataset (download from the official repository and copy the `julia_problems/` folder and 's2mpjl.jl' file into this repo root)

## Quick start
1. Open a terminal in the repository root (the folder containing this `README.md`).
2. (Optional) set number of Julia threads for parallel problem runs:

PowerShell:

```powershell
$env:JULIA_NUM_THREADS = 8
```

Linux/macOS:

```bash
export JULIA_NUM_THREADS=8
```

3. Install required packages (one-time):

```bash
julia --project=. -e "using Pkg; Pkg.add([\"Plots\", \"CSV\", \"DataFrames\", \"StatsBase\"]); Pkg.precompile()"
```

4. Run the test harness (this executes the experiments and writes CSV + figures):

```bash
julia --project=. testsHW.jl
```

The harness will save CSV results (e.g. `test_results_*.csv`) and figures into folders like `Figures_<test_name>`.

## Run a single problem interactively
Start Julia in the repo root and load helpers:

```julia
include("s2mpjlib.jl")
include("functions.jl")          # loads problem definitions
include("processingFunctions.jl")# provides initProblem()
include("trustRegionNesterov.jl")

pb, pbm, probfunc = initProblem("AIRCRFTB")  # pick a problem name from functions.jl
x, g, k, tol, G = trustRegionNesterovDoglegHessian(pb, pbm, probfunc)
```

Adjust the solver call to one of the available functions (see `trustRegionNesterov.jl`).

## Outputs
- CSV results: saved by `testsHW.jl` as `test_results_<name>.csv`.
- Figures: saved under `Figures_<test_name>`; filenames like `performance_profile_*.png`.

## Reproducibility notes
- Paper experiments used `JULIA_NUM_THREADS=8` and Julia 1.12.5.

## Contact & citation
Authors: Jose Miguel Saavedra (jose.saavedra@cimat.mx), Oscar Dalmau (dalmau@cimat.mx).
If you use this code, please cite the manuscript.

## License
