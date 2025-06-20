# MATSim vs METROPOLIS Trip Comparison

This repository contains code and data processing pipelines for comparing transportation simulation results between [**MATSim**](https://www.matsim.org/) and [**METROPOLIS**](https://metropolis.lucasjavaudin.com/) models. 
The main focus is on trip-level comparisons in terms of travel time and routing behavior, as well as evaluating the consistency of METROPOLIS under multiple simulation runs.
This testing ground for this comparison is the [**Open Berlin Scenario**](https://github.com/matsim-scenarios/matsim-berlin).

## Project Overview

This project handles:

- Preprocessing of MATSim's Open Berlin Scenario output to transform it into METROPOLIS inputs.
- Matching trips across the two models using agent-level and trip-level identifiers.
- Building consistent `agent_id` and `trip_index` keys to enable trip alignment.
- Computing differences in travel times and analyzing their distribution.

## Main Features

### 🔍 Trip-level Comparison

- Aligns MATSim and METROPOLIS trips using common identifiers.
- Calculates travel time discrepancies and plots them using 2D histograms.
- Highlights over- and under-estimations by either model.


### ⚙️ Data Processing

- Written in **Python** using [Polars](https://pola.rs/).
- Handles millions of rows with small computation time.
- Prepares input structures compatible with METROPOLIS.
- Allows for a structured and rigorous comparison of both models' outputs.

## Folder Structure
Currently, there are two scripts to transform the MATSim Berlin output into Metropolis 2 input.
There is also a folder containing the functions to use these scripts. One script with the Demand-side functions and one with the Supply-side functions.
