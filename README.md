# Replicating the MATSim Berlin Scenario using METROPOLIS2

This repository contains scripts and functions to **replicate the MATSim Open Berlin Scenario (v6.4)** using the **METROPOLIS2 simulator (v1.1.0)**. The goal is to convert MATSim inputs into METROPOLIS-compatible format, use the newly generated output and compare it with the MATSim outputs to evaluate their eprformance in different scenarios.

---

## Overview

The project uses input and output files from the [Open Berlin Scenario](https://github.com/matsim-scenarios/matsim-berlin) and replicates four simulation setups using METROPOLIS2:

1. **Single-trip agents**: each trip in MATSim represents a single agent in METROPOLIS. No modifications to route/mode/departure time, purely replication.
2. **Multi-trip agents**: a trip in METROPOLIS is made of "a tour" - defined by all trips between two "main" activities (activities whose duration is limited by an `end_time` and not a `max_duration`).
3. **Routing choice scenario**: same set up as the previous scenario, with METROPOLIS routing choice active, to evaluate both simulators' routing algorithms.
4. **Departure-time choice scenario**: each MATSim agent is represented by a single METROPOLIS agent. Routing choice and departure time choice are both active.

Each scenario includes custom processing and conversion of MATSim input data (plans, network, and transit schedules) into METROPOLIS format, followed by simulation runs and output comparisons.


---

## 🗂️ Structure

```bash
├── Execution Scripts/
│   └── Long-process utilities (e.g., reading MATSim .xmls, or generating base datasets for more scenarios)
│
├── Notebooks/
│   ├── Input generation for each scenario
│   ├── Analysis & visualization of outputs
│   └── Scripts for demand and supply transformation
│
├── functions/
│   ├── Demand_functions.py     # Generates demand inputs for METROPOLIS
│   ├── Supply_functions.py     # Generates supply inputs for METROPOLIS
│   └── mpl_utils.py            # Unused, kept for legacy reasons
```


# ⚙️ Requirements	
- Python 3.10+	
- METROPOLIS2 simulator v1.1.0 (precompiled CLI version)
- Typical Python scientific stack: pandas, numpy, matplotlib, etc.
- Other Python modules: polars, XML Tree, sys, seaborn, json, etc.
### Important: 
You must add the functions/ directory to your path in each notebook:

```python
import sys
sys.path.append("../../functions")
```

---
# 📊 Results
- Outputs include travel time distributions, departure times, and visualizations available in the Notebooks/ folder.
- METROPOLIS2 results are compared to those from MATSim for validation purposes.

---
# 📦 Data Sources
All input data comes from the Open Berlin Scenario:
- Repository: [matsim-berlin](https://github.com/matsim-scenarios/matsim-berlin)

---
# 📝 References & Acknowledgements

-The MATSim Open Berlin Scenario is developed and maintained by the MATSim community.
- The METROPOLIS simulator was developed by Lucas Javaudin and André de Palma.
- This replication study was developed at Cergy Paris Université.

import Demand_functions as dmd
import Supply_functions as sup
```
