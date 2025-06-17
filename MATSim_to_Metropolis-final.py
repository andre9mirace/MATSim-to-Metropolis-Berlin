#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import pyarrow
import sys
import json
import math
import mpl_utils
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
import polars as pl
import xml.etree.ElementTree as ET

from xopen import xopen
from datetime import time


# # Directories

# In[2]:


# General directories
general_directory = '/Users/andre/Desktop/Cergy/'

berlin_directory = 'MATSim/matsim-berlin/input/v6.4/'

pt_10pct_dir = "Python_Scripts/runs/pt_10pct/" # This script's output

# Functions
sys.path.append("../functions")


# ### Three files for Berlin

# In[3]:


# MATSim Berlin paths
# Supply
NETWORK_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4-network.xml.gz")) # Network

VEHICLE_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4-vehicleTypes.xml")) # Vehicles

# Demand
PLAN_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4.output_plans.xml.gz")) # Plans


# ## Metropolis directories

# In[4]:


# Metropolis directories
METRO_INPUT = (os.path.join(general_directory, pt_10pct_dir, "metro_inputs/")) 
METRO_OUTPUT = (os.path.join(general_directory, pt_10pct_dir, "metro_outputs/"))


# # Parameters

# In[31]:


POPULATION_SHARE = 0.10 # 10% of total population to match MATSim Berlin

# Parameters to use for the simulation.
PARAMETERS ={
    "input_files": {
      "agents": (os.path.join(METRO_INPUT, "agents.parquet")) ,
      "alternatives": (os.path.join(METRO_INPUT, "alts.parquet")),
      "trips": (os.path.join(METRO_INPUT, "trips.parquet")),
      "edges": (os.path.join(METRO_INPUT, "edges.parquet")),
      "vehicle_types": (os.path.join(METRO_INPUT, "vehicles.parquet"))
                },
    "output_directory": METRO_OUTPUT,
    "period": [0.0, 86400.0],
    "road_network": {
        "recording_interval": 950.0,
        "approximation_bound": 1.0,
        "spillback": True,
        "backward_wave_speed": 15.0,  # 15km/h to match MATSim
        "max_pending_duration": 30.0, # to avoid gridlock - 30s to avoid gridlock
        "constrain_inflow": True,
        "algorithm_type": "Best"
    },
    "learning_model": {
      "type": "Linear"
    },
    "init_iteration_counter": 1,
    "max_iterations": 1,
    "update_ratio": 1.0,
    "random_seed": 13081996,
    "nb_threads": 16, # to match MATSim Berlin
    "saving_format": "Parquet",
    "only_compute_decisions": False
}


# # Supply

# In[5]:


import Supply_functions as sup


# In[6]:


links = sup.read_network(NETWORK_PATH)            # Read network xml
vehicle_types =  sup.vehicle_reader(VEHICLE_PATH) # Read vehicle xml


# In[7]:


edges = sup.make_edges_df(links)                # transform to metropolis network
vehicles = sup.make_vehicles_df(vehicle_types)  # transform to metropolis vehicles


# In[34]:


edges_df, vehicles_df = sup.format_supply(edges, vehicles) # format for metropolis input


# # Demand

# In[24]:


import Demand_functions as dmd


# In[10]:


plans_df = dmd.plan_reader_dataframe(PLAN_PATH) # read output_plans.xml


# In[11]:


persons = plans_df[0]    # person_id and agent characteristics (economic status, employment, vehicles, etc.)
plans = plans_df[1]      # plan_id with assigned person_id and score
activities = plans_df[2] # activities (place, duration, etc.) with associated to their respective plan
legs = plans_df[3]       # legs associated to a plan (with mode, departure times and routing mode)
routes = plans_df[4]     # routes, travel times, distances and travel times associated to a leg


# In[12]:


dir(dmd)


# In[13]:


matsim_trips = dmd.generate_sequence(plans, activities, legs, routes) # organize MATSim output


# In[21]:


clean_trips = dmd.summarize_trips(matsim_trips) # filter out trips according to certain criteria


# In[26]:


metro_trips = dmd.generate_trips(clean_trips, edges, vehicles) # create metropolis trips


# In[29]:


agents_df, alts_df, trips_df = dmd.format_demand(metro_trips) # create the metropolis input in the correct format


# # Write Metropolis Input

# ## Parameters
# 

# In[32]:


print("Writing Metropolis parameters")
with open(os.path.join(METRO_INPUT, "parameters.json"), "w") as f:
    f.write(json.dumps(PARAMETERS))


# ## Supply files

# In[35]:


# Writing files
print("Writing Metropolis supply in ", METRO_INPUT)
edges_df.write_parquet(METRO_INPUT + "edges.parquet")
vehicles_df.write_parquet(METRO_INPUT + "vehicles.parquet")


# ## Demand files

# In[36]:


# Writing files
print("Writing Metropolis input to", METRO_INPUT)

print("Writing Metropolis agents")
agents_df.write_parquet(METRO_INPUT + "agents.parquet")

print("Writing Metropolis alternatives")
alts_df.write_parquet(METRO_INPUT + "alts.parquet")

print("Writing Metropolis trips")
trips_df.write_parquet(METRO_INPUT + "trips.parquet")
print("Input files have been successfully written")

