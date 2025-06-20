#!/usr/bin/env python
# coding: utf-8




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




# General directories
general_directory = '/Users/andre/Desktop/Cergy/'

berlin_directory = 'MATSim/matsim-berlin/input/v6.4/'

# Functions
sys.path.append("../functions")


# ## Input files for this script




# MATSim Berlin paths
# Demand
PLAN_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4.output_plans.xml.gz")) # Plans

# this script's output
MATSIM_TRIPS = (os.path.join(general_directory, berlin_directory, "parquet/")) # Plans


# # Demand




import Demand_functions as dmd





plans_df = dmd.plan_reader_dataframe(PLAN_PATH) # read output_plans.xml





persons = plans_df[0]    # person_id and agent characteristics (economic status, employment, vehicles, etc.)
plans = plans_df[1]      # plan_id with assigned person_id and score
activities = plans_df[2] # activities (place, duration, etc.) with associated to their respective plan
legs = plans_df[3]       # legs associated to a plan (with mode, departure times and routing mode)
routes = plans_df[4]     # routes, travel times, distances and travel times associated to a leg


# # Write MATSim `output_plans`




# MATSim files
# Writing files
print("Writing files to", MATSIM_TRIPS)
persons.to_parquet(MATSIM_TRIPS + "MATSim_persons.parquet")
plans.to_parquet(MATSIM_TRIPS + "MATSim_plans.parquet")
activities.to_parquet(MATSIM_TRIPS + "MATSim_activities.parquet")
legs.to_parquet(MATSIM_TRIPS + "MATSim_legs.parquet")
routes.to_parquet(MATSIM_TRIPS + "MATSim_routes.parquet")

print("MATSim files have been successfully written")

