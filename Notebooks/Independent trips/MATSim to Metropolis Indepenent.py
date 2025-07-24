#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pyarrow
import sys
import json
import math
import mpl_utils
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import polars as pl
import xml.etree.ElementTree as ET

from xopen import xopen


# ## Directories

# In[ ]:


# General directories
general_directory = '/Users/andre/Desktop/Cergy/'

berlin_directory = 'MATSim/matsim-berlin/input/v6.4/'

run_dir = "Python_Scripts/runs/"

# metro
METRO_INPUT = (os.path.join(general_directory, run_dir, "fixed_10pct/metro_inputs/"))
METRO_OUTPUT = (os.path.join(general_directory, run_dir, "fixed_10pct/metro_outputs/"))

# Matsim
NETWORK_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4-network.xml.gz"))

VEHICLE_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4-vehicleTypes.xml"))

PLAN_PATH = ((os.path.join(general_directory, berlin_directory, "berlin-v6.4.output_plans.xml.gz")))


# In[ ]:


sys.path.append("../../functions")
import Demand_functions as dmd
import Supply_functions as sup


# In[ ]:


def parse_attributes(elem, my_dict):
    for attrib in elem.attrib:
        my_dict[attrib] = elem.attrib[attrib]


# # Parameters

# In[ ]:


# Share of the population that is simulated.
POPULATION_SHARE = 0.10

# Parameters to use for the simulation.
PARAMETERS ={
    "input_files": {
      "agents": (os.path.join(METRO_INPUT, "agents.parquet")),
      "alternatives": os.path.join(METRO_INPUT,"alts.parquet"),
      "trips": os.path.join(METRO_INPUT,"trips.parquet"),
      "edges": os.path.join(METRO_INPUT,"edges.parquet"),
      "vehicle_types": os.path.join(METRO_INPUT, "vehicles.parquet")
    },
    "output_directory": METRO_OUTPUT,
    "period": [0.0, 86400.0],
    "road_network": {
        "recording_interval": 950.0,
        "approximation_bound": 1.0,
        "spillback": True,
        "backward_wave_speed": 15.0,
        "max_pending_duration": 30.0,
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
    "nb_threads": 16,
    "saving_format": "Parquet",
    "only_compute_decisions": False
}


# In[ ]:


# Parameters
print("Writing Metropolis parameters")
with open(os.path.join(OUTPUT_DIR, "parameters.json"), "w") as f:
    f.write(json.dumps(PARAMETERS))


# # Supply

# In[ ]:


# Extract MATSim supply
print("Reading MATSim network")
links = sup.read_network(NETWORK_PATH)

print("Reading MATSim vehicles")
vehicle = sup.vehicle_reader(VEHICLE_PATH)

# Generate METRO supply
print("Generating Metropolis network")
edges = sup.make_edges_df(links)
print("Generating Metropolis vehicles")
vehicles = sup.make_vehicles_df(vehicle)

# Formating
edges_df = sup.format_supply(edges, vehicles)[0]
vehicles_df = sup.format_supply(edges, vehicles)[1]

# Writing files
print("Writing files to", OUTPUT_DIR)
edges_df.write_parquet(OUTPUT_DIR + "edges.parquet")
vehicles_df.write_parquet(OUTPUT_DIR + "vehicles.parquet")


# # Demand

# In[ ]:


def generate_trips(plans, legs, routes, edges, vehicles):
    
    # Only selected plans
    plans = plans.loc[plans["selected"] == "yes"].copy()
    
    # Filter by legs corresponding to selected plans
    legs = legs.loc[legs["plan_id"].isin(plans["id"])].copy()
    legs["dep_time"] = pd.to_timedelta(legs["dep_time"]).dt.total_seconds()
    legs["trav_time"] = pd.to_timedelta(routes["trav_time"]).dt.total_seconds()
    
    # "Add" travel time to the legs dataframe
    routes["trav_time"] = pd.to_timedelta(routes["trav_time"]).dt.total_seconds()
    legs = legs.merge(
        routes[["leg_id", "trav_time"]],
        left_on="id",
        right_on="leg_id",
        how="left"
    )
    legs.drop(columns=["leg_id"], inplace=True)
    legs.rename(columns={"trav_time_y": "trav_time"}, inplace=True)
    legs.drop(columns=["trav_time_x"], inplace=True)
    
    # Eliminate pt and walking segments
    legs = legs[(legs['mode'] == 'car') & (legs["routingMode"]=="car")]
    
    
    # Create Metropolis routes by matching links to edges in a dictionary
    matsim_to_metro_routes = dict(zip(edges["MATSim_id"].astype(str), edges["edge_id"]))

    routes["split_links"] = routes.apply(
        lambda row: None if row["type"] in ["generic", "default_pt"] or pd.isnull(row["value"])
        else row["value"].strip().split(),
        axis=1
    )

    routes["class.route"] = routes["split_links"].apply(
        lambda link_list: None if link_list is None
        else [matsim_to_metro_routes.get(link) for link in link_list[1:]]
    )
    
    
    # Transform routes, legs, vehicles and edges to polars
    edges = pl.from_pandas(edges)
    vehicles = pl.from_pandas(vehicles)
    
    routes = (
        pl.from_pandas(routes)
        .filter(pl.col("leg_id").is_in(legs["id"]))
        .with_columns(pl.col("value").str.split(" "))
    )
    legs = (
        pl.from_pandas(legs)
        .with_columns(
            pl.col("dep_time").shift(-1).over("plan_id").alias("next_dep_time"),
            (pl.col("dep_time") + pl.col("trav_time")).alias("arr_time"),
        )
        .with_columns(
            (pl.col("next_dep_time") - pl.col("arr_time")).fill_null(0.0).alias("stopping_time")
        )
    )
    
    # Join legs and routes
    legs = legs.join(routes, left_on="id", right_on="leg_id", how="left") # to get "start and end links"

    # Define trips as startig from the start_link target node and end at the 
    # Join with edges for start_link's from and to nodes
    legs = legs.join(edges.select([
            pl.col("MATSim_id").alias("start_link"),
            pl.col("target").alias("end_source")
        ]),
        on="start_link",
        how="left")
    
    # Join for end_link
    legs = legs.join(edges.select([
            pl.col("MATSim_id").alias("end_link"),
            pl.col("target").alias("end_target")
        ]),
        on="end_link",
        how="left")
    
    # Loop per agent
    all_metro_legs = []
    
    for plan_id in plans["id"]:
        if not plan_id in legs["plan_id"]:
            # Missing plan.
            continue
        
        metro_legs = legs.filter(pl.col("plan_id") == plan_id)
        metro_legs = metro_legs.with_columns(pl.col("id").cast(pl.Int64))
        metro_legs = metro_legs.drop(['id_right', 'trav_time_right', 'distance'])
                

        # class.destination
        metro_legs = metro_legs.join(
            edges.select([pl.col("MATSim_id").alias("end_link"), pl.col("edge_id").alias("class.destination")]),
            on="end_link",
            how="left"
        )
        
        # class.vehicle
        metro_legs = metro_legs.join(
            vehicles.select([
                pl.col("vehicle_type").alias("mode"),
                pl.col("vehicle_id").alias("class.vehicle")]),
            on="mode",
            how="left"
        )
        
        # Condition for class.type = "Virtual"
        virtual_condition = pl.col("type").is_in(["generic", "default_pt"]) | pl.col("value").is_null()

        metro_legs = metro_legs.with_columns([
            
            
            # class.type
            pl.when(virtual_condition)
            .then(pl.lit("Virtual"))
            .otherwise(pl.lit("Road"))
            .alias("class.type"),
            
            # class.origin
            pl.col("end_source").alias("class.origin"),
            
            # class.destination
            pl.col("end_target").alias("class.destination"),
            
            
            # class.travel_time
            pl.when(virtual_condition)
              .then(pl.col("trav_time"))
            .otherwise(None)
            .alias("class.travel_time"),


            
            # stopping_time
            pl.when(pl.col("stopping_time") > 0)
              .then(pl.col("stopping_time"))
              .otherwise(None)
              .alias("stopping_time")
        ])
        
        metro_legs = metro_legs.with_columns([
            pl.lit(1).alias("alt_id"),
            pl.lit("Constant").alias("dt_choice.type"),
            pl.col("dep_time").alias("dt_choice.departure_time")])
        
        metro_legs = metro_legs.drop("dep_time", "value", "routingMode", "trav_time", "vehicleRefId",
                                     "start_link" ,"end_link", "end_source", "end_target")
                
        all_metro_legs.append(metro_legs)
        
    all_metro_legs = pl.concat(all_metro_legs, how="vertical")
    return all_metro_legs


# In[ ]:


def format_demand(trips):
    # format trips
    trips = trips.rename({"id": "trip_id","plan_id": "agent_id"}).select([
        "agent_id", "alt_id", "trip_id", 
        "class.type", "class.origin", "class.destination", "class.vehicle", "class.route", "class.travel_time", "stopping_time", 
        "dt_choice.type", "dt_choice.departure_time"
    ])

    # Format trips (filter out bad Road legs first)
    trips = trips.filter(~((pl.col("class.type") == "Road") &
                           (pl.col("class.origin").is_null() | pl.col("class.destination").is_null())
        ))
            
    # format agents
    agents = trips.select("agent_id").unique().with_columns([
        pl.lit("Deterministic").alias("alt_choice.type"),
        pl.lit(0.0).alias("alt_choice.u"),
        pl.lit(None).alias("alt_choice.mu")
    ]).sort("agent_id")

    # format alts
    alts = (
        trips.sort("dt_choice.departure_time")
        .unique(subset=["agent_id"], keep="first")
        .with_columns([
            pl.lit(1).alias("alt_id"),
            pl.col("dt_choice.departure_time")
        ])
        .select([
            "agent_id",
            "alt_id",
            pl.lit(None).alias("origin_delay"),
            pl.col("dt_choice.type"),
            "dt_choice.departure_time",

            pl.lit(None).alias("dt_choice.interval"),
            pl.lit(None).alias("dt_choice.model.type"),
            pl.lit(0.0).alias("dt_choice.model.u"),
            pl.lit(0.0).alias("dt_choice.model.mu"),
            pl.lit(None).alias("dt_choice.offset"),

            pl.lit(0.0).alias("constant_utility"),
            pl.lit(None).alias("total_travel_utility.one"),
            pl.lit(None).alias("total_travel_utility.two"),
            pl.lit(None).alias("total_travel_utility.three"),
            pl.lit(None).alias("total_travel_utility.four"),

            pl.lit(None).alias("origin_utility.type"),
            pl.lit(0.0).alias("origin_utility.tstar"),
            pl.lit(0.0).alias("origin_utility.beta"),
            pl.lit(0.0).alias("origin_utility.gamma"),
            pl.lit(0.0).alias("origin_utility.delta"),

            pl.lit(None).alias("destination_utility.type"),
            pl.lit(0.0).alias("destination_utility.tstar"),
            pl.lit(0.0).alias("destination_utility.beta"),
            pl.lit(0.0).alias("destination_utility.gamma"),
            pl.lit(0.0).alias("destination_utility.delta"),

            pl.lit(True).alias("pre_compute_route")
        ])
    )
    alts = alts.sort("agent_id")
    
    trips = trips.drop(["dt_choice.type", "dt_choice.departure_time"])

    
    return agents, alts, trips


# # Demand

# In[ ]:


# Run this line if you do not have the MATSim data frames, otherwise, just read them
#plans_df = dmd.plan_reader_dataframe(PLAN_PATH) # read output_plans.xml
#persons = plans_df[0]    # person_id and agent characteristics (economic status, employment, vehicles, etc.)
#plans = plans_df[1]      # plan_id with assigned person_id and score
#activities = plans_df[2] # activities (place, duration, etc.) with associated to their respective plan
#legs = plans_df[3]       # legs associated to a plan (with mode, departure times and routing mode)
#routes = plans_df[4]     # routes, travel times, distances and travel times associated to a leg

def read_matsim_plans():
    persons = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_persons.parquet'))
    plans = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_plans.parquet'))
    activities = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_activities.parquet'))
    legs = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_legs.parquet'))
    routes = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_routes.parquet'))

    return persons, plans, activities, legs, routes


# # Separating all trips as a single agent

# In[ ]:


ind_trips = trips.filter(
    pl.col("class.type") == "Road", # Eliminate Virtual trips
    pl.col("dt_choice.departure_time") <= 108000).with_columns(
    
    pl.lit(1).alias("alt_id"),
    pl.concat_str([
        pl.col("plan_id").cast(pl.Utf8),
        pl.col("alt_id").cast(pl.Utf8).str.zfill(2),
        pl.col("id").cast(pl.Utf8)
    ]).cast(pl.Int64).alias("plan_id")
)


# In[ ]:


# Formating
ind_agents_df = format_demand(ind_trips)[0]
ind_alts_df = format_demand(ind_trips)[1]
ind_trips_df = format_demand(ind_trips)[2]

# Write files
ind_agents_df.write_parquet(OUTPUT_DIR + "agents.parquet")
ind_alts_df.write_parquet(OUTPUT_DIR + "alts.parquet")
ind_trips_df.write_parquet(OUTPUT_DIR + "trips.parquet")
print("Input files have been successfully written")

