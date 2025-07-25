#!/usr/bin/env python
# coding: utf-8

 


import numpy as np
import os
import pyarrow
import sys
import json
import math
import gzip

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import xml.etree.ElementTree as ET

import pandas as pd
import polars as pl


# # Directories

 


sys.path.append("../../functions")
import Demand_functions as dmd
import Supply_functions as sup


 


# General directories
general_directory = '/Users/andre/Desktop/Cergy/'

berlin_directory = 'MATSim/matsim-berlin/input/v6.4/'

run_dir = "Python_Scripts/runs/"


 


# metro
METRO_INPUT = (os.path.join(general_directory, run_dir, "dt_choice/metro_inputs/"))
METRO_OUTPUT = (os.path.join(general_directory, run_dir, "dt_choice/metro_outputs/"))

# Matsim
NETWORK_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4-network.xml.gz"))

VEHICLE_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4-vehicleTypes.xml"))

# Events path is the folder where we exported the output_plans' content (persons, plans, activities, legs, routes)
# If you do not have these files, run 'Read and write MATSim's plans' script
EVENTS_PATH = ((os.path.join(general_directory, berlin_directory, "parquet/")))

ACTIVITY_TYPES_PATH = ((os.path.join(general_directory, berlin_directory, "berlin-v6.4.output_config_reduced.xml")))

# Path to MATSim's experienced plans.
PLAN_PATH = "/Users/andre/Desktop/Cergy/Python_Scripts/runs/fixed_10pct/matsim/"

DEPARTURES_PATH = (os.path.join(general_directory, berlin_directory, "berlin-v6.4.10pct.plans-initial.xml.gz"))

# Path to the directory where the Metropolis output is stored.
#MATSIM_TRIPS = (os.path.join(general_directory, run_dir, "avg_10runs/metro_outputs/"))


 


# Parameters
SEED = 13081996


# # Read Plans

 


def read_matsim_plans():
    persons = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_persons.parquet'))
    plans = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_plans.parquet'))
    activities = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_activities.parquet'))
    legs = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_legs.parquet'))
    routes = pl.read_parquet(os.path.join(PLAN_PATH, 'MATSim_routes.parquet'))

    return persons, plans, activities, legs, routes


 


print("Reading MATSim experienced plans")
persons, plans, activities, legs, routes = read_matsim_plans()

print("Reading MATSim detailed trips")
matsim_trips = pl.read_parquet(os.path.join(EVENTS_PATH, 'MATSim_trips.parquet'))

print("Reading MATSim's activity parameters")
activity_types = dmd.read_activity_parameters(ACTIVITY_TYPES_PATH)


# # Add Travel Cost and Activity Parameters

 


def gen_trips_with_activities(matsim_trips, activity_types, activities):
    
    # Index trips to join trip i with activity j+1
    matsim_trips = matsim_trips.with_columns([pl.arange(0, pl.len()).over("plan_id").alias("activity_index")])
    
    activities_clean = (
    activities
    .join(activity_types, on='type')
    .join(plans, left_on='plan_id', right_on='id')
    .with_columns([
        pl.arange(-1, pl.len()-1).over("plan_id").alias("activity_index"), # Index N-1 actvities to join w/trips
        dmd.hhmmss_str_to_seconds_expr(('typical_duration')),
        dmd.hhmmss_str_to_seconds_expr(('opening_time')),
        dmd.hhmmss_str_to_seconds_expr(('closing_time'))
                  ])
    .select(['person_id','plan_id', 'activity_index', 'id', 'type', 'typical_duration_secs',
             'opening_time_secs', 'closing_time_secs', 'score'])
    )
    
    trips_with_activities = (
    matsim_trips
    .join(activities_clean, on=['plan_id', 'activity_index'], how='left')
    .rename({'type':'following_activity_type',
             'typical_duration_secs': 'typical_duration', 
             'opening_time_secs': 'opening_time',
             'closing_time_secs': 'closing_time'})
    )
    
    trips_with_activities = dmd.mode_utility_params(trips_with_activities)
    
    return trips_with_activities


 


trips_with_activities = gen_trips_with_activities(matsim_trips, activity_types, activities)


# # Generate Metro Input

 


def generate_trips_dt(matsim_trips, edges, vehicles):
    
    # link (matsim) to edge (metro) dictionary
    matsim_to_metro_links = dict(zip(edges["MATSim_id"].cast(pl.Utf8), edges["edge_id"]))
        
    # class.vehicle
    metro_trips = (
        matsim_trips
        .join(vehicles.select([
            pl.col("vehicle_type").alias("mode"),
            pl.col("vehicle_id").alias("class.vehicle")]),on="mode", how="left")
        .with_columns([

            # class.type
            pl.when(pl.col("mode").is_in(['truck', 'car', 'freight', 'ride'])
                   )
            .then(pl.lit("Road"))
            .otherwise(pl.lit("Virtual"))
            .alias("class.type")])
    )
    
    
    metro_trips = (
        
    metro_trips
    .with_columns([
                
        # class.routes
        pl.when(pl.col("class.type") == "Road")
          .then(pl.col("route").str.split(" ") # split route string

                # map in the dictionary
                .map_elements(lambda link_list: None if link_list is None
                              else [matsim_to_metro_links.get(link) for link in link_list[1:]],
                              return_dtype=pl.List(pl.Int64))
                .alias("class.route"))
          .otherwise(None),


        # class.travel_time
        pl.when(pl.col("class.type") == "Road")
          .then(None)
          .otherwise(pl.col("duration"))
        .alias("class.travel_time")
        ])
        .drop(['person_id' , 'route', 'duration', 'end_time', 'mode', 'start_time'])
    )
    
    # Join with edges for start_link's from and to nodes
    metro_trips = (
        metro_trips
        .join(
            edges.select([
                pl.col("MATSim_id").alias("start_link"),
                pl.col("target").alias("class.origin")]), # class.origin
            on="start_link",how="left")
        .drop(pl.col('start_link'))
        .join(
            edges.select([
                pl.col("MATSim_id").alias("end_link"),
                pl.col("target").alias("class.destination")]), # class.destination
            on="end_link", how="left")
        .drop(pl.col('end_link'))
    )
    
    
    metro_trips = (
        metro_trips
        .with_columns([
            pl.lit(1).alias("alt_id"),
            pl.lit("Discrete").alias("dt_choice.type"), ### CHANGED IT FROM THE ORIGINAL FUNCTION
            ((pl.col("plan_id"))).cast(pl.Int64).alias("agent_id")]) # agent_id = plan_id
    )
    
    # Prep next trip for additional stopping times
    metro_trips = (
        metro_trips
        .with_columns([
            # Get class.type of next trip within each agent
            pl.col("class.type")
            .shift(-1)
            .over("agent_id")
            .alias("next_class_type")
        ])
    )
    
        # Prep next trip for additional stopping times
    metro_trips = (
        metro_trips
        .with_columns([
        # Add 2 to stopping_time if the next trip is of type "Road"
            pl.when(
                pl.col("stopping_time").is_not_null() &
                (pl.col("next_class_type") == "Road")
            )
            .then(pl.col("stopping_time") + 2)
            .otherwise(pl.col("stopping_time"))
            .fill_null(strategy='zero')
            .alias("stopping_time"),
            
            # Fill nulls for typical duration set to 0
            #pl.col('typical_duration').fill_null(0),
            
            # alpha = -beta_trav + beta_perf * t_typ/t_dur 
            (pl.when(pl.col('stopping_time') == 0)
             .then(0.0)
             .otherwise(pl.when(pl.col('class.vehicle') == 2)
                        .then(6.88 / 3600 + (6.88 / 3600) * pl.col('typical_duration') / pl.col('stopping_time'))
                        .otherwise((6.88 / 3600) * pl.col('typical_duration') / pl.col('stopping_time'))
                       ))
            .fill_null(strategy='zero')
            .alias('alpha'),
            
            pl.col('C').alias('constant_utility'),
            
            pl.col('opening_time').alias('tstar_inf'),
            (pl.col('closing_time')-pl.col('stopping_time')).fill_null(pl.col('opening_time')).alias('tstar_sup')
            
        ])
        #.with_columns([
        #    pl.when(pl.col('stopping_time').is_in([0,2]))
        #    .then(0.0)
        #    .when(pl.col('stopping_time') >= pl.col('typical_duration')*np.exp(-1))
        #    .then(
        #        6.88 / 3600 * pl.col('typical_duration') *
        #        (1 + (pl.col('stopping_time') / pl.col('typical_duration')).log()))
        #    .otherwise(6.88 / 3600 *(pl.col('stopping_time')*np.e-pl.col('typical_duration')))
        #    .fill_null(0).alias('activity_utility')])
        
        # Select columns
        .select(['agent_id', 'alt_id', 'trip_id',
                 'class.type', 'class.origin', 'class.destination', 'class.vehicle', 'class.route', 
                 'class.travel_time', 'stopping_time', 'dt_choice.type',
                 'typical_duration', 'opening_time', 'closing_time', 'alpha', 'constant_utility', 
                 'daily_monetary_constant', 
                 'tstar_inf', 'tstar_sup' #, 'activity_utility'  # SIX NEW VARIABLES TO CONSIDER
                ])
    )
    

    return metro_trips


 


# Get network and vehicles
links = sup.read_network(NETWORK_PATH)
vehicle_types = sup.vehicle_reader(VEHICLE_PATH)

# Metro conversion for generating agents
edges = sup.make_edges_df(links)
vehicles = sup.make_vehicles_df(vehicle_types)


 


# Generate trips for dep time choice
metro_trips_with_activities = generate_trips_dt(trips_with_activities, edges, vehicles)


# ### Import Activity Utility and Earliest departures

 


# activity utility is obtained with the 
activity_utility = pl.read_parquet(os.path.join(EVENTS_PATH, 'plan_utilities.parquet'))
#initial_departures = dmd.get_initial_dep_times(DEPARTURES_PATH) # 5 minutes
initial_departures = pl.read_parquet(os.path.join(EVENTS_PATH, 'initial_departures.parquet'))

initial_departures = initial_departures.with_columns([dmd.hhmmss_str_to_seconds_expr(('dep_time'))])


 


# dt_choice offset
RNG = np.random.default_rng(SEED)
INTERVAL = 300


 


def format_demand_dt_choice(trips, activity_utility, initial_departures):
    
    # default interval value = 300 representing 5 minute separations for departure time choices 
    
    
    # format trips
    # Eliminate trips departing after 48 hours
    trips = trips.filter(
        #pl.col("dt_choice.departure_time") <= 108000,
                         ~((pl.col("class.type") == "Road") &
                           (pl.col("class.origin").is_null()|pl.col("class.destination").is_null())
                          ))
            
    # format agents
    agents = trips.select("agent_id").unique().with_columns([
        #pl.lit("Deterministic").alias("alt_choice.type"),
        #pl.lit(0.0).alias("alt_choice.u"),
        #pl.lit(None).alias("alt_choice.mu")
    ]).sort("agent_id")

    # format alts
    alts = (
        trips
        #.sort("dt_choice.departure_time")
        .unique(subset=["agent_id"], keep="first")
        .join(activity_utility, left_on='agent_id', right_on='plan_id', how='left')
        .join(initial_departures, left_on='agent_id', right_on='plan_id', how='left')
    )
    
    N = alts.height
    
    alts = (
    alts
        .select([
            "agent_id",
            "alt_id",
            pl.lit(None).alias("origin_delay"),
            "dt_choice.type",
            #"dt_choice.departure_time",
            
            # dt_choice.period
            # We take MATSim's initial dt and create a ±30 min interval around it, emulating MS
            pl.concat_list(
                [pl.max_horizontal(pl.col("dep_time_secs") - 1800, 0), # t0 = max(0, dt-30min)
                 pl.min_horizontal(pl.col("dep_time_secs") + 1800, 86400)] # t1 = min(24h, dt+30min)
            ).alias("dt_choice.period"),
            
            
            # CONSTANT UTILITY (to be revisited)
            (pl.col("daily_monetary_constant") + pl.col("activity_utility")).alias('constant_utility'),
            
            
            pl.lit(300).alias("dt_choice.interval"), # each dt option is spread by 5min
            
            pl.lit('Deterministic').alias("dt_choice.model.type"),
            pl.lit(RNG.uniform(0, 1, size=N)).alias("dt_choice.model.u"),
            pl.lit(RNG.uniform(-INTERVAL / 2, INTERVAL / 2, size=N)).alias("dt_choice.offset"),
                        
            # alpha = -beta_trav + beta_perf * t_typ/t_dur 
            #(pl.when(pl.col('stopping_time') == 0)
            #.then(0.0)
            #.otherwise(pl.when(pl.col('class.vehicle') == 2)
            #            .then(6.88 / 3600 + (6.88 / 3600) * pl.col('typical_duration') / pl.col('stopping_time'))
            #            .otherwise((6.88 / 3600) * pl.col('typical_duration') / pl.col('stopping_time'))
            #           ))
            #.fill_null(strategy='zero').alias('alpha'),
            
            
            # ORIGIN UTILITY
            #pl.when(pl.col('opening_time').is_null() | pl.col('stopping_time').is_null())
            #.then(pl.lit(None))
            #.otherwise(pl.lit('Linear')).alias("origin_utility.type"),
            #pl.col('closing_time').alias("origin_utility.tstar"), # closing time for origin utility
            #pl.lit(0.0).alias("origin_utility.beta"),
            
            #pl.when(pl.col('stopping_time')==0)
            #.then(pl.lit(0.0))
            #.otherwise((6.88/3600)*pl.col('typical_duration')/pl.col('stopping_time')) # beta_perf * t_typ/t_dur
            #.fill_null(strategy='zero').alias("origin_utility.gamma"),
                        
            # DESTINATION UTILITY
            #pl.when(pl.col('opening_time').is_null() | pl.col('stopping_time').is_null())
            #.then(pl.lit(None))
            #.otherwise(pl.lit('Linear')).alias("destination_utility.type"), 
            #pl.col('opening_time').alias("destination_utility.tstar"),
            
            #pl.when(pl.col('stopping_time')==0)
            #.then(pl.lit(0.0))
            #.otherwise((6.88/3600)*pl.col('typical_duration')/pl.col('stopping_time')) # beta_perf * t_typ/t_dur
            #.fill_null(strategy='zero').alias("destination_utility.beta"),
            #pl.lit(0.0).alias("destination_utility.gamma"),

            pl.lit(True).alias("pre_compute_route")
        ])
        .sort("agent_id")
    )

    
    trips = (
        trips
        .with_columns([
            pl.lit(None).alias('class.route'), # ENDOGENOUS ROUTING
            
            pl.when(pl.col('opening_time').is_null() | pl.col('stopping_time').is_null())
            .then(pl.lit(None))
            .otherwise(pl.lit('Linear')).alias('schedule_utility.type'), # Linear AlphaBetaGamma as utility function
            
            
            ((pl.col('tstar_inf')+pl.col('tstar_sup'))/2).alias('schedule_utility.tstar'),
            
            pl.when(pl.col('stopping_time')==0)
            .then(pl.lit(0.0))
            .otherwise((6.88/3600)*pl.col('typical_duration')/pl.col('stopping_time'))
            .fill_null(strategy='zero')
            .alias('schedule_utility.beta'),
            
            pl.when(pl.col('stopping_time')==0)
            .then(pl.lit(0.0))
            .otherwise((6.88/3600)*pl.col('typical_duration')/pl.col('stopping_time'))
            .fill_null(strategy='zero').alias('schedule_utility.gamma'),
            
            pl.when(pl.col('tstar_inf') > pl.col('tstar_sup'))
            .then(pl.lit(0))
            .otherwise(pl.col('tstar_sup')-pl.col('tstar_inf'))
            .alias('schedule_utility.delta')
            
            
            
        ])
        .drop(['dt_choice.type', #, "dt_choice.departure_time"
               'typical_duration', 'closing_time', 'opening_time', 'daily_monetary_constant',
               'tstar_sup', 'tstar_inf'])    
    )

    return agents, alts, trips


 


demand = format_demand_dt_choice(metro_trips_with_activities, activity_utility, initial_departures)


 


agents_df = demand[0]
alts_df = demand[1]
trips_df = demand[2]


# # Write Metropolis Inputs

 


# Writing files
print("Writing Metropolis input to", METRO_INPUT)

print("Writing Metropolis agents")
agents_df.write_parquet(METRO_INPUT + "agents.parquet")

print("Writing Metropolis alternatives")
alts_df.write_parquet(METRO_INPUT + "alts.parquet")

print("Writing Metropolis trips")
trips_df.write_parquet(METRO_INPUT + "trips.parquet")
print("Input files have been successfully written")


 


# supply
supply = sup.format_supply(edges, vehicles)
edges_df = supply[0]
vehicles_df = supply[1]

print("Writing Metropolis supply in ", METRO_INPUT)
edges_df.write_parquet(METRO_INPUT + "edges.parquet")
vehicles_df.write_parquet(METRO_INPUT + "vehicles.parquet")


# ## Write Parameters

 


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
        "backward_wave_speed": 15.0,
        "max_pending_duration": 30.0,
        "constrain_inflow": True,
        "algorithm_type": "Best"
    },
    "learning_model": {
      "type": "Linear"
    },
    "init_iteration_counter": 1,
    "max_iterations": 5,
    "update_ratio": 1.0,
    "random_seed": 13081996,
    "nb_threads": 16,
    "saving_format": "Parquet",
    "only_compute_decisions": False
}


 


# Parameters
print("Writing Metropolis parameters")
with open(os.path.join(METRO_INPUT, "parameters.json"), "w") as f:
    f.write(json.dumps(PARAMETERS))


# ## Debugging
