#!/usr/bin/env python
# coding: utf-8


import math
import polars as pl
import xml.etree.ElementTree as ET

from xopen import xopen


def parse_attributes(elem, my_dict):
    for attrib in elem.attrib:
        my_dict[attrib] = elem.attrib[attrib]


# # Read `output_plans`



def plan_reader_dataframe(plan_path, selected_plans_only=True):
    
    """
    Parses a MATSim population XML output file (typically `output_plans`) and converts it into multiple 
    Polars DataFrames representing persons, plans, activities, legs, and routes.

    Parameters
    ----------
    plan_path : str
        Path to the MATSim plans XML file.
    selected_plans_only : bool, optional
        If True (default), only parses plans marked as selected="yes". If False, parses all plans.

    Returns
    -------
    tuple of pl.DataFrame
        A tuple containing the following Polars DataFrames:
            - persons: Information about each person (agent).
            - plans: associated person and score for each selected plan.
            - activities: Activity elements associated with each plan.
            - legs: Leg elements (i.e., trips between activities).
            - routes: Route information (if any) associated with legs.

    Notes
    -----
    - The function performs an iterative XML parsing for memory efficiency.
    - Each plan is associated with one person.
    - Only selected plans are parsed if `selected_plans_only=True`.
    - Custom attributes found within <attribute> tags are included in the respective dictionary.
    - The IDs for plans, activities, legs, and routes are assigned sequentially.
    - Route content is saved under the key 'value' in the routes DataFrame.
    
    """
    
    tree = ET.iterparse(xopen(plan_path), events=["start", "end"])
    persons = []
    plans = []
    activities = []
    legs = []
    routes = []
    current_person = {}
    current_plan = {}
    current_activity = {}
    current_leg = {}
    current_route = {}
    current_person = {}
    current_plan = {}
    current_activity = {}
    current_leg = {}
    current_route = {}

    is_parsing_person = False
    is_parsing_activity = False
    is_parsing_leg = False
    is_selected_plan = True

    current_person_id = None
    current_plan_id = 0
    current_activity_id = 0
    current_leg_id = 0
    current_route_id = 0

    for xml_event, elem in tree:
        if xml_event == "start":
            
            if elem.tag == "person":
                current_person["id"] = elem.attrib["id"]
                current_person_id = elem.attrib["id"]
                is_parsing_person = True
            
            # PLAN
            if elem.tag == "plan":
                is_selected_plan = not selected_plans_only or elem.attrib.get("selected", "no") == "yes"
                if not is_selected_plan:
                    continue
                current_plan["id"] = current_plan_id
                current_plan["person_id"] = current_person_id
                current_plan_id += 1
                parse_attributes(elem, current_plan)

            # ACTIVITY
            elif elem.tag == "activity" and is_selected_plan:
                is_parsing_activity = True
                current_activity_id += 1
                current_activity["id"] = current_activity_id
                current_activity["plan_id"] = current_plan_id-1
                parse_attributes(elem, current_activity)

            # LEG
            elif elem.tag == "leg" and is_selected_plan:
                is_parsing_leg = True
                current_leg_id += 1
                current_leg["id"] = current_leg_id
                current_leg["plan_id"] = current_plan_id-1
                parse_attributes(elem, current_leg)

            # ROUTE
            elif elem.tag == "route" and is_selected_plan:
                current_route_id += 1
                current_route["id"] = current_route_id
                current_route["leg_id"] = current_leg_id
                parse_attributes(elem, current_route)
        
        elif xml_event == "end":
            
        # PERSON
            if elem.tag == "person":
                persons.append(current_person)
                current_person = {}
                is_parsing_person = False

        
        # PLAN 
            elif elem.tag == "plan" and is_selected_plan:
                plans.append(current_plan)
                current_plan = {}
                
         # ACTIVITY
            elif elem.tag == "activity" and is_parsing_activity and is_selected_plan:
                activities.append(current_activity)
                current_activity = {}
                is_parsing_activity = False
                
        # LEG
            elif elem.tag == "leg" and is_parsing_leg and is_selected_plan:
                legs.append(current_leg)
                current_leg = {}
                is_parsing_leg = False
                
        # ROUTE
            elif elem.tag == "route" and is_selected_plan:
                current_route["value"] = elem.text
                routes.append(current_route)
                current_route = {}
                
            elif elem.tag == "attribute":
                attribs = elem.attrib
                if is_parsing_activity and is_selected_plan:
                    current_activity[attribs["name"]] = elem.text
                elif is_parsing_leg and is_selected_plan:
                    current_leg[attribs["name"]] = elem.text
                elif is_parsing_person:
                    current_person[attribs["name"]] = elem.text
            elem.clear()


    # Convert to DataFrames
    return (
        pl.DataFrame(persons),
        pl.DataFrame(plans),
        pl.DataFrame(activities),
        pl.DataFrame(legs),
        pl.DataFrame(routes)
    )


# # Sequence activity - leg - […] - activity

def hhmmss_str_to_seconds_expr(col: str) -> pl.Expr:
    return (
        pl.col(col)
        .map_elements(
            lambda t: sum(x * m for x, m in zip(map(int, str(t).split(":")), [3600, 60, 1]))
            if isinstance(t, str) and ":" in t else None,
            return_dtype=pl.Int32
        )
        .alias(f"{col}_secs")
    )




def generate_sequence (plans, activities, legs, routes):
    """
    Generates a trip-level table from MATSim plans by combining information on activities, legs, and routes.
    
    This function transforms the parsed MATSim XML plan data into a structured trip-level Polars DataFrame
    suitable for further simulation or analysis. 
    It processes time information, identifies trip sequences, computes durations, and determines tour anchors 
    that will define metropolis agents to control for departure times.

    Parameters
    ----------
    plans : pl.DataFrame
        DataFrame containing plan-level information with one row per plan.
    
    activities : pl.DataFrame
        DataFrame containing activity data from the MATSim XML (e.g., type, time, and location info).
    
    legs : pl.DataFrame
        DataFrame containing leg data representing travel between activities, including dep_time and trav_time.
    
    routes : pl.DataFrame
        DataFrame with route strings associated with legs.

    Returns
    -------
    pl.DataFrame
        A cleaned and structured Polars DataFrame of trips with the following information:
            - person_id
            - plan_id
            - tour_id
            - trip_id
            - seq_index
            - mode
            - start_time (in seconds)
            - end_time (in seconds)
            - duration (in seconds) i.e. travel-time
            - route
            - start_link
            - end_link
            - stopping_time (duration of the next activity)

    Notes
    -----
    - Converts all time fields from "HH:MM:SS" strings to seconds.
    - Interleaves activities and legs using a sequence index to create a full chronological order.
    - Reconstructs activity durations when missing based on surrounding leg times.
    - Calculates start and end times for all trip elements.
    - Infers tours using anchor activity types with known end times (e.g., "home").
    - Filters out activities from the final result, returning only leg rows with associated stopping time.
    """
    
    legs = (
        legs
        .join(routes, how='left', left_on='id', right_on='leg_id')
        .with_columns(pl.col('id').alias('leg_id'),
                      hhmmss_str_to_seconds_expr("dep_time"),
                      hhmmss_str_to_seconds_expr("trav_time")
                     )
    )

    activities = (activities
                  .drop(["facility", "initialEndTime", "orig_duration"])
                  .with_columns([
                      hhmmss_str_to_seconds_expr("end_time"),
                      hhmmss_str_to_seconds_expr("max_dur")
                  ])
                 )
    
    # pair seq IDs for activities
    activities = activities.with_columns([
        ((pl.cum_count("plan_id").over("plan_id") - 1) * 2).alias('seq_index'),
        pl.lit('activity').alias('element_type'),
        pl.col('type').alias('type_or_mode'),
        hhmmss_str_to_seconds_expr("max_dur").cast(pl.Float64).alias("duration"),
        pl.col('link').alias('route'),
        pl.col('link').alias('start_link'),
        pl.col('link').alias('end_link')
    ])

    # odd seq IDs for legs
    legs = legs.with_columns([
        ((pl.cum_count("plan_id").over("plan_id") - 1) * 2 + 1).alias('seq_index'),
        pl.lit('leg').alias('element_type'),
        pl.col('mode').alias('type_or_mode'),
        pl.col('trav_time_secs').cast(pl.Float64).alias('duration'),
        pl.col('value').alias('route')
                             ])
    activities_secs = activities.select([
    "plan_id",
    ((pl.cum_count("plan_id").over("plan_id") - 1) * 2).alias("seq_index"),
    "end_time_secs",
    "max_dur_secs",
    pl.lit(None).cast(pl.Int32).alias("dep_time_secs"),
    pl.lit(None).cast(pl.Float64).alias("trav_time_secs")
    ])
    
    legs_secs = (
        legs
        .with_columns(hhmmss_str_to_seconds_expr("trav_time"))
        .select(["plan_id",
                 ((pl.cum_count("plan_id").over("plan_id") - 1) * 2 + 1).alias("seq_index"),
                 pl.lit(None).cast(pl.Int32).alias("end_time_secs"),
                 pl.lit(None).cast(pl.Int32).alias("max_dur_secs"),
                 pl.col("dep_time_secs"),
                 hhmmss_str_to_seconds_expr('trav_time').cast(pl.Float64)]
                 )
    )
    
    
    extra_cols = pl.concat([activities_secs, legs_secs])
    clean_cols = ["plan_id", "seq_index", "element_type", "type_or_mode", "start_link",
                  "end_link", "route", "duration"]

    activities_clean = activities.select(clean_cols)
    legs_clean = legs.select(clean_cols)
    
    matsim_trips = pl.concat([activities_clean, legs_clean]).sort(['plan_id', 'seq_index'])

    matsim_trips = matsim_trips.with_columns([
        # Indicate if activity is not interaction
        ((pl.col('element_type') == 'leg'))
        .cast(pl.Int8).alias('is_trip_start')
    ])

    matsim_trips = (matsim_trips
                    .with_columns([pl.col('is_trip_start').cum_sum().over('plan_id').alias('trip_id')])
                    .drop('is_trip_start')
                    .join(extra_cols, on=["plan_id", "seq_index"], how="left")
        )
    
    # Record start and end times for activities and legs
    matsim_trips = matsim_trips.with_columns([
    pl.col("dep_time_secs").shift(1).alias("prev_leg_dep_secs"),
    pl.col("trav_time_secs").shift(1).alias("prev_leg_trav_secs"),
    ])

    # Activity duration
    matsim_trips = matsim_trips.with_columns([
        pl.when((pl.col("element_type") == "activity") & pl.col("max_dur_secs").is_not_null())
          .then(pl.col("max_dur_secs"))

        .when((pl.col("element_type") == "activity") &
              pl.col("end_time_secs").is_not_null() &
              pl.col("prev_leg_dep_secs").is_not_null() &
              pl.col("prev_leg_trav_secs").is_not_null())
          .then(pl.col("end_time_secs") - (pl.col("prev_leg_dep_secs") + pl.col("prev_leg_trav_secs")))

        .otherwise(None)
        .alias("activity_duration_secs")
    ])
    # Gather "activity_duration" and "travel_time" into a single variable
    matsim_trips = matsim_trips.with_columns([
        pl.when(pl.col("element_type") == "activity")
          .then(pl.col("activity_duration_secs"))
          .when(pl.col("element_type") == "leg")
          .then(pl.col("trav_time_secs"))
          .otherwise(None)
          .alias("duration")
    ])

    # get arrival time for legs
    matsim_trips = matsim_trips.with_columns((pl.col('dep_time_secs')+pl.col('duration')).alias('arrival_time'))
    
    # Start times
    matsim_trips = matsim_trips.with_columns([
        pl.when(pl.col("element_type") == "leg")
          .then(pl.col("dep_time_secs"))

        .when((pl.col("element_type") == "activity") & pl.col("end_time_secs").is_not_null())
          .then(pl.col("end_time_secs") - pl.col("duration"))

        .when((pl.col("element_type") == "activity") & pl.col("prev_leg_dep_secs").is_not_null())
          .then(pl.col("prev_leg_dep_secs") + pl.col("prev_leg_trav_secs"))

        .otherwise(None)
        .alias("start_time_secs")
    ])

    # End times
    matsim_trips = (
        matsim_trips
        .with_columns([
        pl.when(pl.col("element_type") == "leg")
          .then(pl.col("dep_time_secs") + pl.col("duration"))
        .when(pl.col("element_type") == "activity")
          .then(pl.col("start_time_secs") + pl.col("duration"))
        .otherwise(None)
        .alias("end_time_secs")])
        .join(plans.select(['id', 'person_id']), how='left', left_on='plan_id', right_on='id')
        .select(['person_id', "plan_id", "trip_id", "seq_index", "element_type", "type_or_mode", 
                                        "start_time_secs", "end_time_secs", "duration", 
                                        "route", "start_link", "end_link"])    # Select and rearrange variables

    )
    
    # Define tours
    # Record start and end times for activities and legs
    # look for activiy types with an end_time
    tour_anchor_types = (list(set(
        activities.filter(pl.col("end_time").is_not_null())
        .select("type").unique().to_series().to_list()
    )))

    # Add walking legs to separate walking legs in metropolis
    tour_anchor_types = list(set(tour_anchor_types))

    # Create a tour flag
    matsim_trips = matsim_trips.with_columns([
        pl.col("type_or_mode").is_in(tour_anchor_types)
        .alias("is_tour_anchor")
    ])

    # Create tours
    matsim_trips = matsim_trips.with_columns([
        pl.col("is_tour_anchor")
          .cast(pl.Int32)
          .cum_sum()
          .over("plan_id")
          .alias("tour_id")
    ])
    
    # Define stopping times
    stopping_time_df = (
        matsim_trips
        .filter(pl.col("element_type") == "activity")
        .with_columns([
            pl.col("duration").alias("stopping_time")
        ])
        .sort(['plan_id', 'trip_id'])
    )
    
    matsim_trips = (
    matsim_trips
    .filter(pl.col("element_type") == "leg")
    .rename({"start_time_secs":"start_time",
             "end_time_secs": "end_time",
             "type_or_mode":"mode"
            })
    .with_columns([
        # Travel_time per trip
        (pl.col("end_time") - pl.col("start_time")).alias("duration")
    ])
    .select([
        "person_id", "plan_id", "tour_id", "trip_id", "seq_index", "mode", "start_time", "end_time", 
        "duration", "route", "start_link", "end_link"
    ])
    .sort(["plan_id", "trip_id", "tour_id"])
    )
    # Join stopping_time
    matsim_trips = matsim_trips.join(stopping_time_df, on=["plan_id", "trip_id"], how="left")
    
    
    return matsim_trips



def summarize_trips(matsim_trips):
    """
    Cleans and filters trips based on travel duration and stopping time criteria.

    This function scans the `matsim_trips` DataFrame given by the `generate_sequence` function for invalid trips
    and removes all trips in a plan that occur *after* the first invalid one. 
    A trip is considered invalid if:
        - Its stopping_time is negative (invalid activity).

    Parameters
    ----------
    matsim_trips : pl.DataFrame
        A Polars DataFrame of trips generated from MATSim plans, including durations and stopping times.

    Returns
    -------
    pl.DataFrame
        A cleaned Polars DataFrame with only valid trips, dropping those that follow any detected invalid trip 
        within the same plan.

    Notes
    -----
    - If a plan contains one or more invalid trips, all subsequent trips (by trip_id) are excluded.
    - The function joins back to the original data and removes intermediate columns used during filtering.
    """
    invalid_starts = (
        matsim_trips
        .filter((pl.col("stopping_time") < 0))
        .group_by("plan_id")
        .agg(pl.col("trip_id").min().alias("first_invalid_trip"))
    )

    trips_cleaned = (
        matsim_trips
        .join(invalid_starts, on="plan_id", how="left")
        .filter(
            (pl.col("first_invalid_trip").is_null()) |  
            (pl.col("trip_id") < pl.col("first_invalid_trip"))
        )
        .drop("first_invalid_trip", "duration_right", 'route_right', 'start_link_right', 'end_link_right',
              'person_id_right', 'tour_id_right', 'seq_index')
    )
    return trips_cleaned


# # Generate Metropolis input


def generate_trips(matsim_trips, edges, vehicles):
    
    """
    Transforms MATSim-formatted trips into the METROPOLIS-compatible trip format.

    This function takes MATSim trip data and enriches it with vehicle classes, routing information,
    node origins and destinations, and final formatting required for input into the METROPOLIS simulation model.

    Parameters
    ----------
    matsim_trips : pl.DataFrame
        A Polars DataFrame containing processed trip legs and sequences from MATSim plans (`summarize_trips` 
        function).
    edges : pl.DataFrame
        A Polars DataFrame with edge attributes from the MATSim network (`make_edges_df` function).
    vehicles : pl.DataFrame
        A Polars DataFrame describing available vehicle types and their characteristics (`make_vehicles_df` 
        function).

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame in METROPOLIS trip format, including attributes like:
        - agent_id, trip_id
        - class.type, class.origin, class.destination, class.vehicle, class.route, 
        class.travel_time, stopping_time, dt_choice.type, dt_choice.departure_time

    Key Processing Steps
    --------------------
    - Maps vehicle modes from MATSim to METROPOLIS class types (`Road` vs `Virtual`).
    - Converts MATSim link routes to METROPOLIS edge IDs.
    - Adds synthetic stopping times before Road trips to match activity behavior.
    - Adjusts walking trips preceding freight trips to have a minimum duration of 1.
    - Constructs agent identifiers and reshapes columns for METROPOLIS input.

    Notes
    -----
    - `agent_id` is generated using: `agent_id = plan_id * 100 + tour_id`
    - `class.route` is extracted only for Road trips using edge mappings.
    - Assumes `vehicle_id == 3` corresponds to freight agents for the special walking time fix.
    - This function must follow `generate_sequence()`, `make_edges_df()` and `make_vehicles_df` to work correctly.
    """
    
    
    # link (matsim) to edge (metro) dictionary
    matsim_to_metro_links = dict(zip(edges["MATSim_id"].cast(pl.Utf8), edges["edge_id"]))
    
    metro_trips = matsim_trips
    
    # class.vehicle
    metro_trips = (
        metro_trips
        .join(vehicles.select([
            pl.col("vehicle_type").alias("mode"),
            pl.col("vehicle_id").alias("class.vehicle")]),on="mode", how="left")
        .with_columns([

            # class.type
            pl.when(pl.col("mode").is_in(['truck', 'car', 'freight', 'ride']) # define Road trips
                   )
            .then(pl.lit("Road"))
            .otherwise(pl.lit("Virtual"))
            .alias("class.type")])
    )
    
    
    metro_trips = (
        
    # class.type
    metro_trips
    .rename({'start_time':'dt_choice.departure_time'
            })
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
    .drop(['person_id' , 'route', 'duration', 'end_time', 'mode'])
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
            pl.lit("Constant").alias("dt_choice.type"),
            ((pl.col("plan_id")*100).cast(pl.Utf8)+ pl.col("tour_id").cast(pl.Utf8))
            .cast(pl.Int64).alias("agent_id")]) # agent_id ={plan_id*100;tour_id}
    )
    
    # Prep next trip for additional stopping times
    metro_trips = (
        metro_trips
        .with_columns([
            # Get class.type of next trip within each agent
            pl.col("class.type")
            .shift(-1)
            .over("agent_id")
            .alias("next_class_type")])
    )
    
        # Prep next trip for additional stopping times
    metro_trips = (
        metro_trips
        .with_columns([
        # Add +2 to stopping_time if the next trip is of type "Road"
            pl.when(
                pl.col("stopping_time").is_not_null() &
                (pl.col("next_class_type") == "Road")
            ) # +1 for person enters vehicle; +1 for 'vehicle_enters_trafic'
            .then(pl.col("stopping_time") + 2)         
            .otherwise(pl.col("stopping_time"))
            .alias("stopping_time")
        ])
        # Select columns
        .select(['agent_id', 'alt_id', 'trip_id',
                 'class.type', 'class.origin', 'class.destination', 'class.vehicle', 'class.route', 
                 'class.travel_time', 'stopping_time', 'dt_choice.type', 'dt_choice.departure_time'
                ])
    )
    
    # Set the minimal walking leg travel-time before a freight trip to 1 to match `output_events`
    freight_agents = metro_trips.filter(pl.col("class.vehicle") == 3).select("agent_id").unique()
    
    metro_trips = metro_trips.with_columns([
        pl.when(
            pl.col("agent_id").is_in(freight_agents["agent_id"]) &
            pl.col("class.travel_time").is_not_null()
        )
        .then(pl.max_horizontal([pl.col("class.travel_time"), pl.lit(1)]))
        .otherwise(pl.col("class.travel_time"))
        .alias("class.travel_time")
    ])
            
                
    return metro_trips


# # Format


def format_demand(trips):
    """
    Formats the MATSim-derived trips DataFrame into METROPOLIS-compatible demand components.

    This function processes and filters trip records to build the three key demand components 
    required for METROPOLIS simulations: agents, alternatives (alts), and trips. It cleans the 
    input data, assigns default values to required fields, and ensures alignment with the expected schema.

    Parameters
    ----------
    trips : pl.DataFrame
        A Polars DataFrame containing the METROPOLIS-formatted trips from `generate_trips()`.

    Returns
    -------
    agents : pl.DataFrame
        A DataFrame with one row per agent, indicating deterministic choice modeling (e.g., no utility noise).
        Contains fields such as:
        - agent_id
        - alt_choice.type
        - alt_choice.u
        - alt_choice.mu

    alts : pl.DataFrame
        A DataFrame representing each agent’s alternative (one per agent), including departure time,
        utility parameters (mostly set to default or None), and route pre-computation flag.
        Fields include:
        - agent_id, alt_id, dt_choice.*, origin_utility.*, destination_utility.*, etc.

    trips : pl.DataFrame
        A cleaned version of the input trips, ready to be matched with the alternative and agent tables.

    Filtering Logic
    ---------------
    - Removes trips departing after 30 simulation hours (i.e., 108000 seconds).
    - Removes Road-type trips missing a valid origin or destination node.

    Notes
    -----
    - Each agent has exactly one alternative (i.e., deterministic choice).
    - Assumes prior call to `generate_trips()` for schema compatibility.
    - Utility-related fields are initialized but left flexible for model calibration.
    """
    
    # format trips
    # Eliminate trips departing after 48 hours
    trips = trips.filter(pl.col("dt_choice.departure_time") <= 108000,
                         # filter out Road trips having no origin nor destination
                         ~((pl.col("class.type") == "Road") &
                           (pl.col("class.origin").is_null()|pl.col("class.destination").is_null())
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

