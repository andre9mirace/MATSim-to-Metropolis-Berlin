#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import polars as pl
import xml.etree.ElementTree as ET

from xopen import xopen

def parse_attributes(elem, my_dict):
    for attrib in elem.attrib:
        my_dict[attrib] = elem.attrib[attrib]

# # Vehicles


def vehicle_reader(vehcile_path):
    """ 
    This function reads and extracts vehicles from the vehicle_types.xml MATSim input file.
    Its only input argument is the directory path to said file.
    The output is a pandas data frame containing all vehicle attributes given in MATSim.
    """
    
    tree = ET.iterparse(xopen(vehcile_path, "r"), events=["start", "end"])
    vehicle_types = []
    current_vehicle_type = {}
    is_parsing_vehicle_type = False
    for xml_event, elem in tree:
        _, _, elem_tag = elem.tag.partition("}")  # Removing xmlns tag from tag name
        # VEHICLETYPES
        if elem_tag == "vehicleType" and xml_event == "start":
            parse_attributes(elem, current_vehicle_type)
            is_parsing_vehicle_type = True
        # ATTRIBUTES
        elif elem_tag == "attribute" and xml_event == "start":
            current_vehicle_type[elem.attrib["name"]] = elem.text
        # LENGTH / WIDTH
        elif elem_tag in ["length", "width"] and xml_event == "start":
            current_vehicle_type[elem_tag] = elem.attrib["meter"]
        # VEHICLETYPES
        elif elem_tag == "vehicleType" and xml_event == "end":
            vehicle_types.append(current_vehicle_type)
            current_vehicle_type = {}
            elem.clear()
            is_parsing_vehicle_type = False
        # EVERYTHING ELSE
        elif is_parsing_vehicle_type and elem_tag not in ["attribute", "length", "width"]:
            parse_attributes(elem, current_vehicle_type)
    vehicle_types = pd.DataFrame.from_records(vehicle_types)
    col_types = {
        "accessTimeInSecondsPerPerson": float,
        "egressTimeInSecondsPerPerson": float,
        "seats": int,
        "standingRoomInPersons": int,
        "length": float,
        "width": float,
        "pce": float,
        "factor": float,
    }
    for col, dtype in col_types.items():
        if col in vehicle_types.columns:
            try:
                vehicle_types[col] = vehicle_types[col].astype(dtype)
            except:
                print(f"dataframe types conversion failed for column {col}")
    return vehicle_types




def make_vehicles_df(vehicle_types, POPULATION_SHARE=0.1):
    """
    Transforms MATSim vehicle data into a standardized Polars DataFrame format for simulation.

    This function processes the output of the MATSim `vehicle_reader` and constructs a clean
    vehicle attributes table used in further steps of the transformation pipeline.
    The output of thius function returns metropolis-style input data with a bit more detailed 
    information extracted from the MATSim files.

    Parameters
    ----------
    vehicle_types : pandas.DataFrame
        A DataFrame containing MATSim vehicle definitions, including fields like 'id', 'length', and 'pce'.

    POPULATION_SHARE : float, optional
        A scaling factor used to adjust the passenger car equivalent (pce) for vehicle types,
        especially useful when working with a population sample (default is 0.1, i.e., 10%).
        This parameter must match the population sample fraction for a correct simulation.
        Default set to 10% to match the Open Berlin Scenario.

    Returns
    -------
    vehicles : polars.DataFrame
        A Polars DataFrame containing standardized vehicle attributes, including:
        - vehicle_id
        - vehicle_type
        - headway
        - pce (adjusted)
        - speed_function.type (set to "Base")
        - speed_function.upper_bound (None)
        - speed_function.coef (None)

    Notes
    -----
    If the vehicle type is "ride", the pce is set to 0 regardless of the original value.

    Example
    -------
    >>> vehicles_df = make_vehicles_df(vehicle_types)
    >>> print(vehicles_df)
    """
    
    vehicle_list = []

    for idx, row in vehicle_types.iterrows():
        if row["id"] == "ride":
            vehicle = {
                "vehicle_id": idx,
                "vehicle_type": row["id"],
                "headway": float(row["length"]),
                "pce": 0.0,
                "speed_function.type": "Base",
                "speed_function.upper_bound": None,
                "speed_function.coef": None,
            }
        else:
            vehicle = {
                "vehicle_id": idx,
                "vehicle_type": row["id"],
                "headway": float(row["length"]),
                "pce": float(row["pce"]) / POPULATION_SHARE,
                "speed_function.type": "Base",
                "speed_function.upper_bound": None,
                "speed_function.coef": None,
            }
        vehicle_list.append(vehicle)

    vehicles = pl.DataFrame(vehicle_list)
    return vehicles


# # Network



def read_network(network_path):
    """
    Parses and processes a MATSim network XML file into a cleaned and conflict-free pandas DataFrame.

    This function reads the MATSim network XML file (typically `network.xml`), extracts all `link` elements,
    resolves potential issues such as parallel edges (not supported by Metropolis), 
    and returns a network suitable for simulation.

    Parameters
    ----------
    network_path : str
        The file path to the MATSim network XML file.

    Returns
    -------
    links : pandas.DataFrame
        A DataFrame containing only the links that allow car traffic, with columns:
        - link_id : int
        - numeric_link_id : int
        - from_node : int
        - to_node : int
        - length : float
        - freespeed : float
        - capacity : float
        - permlanes : float
        - volume : float

    Notes
    -----
    - Parallel edges (multiple links between the same node pair) are resolved:
        - If >2 edges exist, the second and beyond are redirected through artificial nodes with zero length.
        - If exactly 2 edges exist, one is redirected similarly.

    Example
    -------
    >>> links_df = read_network("data/network.xml")
    >>> print(links_df.head())
    """
    tree = ET.iterparse(xopen(network_path, "r"), events=["start", "end"])
    links = []
    
    for xml_event, elem in tree:
        if elem.tag == "link" and xml_event == "start":
            atts = elem.attrib
            atts["link_id"] = atts["id"].replace("#", "")
            atts["numeric_link_id"] = int(atts["id"].split("#")[0])            
            atts["from_node"] = atts.pop("from")
            atts["to_node"] = atts.pop("to")

            if "cluster" in atts["from_node"]:
                atts["from_node"] = atts["from_node"].replace("cluster_", "").split("_")[0]
            if "cluster" in atts["to_node"]:
                atts["to_node"] = atts["to_node"].replace("cluster_", "").split("_")[0]

            atts["length"] = float(atts["length"])
            atts["freespeed"] = float(atts["freespeed"])
            atts["capacity"] = float(atts["capacity"])
            atts["permlanes"] = float(atts["permlanes"])
            if "volume" in atts:
                atts["volume"] = float(atts["volume"])
            links.append(atts)

        if elem.tag in ["node", "link"] and xml_event == "end":
            elem.clear()

    links = pd.DataFrame.from_records(links)
    links = links.loc[links["modes"].str.contains("car")].copy()
    links["link_id"] = links["link_id"].astype(int)
    links["from_node"] = links["from_node"].astype(int)
    links["to_node"] = links["to_node"].astype(int)

    node_pair_counts = links[["from_node", "to_node"]].value_counts()

    new_rows = []

    if node_pair_counts.max() > 2:
        print("More than two parallel edges")
        print(f"Maximum of {node_pair_counts.max()} parallel edges")

    # Over >2 parallel edges
    parallel_idx_gt2 = node_pair_counts.loc[node_pair_counts > 2].index
    if len(parallel_idx_gt2):
        print(f"Handling {len(parallel_idx_gt2)} node pairs with more than 2 parallel edges...")
        next_node_id = max(links["from_node"].max(), links["to_node"].max()) + 1
        next_link_id = links["link_id"].max() + 1
        for (source, target) in parallel_idx_gt2:
            mask = (links["from_node"] == source) & (links["to_node"] == target)
            idx = mask[mask].index
            for i in range(1, len(idx)):
                row = links.loc[idx[i]].copy()
                row["length"] = 0.0
                row["from_node"] = next_node_id
                row["link_id"] = next_link_id
                new_rows.append(row)
                links.loc[idx[i], "to_node"] = next_node_id
                next_node_id += 1
                next_link_id += 1

    # 2 parallel edges
    parallel_idx = node_pair_counts.loc[node_pair_counts == 2].index
    if len(parallel_idx):
        print(f"Found {len(parallel_idx)} parallel edges")
        next_node_id = max(links["from_node"].max(), links["to_node"].max()) + 1
        next_link_id = links["link_id"].max() + 1
        for (source, target) in parallel_idx:
            mask = (links["from_node"] == source) & (links["to_node"] == target)
            idx = mask[mask].index
            if len(idx) < 2:
                continue  #
            row = links.loc[idx[1]].copy()
            row["length"] = 0.0
            row["from_node"] = next_node_id
            row["link_id"] = next_link_id
            new_rows.append(row)
            links.loc[idx[1], "to_node"] = next_node_id
            next_node_id += 1
            next_link_id += 1

    if new_rows:
        links = pd.concat((links, pd.DataFrame(new_rows)), ignore_index=True)

    return links





def make_edges_df(links, alpha=1):
    """
    Converts a MATSim links DataFrame into a standardized edge table for Metropolis simulation.

    This function builds a Polars DataFrame representing the network edges used in Metropolis also containing,
    enriched with attributes such as bottleneck flow and constant travel time.
    
    This function processes the output of the MATSim `read_network` and constructs a clean
    network table used in further steps of the transformation pipeline.
    The output of thius function returns metropolis 'edges' input data with the added `MATSim_id` for to match
    links (MATSim) and the newly created edges (Metropolis).
    
    Parameters
    ----------
    links : pandas.DataFrame
        A DataFrame containing link data from a MATSim network, with at least the following columns:
        - 'id' : unique identifier for the link
        - 'from_node' : origin node ID
        - 'to_node' : destination node ID
        - 'freespeed' : free-flow speed (m/s)
        - 'length' : link length (meters)
        - 'permlanes' : number of lanes
        - 'capacity' : total capacity (veh/h)

    alpha : float, optional
        Scaling factor applied to the bottleneck flow. Default is 1. This allows for an addition to congestion,
        dividing the edge capacities. Default=1 as not to add extra congestion to the simulation

    Returns
    -------
    edges : polars.DataFrame
        A Polars DataFrame with one row per edge, containing the following fields:
        - edge_id : unique integer identifier
        - MATSim_id : original MATSim link ID
        - source : source node ID (int)
        - target : target node ID (int)
        - speed : free-flow speed (float)
        - length : length of the edge (float)
        - lanes : number of lanes (float)
        - speed_density.* : placeholder fields for congestion modeling (all None)
        - bottleneck_flow : max flow per lane in veh/sec
        - constant_travel_time : fractional leftover from discretizing travel time (float)
        - overtaking : boolean indicating whether overtaking is allowed (default True)

    Notes
    -----
    - `bottleneck_flow` is computed as `capacity / (lanes * 3600) * alpha`.
    - `constant_travel_time` adjusts for rounding error from converting continuous travel time to integer.

    Example
    -------
    >>> edges_df = make_edges_df(links_df)
    >>> print(edges_df.head())
    """
    
    edge_list = []

    for i, (_, row) in enumerate(links.iterrows()):
        edge = {
            "edge_id": i+1,
            "MATSim_id": row["id"],
            "source": int(row["from_node"]),
            "target": int(row["to_node"]),
            "speed": float(row["freespeed"]),
            "length": float(row["length"]),
            "lanes": float(row["permlanes"]),
            "speed_density.type": "FreeFlow",
            "speed_density.capacity": None,
            "speed_density.min_density": None,
            "speed_density.jam_density": None,
            "speed_density.jam_speed": None,
            "speed_density.beta": None,
            "bottleneck_flow": float(row["capacity"])*alpha/ (row['permlanes']*3600.0),  # capacity per lane in vehicles per hour
            "constant_travel_time": math.ceil(float(row["length"]) / float(row["freespeed"])) - float(row["length"]) / float(row["freespeed"]),
            "overtaking": True
        }
        edge_list.append(edge)

    edges = pl.DataFrame(edge_list)
    return edges


# # Format



def format_supply(edges, vehicles):
    """
    Returns the direct supply inputs for a Metropolis simulation.

    This function drops unnecessary columns from the `edges` and `vehicles` data frames from the functions
    `vehicle_reader` and `read_network` to return only the essential fields required by Metropolis.

    Parameters
    ----------
    edges : polars.DataFrame
        DataFrame containing edge attributes, including a column 'MATSim_id' to be removed.

    vehicles : polars.DataFrame
        DataFrame containing vehicle attributes, including a column 'vehicle_type' to be removed.

    Returns
    -------
    list of polars.DataFrame
        A list containing two Polars DataFrames:
        - `edges` with 'MATSim_id' column dropped
        - `vehicles` with 'vehicle_type' column dropped

    Example
    -------
    >>> formatted_edges, formatted_vehicles = format_supply(edges_df, vehicles_df)
    >>> print(formatted_edges.columns)
    >>> print(formatted_vehicles.columns)
    """
    
    edges = edges.drop(["MATSim_id"])
    vehicles = vehicles.drop(["vehicle_type"])
    
    return [edges, vehicles]

