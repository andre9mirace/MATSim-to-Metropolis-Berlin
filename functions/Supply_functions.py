#!/usr/bin/env python
# coding: utf-8

# In[6]:


import os
import pyarrow
import sys
import math
import mpl_utils

import pandas as pd
import polars as pl
import xml.etree.ElementTree as ET

from xopen import xopen


# In[3]:


def parse_attributes(elem, my_dict):
    for attrib in elem.attrib:
        my_dict[attrib] = elem.attrib[attrib]


# # Read MATSim Supply

# ## Vehicles

# In[4]:


def vehicle_reader(vehcile_path):
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


# ## Network

# In[6]:


def read_network(network_path):
    tree = ET.iterparse(xopen(network_path, "r"), events=["start", "end"])
    links = []
    
    for xml_event, elem in tree:
        
        
                
        if elem.tag == "link" and xml_event == "start":
            atts = elem.attrib
            
            # Remove '#' from link_id
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
            
        # clear the element when we're done, to keep memory usage low
        if elem.tag in ["node", "link"] and xml_event == "end":
            elem.clear()
            
    links = pd.DataFrame.from_records(links)
    links = links.loc[links["modes"].str.contains("car")].copy()
    links["link_id"] = links["link_id"].astype(int)
    links["from_node"] = links["from_node"].astype(int)
    links["to_node"] = links["to_node"].astype(int)
    
    
    
    node_pair_counts = links[["from_node", "to_node"]].value_counts()
    if node_pair_counts.max() > 2:
        print("More than two parallel edges")
        
    parallel_idx = node_pair_counts.loc[node_pair_counts > 1].index
    if len(parallel_idx):
        print("Found {} parallel edges".format(len(parallel_idx)))
        next_node_id = max(links["from_node"].max(), links["to_node"].max()) + 1
        next_link_id = links["link_id"].max() + 1
        new_rows = list()
        for (source, target) in parallel_idx:
            mask = (links["from_node"] == source) & (links["to_node"] == target)
            idx = mask[mask].index
            row = links.loc[idx[1]].copy()
            row["length"] = 0.0
            row["from_node"] = next_node_id
            row["link_id"] = next_link_id
            new_rows.append(row)
            links.loc[idx[1], "to_node"] = next_node_id
            next_link_id += 1
            next_node_id += 1
        links = pd.concat((links, pd.DataFrame(new_rows)))
        
    return links


# Régler question de 3 arcs (//)

# # Inputs METROPOLIS

# ## Vehicles

# In[8]:


def make_vehicles_df(vehicle_types, POPULATION_SHARE=0.1):
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


# ## Edges

# In[19]:


def make_edges_df(links, alpha=1):
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

# In[11]:


def format_supply(edges, vehicles):
    edges = edges.drop(["MATSim_id"])
    vehicles = vehicles.drop(["vehicle_type"])
    
    return [edges, vehicles]

