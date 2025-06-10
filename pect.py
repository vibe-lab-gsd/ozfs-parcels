## Script based on code written by Houpu Li

## See https://tidyparcel-documentation.netlify.app/ for full details

###########################
## Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import geopandas as gpd
import math

from shapely.geometry import LineString, MultiLineString, GeometryCollection
from shapely.geometry import box
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.ops import substring
from shapely.ops import linemerge
from shapely.ops import nearest_points
from shapely.ops import split
from shapely import wkt

from scipy.spatial import cKDTree
from rtree import index

from fuzzywuzzy import fuzz

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)


#############################################
## Load data

parcel_path = 'test-data/parcels.geojson'
parcel = gpd.read_file(parcel_path)

# read the zip file, which contains the shapefile
road_path = 'test-data/roads.geojson'
road = gpd.read_file(road_path)

#############################################
## Clean parcel data

# standlize the column names
parcel_cols = {'OBJECTID': 'parcel_id', 'OBJECTID_1': 'parcel_id', 'SITUS_ADDR': 'parcel_addr', 'STAT_LAND_': 'landuse_spec'}

# rename the columns
parcel.rename(columns=lambda x: parcel_cols.get(x, x), inplace=True)

# Define a function to extract only the road name part (before the first comma)
def optimize_road_name(situs_addr):
    if pd.isna(situs_addr) or situs_addr.strip() == ', ,':
        return None
    else:
        return situs_addr.split(',')[0].strip()

# Apply the function to the 'SITUS_ADDR' column
parcel['parcel_addr'] = parcel['parcel_addr'].apply(optimize_road_name)
parcel['parcel_addr'] = parcel['parcel_addr'].replace(r'^\s*$', None, regex=True)

# extract residential area based on specifical landuse
parcel['landuse'] = parcel['landuse_spec'].apply(lambda x: 'R' if isinstance(x, str) and x[0] in ['A', 'B'] else None)

### read the parcel data and data cleanning, 
#        tips: steps 3 and 4 are necessary to remove duplicate geometries and 
#        ensure that the remaining rows contain the required address information.

# Step 1: Transfer the CRS to 4326
parcel = parcel.to_crs(4326)

# Step 2: Create a column to indicate whether 'parcel_addr' or 'landuse' has a value (True/False)
parcel['has_info'] = (~parcel['parcel_addr'].isna()) | (~parcel['landuse'].isna())

# Step 3: Sort the rows by 'has_info' in descending order to prioritize rows with parcel_addr or landuse values
parcel = parcel.sort_values(by='has_info', ascending=False)

# Step 4: Drop duplicates based on geometry, keeping the first occurrence (which now has priority rows at the top)
parcel = parcel.drop_duplicates(subset='geometry')

# Step 5: Drop the 'has_info' column as it's no longer needed
parcel = parcel.drop(columns=['has_info'])

# Step 6: Initialize 'parcel_labeled' column with None values
parcel['parcel_label'] = None
parcel.loc[parcel['parcel_addr'].isna(), 'parcel_label'] = 'parcel without address'
parcel = parcel.reset_index(drop=True)

# Step 7: Extracted the useful columns
parcel = parcel[['Prop_ID','GEO_ID','parcel_id','parcel_addr','landuse','landuse_spec','parcel_label','geometry']]

# Step 8: Group the duplicate parcel_id values and add a suffix.
parcel.loc[parcel['parcel_id'].duplicated(keep=False), 'parcel_id'] = (
    parcel.loc[parcel['parcel_id'].duplicated(keep=False), 'parcel_id'].astype(str) + '_' +
    parcel.loc[parcel['parcel_id'].duplicated(keep=False)].groupby('parcel_id').cumcount().add(1).astype(str)
)

#######################
## cleaning the road data

# standardize the column names
road_cols = {'LINEARID': 'road_id', 'FULLNAME': 'road_addr'}

# rename the columns
road.rename(columns=lambda x: road_cols.get(x, x), inplace=True)

# Step 1: Transfer the CRS to 4326
road = road.to_crs(4326)

# Step 2: Create a column to indicate whether 'road_addr' has a value (True/False)
road['has_info'] = ~road['road_addr'].isna()

# Step 3: Sort the rows by 'has_info' in descending order to prioritize rows with Situs_Addr or RP values
road = road.sort_values(by='has_info', ascending=False)

# Step 4: Drop duplicates based on geometry, keeping the first occurrence (which now has priority rows at the top)
road = road.drop_duplicates(subset='geometry')

# Step 5: Drop the 'has_info' column as it's no longer needed
road = road.drop(columns=['has_info'])
road = road.reset_index(drop=True)
road = road[['road_id','road_addr','geometry']]

# Reset the index to maintain a clean sequential index after the explosion
parcel = parcel.explode(index_parts=False).reset_index(drop=True)
road = road.explode(index_parts=False).reset_index(drop=True)

#####################################################
#####################################################
## Geometry Extraction

# In this step, we decompose each road line into individual segments, which 
# helps facilitate accurate spatial matching in later analysis. In other words, 
# splitting the road lines into smaller segments allows us to precisely assign 
# each parcel boundary to its nearest road segment, improving spatial alignment 
# and analytical precision.

# Initialize lists to store line segments and corresponding addresses
line_strings = []
addrs = []
linear_ids = []

# Iterate over rows in road
for idx, row in road.iterrows():
    line = row['geometry']  # Assume this is a LineString geometry
    addr = row['road_addr']
    linear_id = row['road_id']
    
    if line.is_valid and isinstance(line, LineString):
        for i in range(len(line.coords) - 1):
            current_line = LineString([line.coords[i], line.coords[i + 1]])
            line_strings.append(current_line)
            addrs.append(addr)
            linear_ids.append(linear_id)
    else:
        print(f"Invalid or non-LineString geometry detected: {line}")
        
# Create GeoDataFrame for the split road segments
road_seg = gpd.GeoDataFrame({'geometry': line_strings, 'road_addr': addrs, 'road_id': linear_ids}, crs=road.crs)
road_seg = road_seg.to_crs(3857)

###############################################################
## Address matching

# transfer the crs to projected crs
parcel = parcel.to_crs(3857)
road = road.to_crs(parcel.crs)

# To match the address names between road segments and parcels, we first need to 
# identify the nearest road. However, the nearest road may not always have the 
# exact same name as the parcel’s address. To improve the matching accuracy, we 
# identify the n nearest roads for each parcel (in this case, n = 50). This 
# approach provides multiple candidate road segments for each parcel, increasing 
# the chances of finding a correct match in subsequent processes. The code 
# calculates the centroid coordinates (x, y) of both road segments and parcel 
# geometries. It then uses the cKDTree from scipy to find the 50 nearest roads 
# for each parcel based on their centroid locations. The results will update in 
# the extracted_parcel GeoDataFrame, adding Nearest_Road columns to indicate the 
# closest road names for each parcel. Finally, the temporary x and y columns are 
# dropped to maintain a clean dataset. This process enables efficient spatial 
# address matching between parcels and road segments for further analysis.

# Find the centroid coordinate for each road and parcel
road_seg['x'] = road_seg.geometry.apply(lambda geom: geom.centroid.x)
road_seg['y'] = road_seg.geometry.apply(lambda geom: geom.centroid.y)

parcel['x'] = parcel.geometry.apply(lambda geom: geom.centroid.x)
parcel['y'] = parcel.geometry.apply(lambda geom: geom.centroid.y)

# find the nearest road by using cKDTree
n = 50
tree = cKDTree(road_seg[['x', 'y']])
distances, indices = tree.query(parcel[['x', 'y']], k=n)  # find the nearest n roads

# Create a temporary DataFrame to store the nearest road names
nearest_road_names = pd.DataFrame({
    f'Nearest_Road_{i+1}_Address': road_seg.iloc[indices[:, i]].road_addr.values
    for i in range(n)
})

# Concatenate the new columns with the original DataFrame
parcel = pd.concat([parcel, nearest_road_names], axis=1)

# Drop the x, y column
parcel = parcel.drop(columns=['x', 'y'])

#######################
## Field Similarity Matching using fuzzywuzzy

# The function to find the match address between the n nearst roads segments and parcels
def check_and_extract_match_info(row):
    # Remove spaces from parcel_addr
    parcel_addr = row['parcel_addr'].replace(' ', '').lower()
    
    # Dynamically generate a list of the nearest n road names, check if they are not NaN
    road_names = [row[f'Nearest_Road_{i+1}_Address'].replace(' ', '').lower() 
                  if pd.notna(row[f'Nearest_Road_{i+1}_Address']) else '' 
                  for i in range(n)]
    
    # Define a similarity threshold (e.g., 50%)
    threshold = 50
    best_match = None
    best_similarity = 0
    
    # Check each road name and record match information
    for road in road_names:
        if road:  # Only proceed if the road name is not empty
            # Calculate the similarity score using fuzz.partial_ratio
            similarity = fuzz.partial_ratio(parcel_addr, road)
        
            # Keep track of the best match
            if similarity > best_similarity and similarity >= threshold:
                best_similarity = similarity
                best_match = road
    
    if best_match:
        match_segment = best_match  # Matched road segment
        original_road = row[f'Nearest_Road_{road_names.index(best_match) + 1}_Address']  # Original road name with spaces
        return pd.Series([True, match_segment, original_road])
    
    return pd.Series([False, None, None])  # Return False and None if no match found

# Step 1: Ensure 'parcel_addr' has no NaN values before applying the function
parcel_clean = parcel.loc[parcel['parcel_addr'].notna()].copy()

# Step 2: Apply the check_and_extract_match_info function to add new columns
parcel_clean[['Found_Match', 'match_segment', 'match_road_address']] = parcel_clean.apply(check_and_extract_match_info, axis=1)

# Step 3: Merge the newly created columns back into the original 'parcel' DataFrame
parcel = parcel.merge(parcel_clean[['Found_Match', 'match_segment', 'match_road_address']], 
                                          left_index=True, right_index=True, 
                                          how='left')
                                          
parcel.loc[parcel['Found_Match'] == False, 'parcel_label'] = 'no_match_address'

# Step 4: Count how many rows have 'Found_Match' == False
len(parcel[parcel['Found_Match'] == False])

### Keep useful columns

parcel = parcel[['Prop_ID','GEO_ID','parcel_id','parcel_addr','landuse','landuse_spec','parcel_label','geometry','Found_Match','match_road_address']]

###########################################################
## Explode parcel polygon into parcel edges

# Function to explode Polygons into individual boundary line segments
def explode_to_lines(gdf):
    # Create a list to store new rows
    line_list = []

    for index, row in gdf.iterrows():
        # Get the exterior boundary of the polygon
        exterior = row['geometry'].exterior
        # Convert the boundary into LineString segments
        lines = [LineString([exterior.coords[i], exterior.coords[i + 1]]) 
                 for i in range(len(exterior.coords) - 1)]
        
        # Create new rows for each line segment, retaining the original attributes
        for line in lines:
            new_row = row.copy()
            new_row['geometry'] = line
            line_list.append(new_row)
    
    # Use pd.concat to generate the final GeoDataFrame
    line_gdf = pd.concat(line_list, axis=1).T
    line_gdf = gpd.GeoDataFrame(line_gdf, geometry='geometry', crs=gdf.crs)
    
    return line_gdf

# Call the function to explode the line segments
parcel_seg = explode_to_lines(parcel)

# Reset the index by group
parcel_seg['new_index'] = parcel_seg.groupby('parcel_id').cumcount()
parcel_seg.set_index('new_index', inplace=True)
parcel_seg.index.name = None

###############################################################################
## Calculate angle between lines

# Function to calculate the bearing of a geometry
def fun_bearing_ra(geom):
    coords = np.array(geom.coords)
    # Use the first and last coordinates to calculate the bearing
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    
    # Calculate the bearing using atan2
    bearing = math.atan2(y2 - y1, x2 - x1)
    
    return bearing
  
# Fuction to calculate the angle based on the bearing
def calculate_angle_difference(line1, line2):
    bearing1 = fun_bearing_ra(line1)
    bearing2 = fun_bearing_ra(line2)
    # Calculate the absolute angle difference and ensure it is <= 180 degrees
    delta_theta = bearing2 - bearing1
    
    # Ensure the angle is between -π and π
    if delta_theta > math.pi:
        delta_theta -= 2 * math.pi
    elif delta_theta < -math.pi:
        delta_theta += 2 * math.pi
    
    # Convert the angle to degrees
    angle_between_degrees = math.degrees(abs(delta_theta))
    
    # Return the smaller angle difference (angle or its supplement)
    return min(angle_between_degrees, 180 - angle_between_degrees)
  
# Check if two segments share a common point (i.e., their start or end point is the same)
def are_segments_connected(line1, line2):
    coords1 = np.array(line1.coords)
    coords2 = np.array(line2.coords)
    
    # Check if the start or end points of the segments are the same
    if np.all(coords1[0] == coords2[0]) or np.all(coords1[0] == coords2[-1]) or \
       np.all(coords1[-1] == coords2[0]) or np.all(coords1[-1] == coords2[-1]):
        return True
    return False
  
# Function to reorder segments based on the turning point
def reorder_segments_by_turning_point(segments, turning_point_index):
    # Reorder segments starting from the identified turning point
    reordered_segments = segments[turning_point_index:] + segments[:turning_point_index]
    return reordered_segments
  
# Main function: Process each parcel_id group and return a new GeoDataFrame
def process_parcel_segments(parcel_seg):
    merged_segments = []  # List to store the reordered segments

    # Group the parcel segments by parcel_id and process each group
    for object_id, group in parcel_seg.groupby('parcel_id'):
        segments = group['geometry'].tolist()  # Get the list of line segments for the current group
        original_indices = group.index.tolist()  # Preserve the original indices
        turning_points = []

        # Loop through all adjacent segments to calculate angle differences
        for i in range(1, len(segments)):
            if are_segments_connected(segments[i-1], segments[i]):
                angle_diff = calculate_angle_difference(segments[i-1], segments[i])
                if angle_diff > 15:  # If angle difference is greater than 15 degrees, mark it as a turning point
                    turning_points.append(i)

        # If there are turning points, reorder the segments starting from the first turning point
        if turning_points:
            turning_point_index = turning_points[0]
            reordered_segments = reorder_segments_by_turning_point(segments, turning_point_index)
            reordered_original_indices = reorder_segments_by_turning_point(original_indices, turning_point_index)
        else:
            # If no turning points, retain the original order
            reordered_segments = segments
            reordered_original_indices = original_indices

        # Store the reordered segments and their attributes
        for j, (line, original_index) in enumerate(zip(reordered_segments, reordered_original_indices)):
            row = group.iloc[0].copy()  # Copy the first row's attributes
            row['geometry'] = line
            row['original_index'] = original_index  # Preserve the original index
            row['new_index'] = j  # Assign the new index based on the reordered list
            merged_segments.append(row)

    # Create a new GeoDataFrame for the reordered segments
    updated_gdf = gpd.GeoDataFrame(merged_segments, columns=parcel_seg.columns.tolist() + ['original_index', 'new_index'])
    updated_gdf = updated_gdf.reset_index(drop=True)

    return updated_gdf

# Run the main function and get the new GeoDataFrame
updated_parcel_seg = process_parcel_segments(parcel_seg)
parcel_seg = updated_parcel_seg

# Group parcel_seg by parcel_id and process each group
merged_segments = []

for object_id, group in parcel_seg.groupby('parcel_id'):
    # Get the list of geometries in the current group
    segments = group.geometry.tolist()
    # Start with the first segment
    merged_lines = [segments[0]]  # Start with the first segment
    
    for i in range(1, len(segments)):
        connected = False
        
        # Always compare the current segment with the previous one
        if are_segments_connected(segments[i-1], segments[i]):
            # Calculate the angle difference between the current segment and the previous one
            angle_diff = calculate_angle_difference(segments[i-1], segments[i])
            
            # If the angle difference is less than 15 degrees, merge the adjacent line segments
            if angle_diff < 15:
                # Merge the current and previous segments
                merged_result = linemerge([merged_lines[-1], segments[i]])
                
                # Check if the result is a MultiLineString, if so, skip the merge
                if isinstance(merged_result, LineString):
                    merged_lines[-1] = merged_result
                    connected = True
                else:
                    # Skip the merge if it's a MultiLineString
                    continue
        
        # If no connected segment is found or the angle difference is too large, add the current segment as a new one
        if not connected:
            merged_lines.append(segments[i])
    
    # Keep the merged results and add other attributes
    for line in merged_lines:
        row = group.iloc[0].copy()  # Copy the first attribute row from the group
        row['geometry'] = line
        merged_segments.append(row)

# Create a new GeoDataFrame from the merged line segments
parcel_seg = gpd.GeoDataFrame(merged_segments, columns=parcel_seg.columns)

# Check for MultiLineString geometries and explode them into LineString
exploded_segments = []

for index, row in parcel_seg.iterrows():
    geom = row['geometry']
    
    if isinstance(geom, MultiLineString):
        # Explode the MultiLineString into individual LineStrings
        for line in geom:
            new_row = row.copy()
            new_row['geometry'] = line
            exploded_segments.append(new_row)
    else:
        # Keep the original LineString geometries
        exploded_segments.append(row)

# Create a new GeoDataFrame from the exploded segments
parcel_seg = gpd.GeoDataFrame(exploded_segments, columns=parcel_seg.columns)
# extract useful columns
parcel_seg.drop(columns=['original_index', 'new_index'], inplace=True)
# Reset the index of the final GeoDataFrame
parcel_seg = parcel_seg.reset_index(drop=True)

##################################################################
## Divide curved sides into two segments

# Divide curves with three edges into two segments at the midpoint of the angle 
# change (>30°) and combine with unchanged segments for final output.

edge_counts = parcel_seg.groupby('parcel_id').size()
parcel_seg['edge_num'] = parcel_seg['parcel_id'].map(edge_counts)

# Function to create tangent lines at both ends of a line segment
def create_tangents(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Create tangents at the start and end of the line segment
    start_tangent = LineString([coords[0], coords[1]])
    end_tangent = LineString([coords[-2], coords[-1]])
    
    return start_tangent, end_tangent

# Function to filter curve segments based on angle difference of tangents > 30 degrees
def filter_curve_segments(parcel_seg, angle_threshold=30):
    filtered_segments = []
    non_filtered_segments = []
    
    for idx, row in parcel_seg.iterrows():
        line = row['geometry']
        start_tangent, end_tangent = create_tangents(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            row_dict = row.to_dict()  # Convert the entire row to a dictionary
            row_dict['index'] = idx  # Preserve the original index
            
            if angle_diff > angle_threshold:
                # Add the entire row to the filtered list
                filtered_segments.append(row_dict)
            else:
                # Add the entire row to the non-filtered list
                non_filtered_segments.append(row_dict)
    
    # Create DataFrames with the filtered and non-filtered results if data exists
    if filtered_segments:
        filtered_df = pd.DataFrame(filtered_segments).set_index('index')
        filtered_gdf = gpd.GeoDataFrame(filtered_df, crs=parcel_seg.crs, geometry=filtered_df['geometry'])
    else:
        # Initialize an empty GeoDataFrame with the same structure if no data
        filtered_gdf = gpd.GeoDataFrame(columns=parcel_seg.columns, crs=parcel_seg.crs)
    
    if non_filtered_segments:
        non_filtered_df = pd.DataFrame(non_filtered_segments).set_index('index')
        non_filtered_gdf = gpd.GeoDataFrame(non_filtered_df, crs=parcel_seg.crs, geometry=non_filtered_df['geometry'])
    else:
        # Initialize an empty GeoDataFrame with the same structure if no data
        non_filtered_gdf = gpd.GeoDataFrame(columns=parcel_seg.columns, crs=parcel_seg.crs)
    
    return filtered_gdf, non_filtered_gdf

# Call the function to filter curve segments and create two GeoDataFrames
filtered_parcel_seg, non_filtered_parcel_seg = filter_curve_segments(parcel_seg[parcel_seg['edge_num'] == 3])

# %%
# Function to create tangent lines and reverse the line if necessary
def create_tangents_with_reversal(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Find the points with the smallest and largest y-coordinate (latitude)
    if coords[0][1] < coords[-1][1]:  # If the first point's y is smaller, it's the start point
        start_point = coords[0]
        end_point = coords[-1]
    else:  # Otherwise, the last point is the start point
        start_point = coords[-1]
        end_point = coords[0]

    # Reverse the line if start_point is not the same as coords[0]
    if start_point != coords[0]:
        coords.reverse()  # Reverse the order of coordinates
    
    # Now create tangents based on the (possibly reversed) coordinates
    start_tangent = LineString([coords[0], coords[1]])  # Tangent from the first to the second point
    end_tangent = LineString([coords[-2], coords[-1]])  # Tangent from the second last to the last point

    return start_tangent, end_tangent, LineString(coords)  # Return the tangents and the (possibly reversed) line

# Function to calculate the split point based on the half rule
def calculate_split_point(line, start_tangent, end_tangent, angle_diff, angle_fraction=0.5):
    coords = list(line.coords)

    # Iterate through the line and find the point where the angle difference is approximately half
    for i in range(1, len(coords) - 1):
        intermediate_tangent = LineString([coords[i - 1], coords[i]])
        current_angle_diff = calculate_angle_difference(start_tangent, intermediate_tangent)
        
        if current_angle_diff >= angle_diff * angle_fraction:
            return coords[i]  # Return the split point

    return coords[-1]  # If no point found, return the endpoint

# Function to process each segment in filtered_parcel_seg
def process_filtered_parcel_seg(filtered_parcel_seg, angle_threshold=30, angle_fraction=0.5):
    new_data = []
    
    for idx, row in filtered_parcel_seg.iterrows():
        line = row['geometry']
        
        # Apply the tangent and reversal function
        start_tangent, end_tangent, adjusted_line = create_tangents_with_reversal(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            
            if angle_diff > angle_threshold:
                # Calculate the split point based on the angle difference and fraction
                split_point = calculate_split_point(adjusted_line, start_tangent, end_tangent, angle_diff, angle_fraction)
                
                # Add split point to row's data
                row_dict = row.to_dict()
                row_dict['split_point'] = Point(split_point)  # Store the split point as geometry
                row_dict['index'] = idx  # Store the original index
                
                new_data.append(row_dict)
            else:
                # If no split needed, just keep the original row
                row_dict = row.to_dict()
                row_dict['split_point'] = None  # No split point, store None
                row_dict['index'] = idx  # Store the original index
                
                new_data.append(row_dict)

    # Convert the processed data back into a GeoDataFrame
    new_df = pd.DataFrame(new_data).set_index('index')  # Use original index
    new_gdf = gpd.GeoDataFrame(new_df, crs=parcel_seg.crs, geometry='split_point')
    
    return new_gdf

# Check if filtered_parcel_seg is non-empty before processing
if not filtered_parcel_seg.empty:
    # Call the function to process the filtered_parcel_seg
    processed_parcel_seg = process_filtered_parcel_seg(filtered_parcel_seg)
else:
    # Handle the case where filtered_parcel_seg is empty
    processed_parcel_seg = gpd.GeoDataFrame(columns=filtered_parcel_seg.columns, crs=parcel_seg.crs)

# %%
# Function to split filtered_parcel_seg using points from processed_parcel_seg
def split_lines_with_points(filtered_parcel_seg, processed_parcel_seg):
    split_segments = []

    for idx, row in filtered_parcel_seg.iterrows():
        line = row['geometry']
        split_point_geom = processed_parcel_seg.loc[idx, 'split_point']  # Get the corresponding point geometry from split_point column
        
        if isinstance(split_point_geom, Point):
            # Check if the split point is on the line
            if line.contains(split_point_geom):
                # If the point is on the line, use it directly for splitting
                split_lines = split(line, split_point_geom)
            else:
                # If the point is not on the line, find the closest point on the line
                projected_distance = line.project(split_point_geom)
                nearest_point = line.interpolate(projected_distance)
                split_lines = split(line, nearest_point)
            
            # Handle GeometryCollection by extracting valid LineString geometries
            if isinstance(split_lines, GeometryCollection):
                split_segments.extend([{
                    **row.to_dict(), 'geometry': geom
                } for geom in split_lines.geoms if isinstance(geom, LineString)])
                continue  # Skip to the next iteration

        # If no valid split point or GeometryCollection, add the original row
        split_segments.append(row.to_dict())
    
    # Convert split_segments to a GeoDataFrame and return
    split_gdf = gpd.GeoDataFrame(split_segments, crs=parcel_seg.crs, geometry='geometry')
    return split_gdf

# Check if both filtered_parcel_seg and processed_parcel_seg are non-empty before processing
if not filtered_parcel_seg.empty and not processed_parcel_seg.empty:
    # Call the function to split lines based on points
    split_parcel_seg = split_lines_with_points(filtered_parcel_seg, processed_parcel_seg)
else:
    # Handle the case where one or both GeoDataFrames are empty
    split_parcel_seg = gpd.GeoDataFrame(columns=filtered_parcel_seg.columns, crs=parcel_seg.crs)

# Function to combine split_parcel_seg and non_filtered_parcel_seg, ensuring parcel_id proximity
def combine_parcel_segs(split_parcel_seg, non_filtered_parcel_seg):
    # Ensure both datasets contain the 'parcel_id' column
    if 'parcel_id' not in split_parcel_seg.columns or 'parcel_id' not in non_filtered_parcel_seg.columns:
        raise ValueError("Both datasets must contain the 'parcel_id' column.")
    
    # Convert parcel_id to string to avoid type errors during sorting
    split_parcel_seg['parcel_id'] = split_parcel_seg['parcel_id'].astype(str)
    non_filtered_parcel_seg['parcel_id'] = non_filtered_parcel_seg['parcel_id'].astype(str)
    
    # Concatenate the two GeoDataFrames and ensure 'crs' and 'geometry' are set
    combined_parcel_seg = gpd.GeoDataFrame(
        pd.concat([split_parcel_seg, non_filtered_parcel_seg], ignore_index=True),
        crs=parcel_seg.crs,  # Use the crs from one of the input GeoDataFrames
        geometry='geometry'  # Ensure the geometry column is correctly set
    )
    
    # Sort by 'parcel_id' to ensure similar parcel_id are together
    combined_parcel_seg_sorted = combined_parcel_seg.sort_values(by='parcel_id')
    
    return combined_parcel_seg_sorted

# Check if both split_parcel_seg and non_filtered_parcel_seg are non-empty before processing
if not split_parcel_seg.empty and not non_filtered_parcel_seg.empty:
    # Call the function to combine the datasets
    reconstr_seg = combine_parcel_segs(split_parcel_seg, non_filtered_parcel_seg)
else:
    # Handle the case where one or both GeoDataFrames are empty
    reconstr_seg = gpd.GeoDataFrame(columns=split_parcel_seg.columns, crs=parcel_seg.crs)

# %%
# Check if reconstr_seg is non-empty before concatenating
if not reconstr_seg.empty:
    parcel_seg = pd.concat([parcel_seg[parcel_seg['edge_num'] != 3], reconstr_seg], ignore_index=True).reset_index(drop=True)

parcel_seg = parcel_seg.drop(columns=['edge_num'])

###############################################################################
# Classify parcels. 

# Determining how to classify parcel edges (and whether they can be classified)
# dependts on whether the parcel is classified in to one of 9 types.

############
#### Type 1: Duplicated address
# Identify duplicated parcel_addr values
duplicated_ids = parcel[parcel['parcel_addr'].notna() & parcel['parcel_addr'].duplicated(keep=False)]

# updated those duplicated parcel_addr rows and lable them in the 'parcel_label' column
parcel.loc[parcel['parcel_addr'].isin(duplicated_ids['parcel_addr']), 'parcel_label'] = 'duplicated address'

################
#### Type 2: Jagged
# Perimeter-Area Ratio (Shape Index)
parcel['shape_index'] = parcel['geometry'].length / (2 * (3.14159 * parcel['geometry'].area)**0.5)

si_threshold = 0.50
column_name = f"{int(si_threshold * 100)}_threshold"
parcel[column_name] = parcel['shape_index'] > parcel['shape_index'].quantile(si_threshold)

# Ensure the geometry is a Polygon type and calculate the number of edges
edge_count = parcel_seg.groupby('parcel_id').size().reset_index(name='num_edges')
parcel = parcel.merge(edge_count, on='parcel_id', how='left')

parcel['parcel_label'] = parcel.apply(
    lambda row: 'jagged parcel' if pd.isna(row['parcel_label']) and row[column_name] and row['num_edges'] >= 6 else row['parcel_label'], 
    axis=1
)

#################
#### Type 3: and 4 Regular inside and corner parcels

# parcel_seg_filter = parcel_seg[(parcel_seg['RP'] == 'R') & (parcel_seg['match_road_address'].notnull())]
parcel_seg_filter = parcel_seg[parcel_seg['match_road_address'].notnull()]

# Initialize lists to store the matched road geometries and distances
matched_road_geometries = []
midpoint_distances = []

# Iterate over each row in parcel_seg_filter
for idx, parcel_row in parcel_seg_filter.iterrows():
    match_addr = parcel_row['match_road_address']
    
    # Calculate the midpoint (centroid) of the parcel geometry
    midpoint = parcel_row.geometry.centroid
    # Filter road_seg to get rows where road_addr matches match_road_address
    matching_road_segs = road_seg[road_seg['road_addr'] == match_addr]
    
    if not matching_road_segs.empty:
        # Calculate distances between the midpoint of the parcel and matching road_seg geometries
        distances = matching_road_segs.geometry.apply(lambda geom: midpoint.distance(geom))
        # Find the index of the nearest road geometry
        nearest_index = distances.idxmin()
        # Append the nearest road geometry to the list
        matched_road_geometries.append(matching_road_segs.loc[nearest_index].geometry)
        # Append the corresponding distance (from midpoint to nearest road) to the list
        midpoint_distances.append(distances[nearest_index])
    else:
        # If no match is found, append None for both geometry and distance
        matched_road_geometries.append(None)
        midpoint_distances.append(None)
        
# Add the matched road geometries and midpoint distances to parcel_seg_filter
parcel_seg_filter['road_geometry'] = matched_road_geometries
parcel_seg_filter['midpoint_distance_to_road'] = midpoint_distances

###########################
## Type 5: Cul-de-sac

# Step 1: Identify end points

def identify_end_points(road_seg):
    # Get the start and end points of each road segment
    road_seg['start_point'] = road_seg['geometry'].apply(lambda geom: Point(geom.coords[0]))
    road_seg['end_point'] = road_seg['geometry'].apply(lambda geom: Point(geom.coords[-1]))
    
    # Create GeoDataFrames for start and end points, including road_addr and road_id
    start_points = road_seg[['road_addr', 'road_id', 'start_point']].rename(columns={'start_point': 'point'})
    end_points = road_seg[['road_addr', 'road_id', 'end_point']].rename(columns={'end_point': 'point'})
    
    # Concatenate start and end points into a single GeoDataFrame
    all_points = gpd.GeoDataFrame(pd.concat([start_points, end_points]), geometry='point')
    
    # Count how many times each point appears (indicating road connections)
    point_counts = all_points.groupby('point').size().reset_index(name='count')
    
    # Filter points that appear only once (end points connected to a single road)
    end_points = point_counts[point_counts['count'] == 1]
    
    # Merge back road_addr and road_id to end points
    end_points = pd.merge(end_points, all_points[['road_addr', 'road_id', 'point']], on='point', how='left')
    
    # Convert to a GeoDataFrame, using 'point' as the geometry
    end_points = gpd.GeoDataFrame(end_points, geometry='point', crs=road_seg.crs)
    
    return end_points

# Use the function to get endpoints that connect to only one road, including road_addr and road_id
end_points = identify_end_points(road_seg)

# Create a buffer for the end road segments (35m)
end_points_buffer = gpd.GeoDataFrame(end_points.copy(), geometry=end_points.geometry.buffer(35), crs=end_points.crs)
# Drop the 'point' column if no longer needed
end_points_buffer.drop(columns=['point'], inplace=True)

# Step 2: Match potential cul-de-sac parcels
# Check if each parcel in 'parcel' intersects with 'road_seg_end_buffer'
intersections = gpd.sjoin(parcel, end_points_buffer, how="inner", predicate="intersects")
intersections = intersections.set_crs(crs=parcel.crs)

# Filter parcels where the 'match_road_address' matches the 'road_addr' in 'road_seg_end_buffer'
filtered_parcels_seg = intersections[intersections['match_road_address'] == intersections['road_addr']]

# Step 3: label cul-de-sab parcels

def label_end_road_parcels(parcel, filtered_parcels_seg, filtered_object_ids):
    # Create a mask to identify rows in parcel that match parcel_id and parcel_addr
    matching_rows = parcel.merge(
        filtered_parcels_seg[['parcel_id', 'parcel_addr']], 
        on=['parcel_id', 'parcel_addr'], 
        how='inner'
    )
    
    # Update the 'parcel_label' column to 'end_road_parcel' for matched rows
    # Only if num_edges > 4 or parcel_id is in filtered_object_ids
    parcel.loc[
        (parcel['parcel_id'].isin(matching_rows['parcel_id']) & 
         parcel['parcel_addr'].isin(matching_rows['parcel_addr'])) & 
        ((parcel['num_edges'] > 4) | 
         parcel['parcel_id'].isin(filtered_object_ids)), 
        'parcel_label'
    ] = 'cul_de_sac parcel'
    
    return parcel

# Call the function to label the end_road_parcel rows only if both filtered_parcels_seg and filtered_object_ids are non-empty
if not filtered_parcels_seg.empty and len(filtered_object_ids) > 0:
    parcel = label_end_road_parcels(parcel, filtered_parcels_seg, filtered_object_ids)
    
###############
#### Type 6: Curved parcels

def label_special_parcels(parcel, filtered_parcel_seg):
    # Create a mask to identify rows in parcel that match parcel_id and parcel_addr
    matching_rows = parcel.merge(
        filtered_parcel_seg[['parcel_id', 'parcel_addr']], 
        on=['parcel_id', 'parcel_addr'], 
        how='inner'
    )
    
    # Update the 'parcel_label' column to 'special parcel' 
    # only for rows where 'parcel_label' is null
    parcel.loc[
        parcel['parcel_label'].isnull() &  # Check for null values
        parcel['parcel_id'].isin(matching_rows['parcel_id']) &
        parcel['parcel_addr'].isin(matching_rows['parcel_addr']), 
        'parcel_label'
    ] = 'curve parcel'
    
    return parcel

# Call the function to label special parcels only if filtered_parcel_seg is non-empty
if not filtered_parcel_seg.empty:
    parcel = label_special_parcels(parcel, filtered_parcel_seg)
    
####################
### Type 7: Special parcels

parcel['parcel_label'] = parcel['parcel_label'].fillna('special parcel')


##############################################################################
##############################################################################
## Label parcel edges

# Get the centroid or representative points of the road segments
road_centroids = np.array([geom.centroid.coords[0] for geom in road_seg.geometry])
# Build the KDTree based on the centroids of the road segments
road_tree = cKDTree(road_centroids)

# Initialize a list to store the matched road geometries
matched_road_geometries = []

# Iterate over each row in parcel
for idx, parcel_row in parcel.iterrows():
    # Check if Found_Match is True
    if parcel_row['Found_Match'] == True:
        match_addr = parcel_row['match_road_address']
        # Filter road_seg to get rows where road_addr matches match_road_address
        matching_road_segs = road_seg[road_seg['road_addr'] == match_addr]
        
        if not matching_road_segs.empty:
            # Calculate distances between the parcel polygon geometry and matching road_seg geometries
            distances = matching_road_segs.geometry.apply(lambda geom: parcel_row.geometry.distance(geom))
            
            # Find the index of the nearest road geometry
            nearest_index = distances.idxmin()
            
            # Append the nearest road geometry to the list
            matched_road_geometries.append(matching_road_segs.loc[nearest_index].geometry)
        else:
            # If no match is found, append None or an empty geometry
            matched_road_geometries.append(None)
    else:
        # If Found_Match is False or NaN, find the nearest road geometry
        # Get the centroid of the current parcel polygon
        parcel_centroid = np.array(parcel_row.geometry.centroid.coords[0])
        
        # Query the KDTree for the nearest road segment
        _, nearest_index = road_tree.query(parcel_centroid)
        
        # Append the nearest road geometry to the list
        matched_road_geometries.append(road_seg.iloc[nearest_index].geometry)
        
# Add the matched road geometries to parcel
parcel['road_geometry'] = matched_road_geometries

# %%
# Function to explode Polygons into individual boundary line segments
def explode_to_lines(gdf):
    # Create a list to store new rows
    line_list = []

    for index, row in gdf.iterrows():
        # Get the exterior boundary of the polygon
        exterior = row['geometry'].exterior
        # Convert the boundary into LineString segments
        lines = [LineString([exterior.coords[i], exterior.coords[i + 1]]) 
                 for i in range(len(exterior.coords) - 1)]
        
        # Create new rows for each line segment, retaining the original attributes
        for line in lines:
            new_row = row.copy()
            new_row['geometry'] = line
            line_list.append(new_row)
    
    # Use pd.concat to generate the final GeoDataFrame
    line_gdf = pd.concat(line_list, axis=1).T
    line_gdf = gpd.GeoDataFrame(line_gdf, geometry='geometry', crs=gdf.crs)
    
    return line_gdf

# Call the function to explode the line segments
parcel_seg = explode_to_lines(parcel)

# Reset the index by group
parcel_seg['new_index'] = parcel_seg.groupby('parcel_id').cumcount()
parcel_seg.set_index('new_index', inplace=True)
parcel_seg.index.name = None


# Function to calculate the bearing of a geometry
def fun_bearing_ra(geom):
    coords = np.array(geom.coords)
    # Use the first and last coordinates to calculate the bearing
    x1, y1 = coords[0]
    x2, y2 = coords[-1]
    
    # Calculate the bearing using atan2
    bearing = math.atan2(y2 - y1, x2 - x1)
    
    return bearing

def calculate_angle_difference(line1, line2):
    bearing1 = fun_bearing_ra(line1)
    bearing2 = fun_bearing_ra(line2)
    # Calculate the absolute angle difference and ensure it is <= 180 degrees
    delta_theta = bearing2 - bearing1
    
    # Ensure the angle is between -π and π
    if delta_theta > math.pi:
        delta_theta -= 2 * math.pi
    elif delta_theta < -math.pi:
        delta_theta += 2 * math.pi
    
    # Convert the angle to degrees
    angle_between_degrees = math.degrees(abs(delta_theta))
    
    # Return the smaller angle difference (angle or its supplement)
    return min(angle_between_degrees, 180 - angle_between_degrees)


# Check if two segments share a common point (i.e., their start or end point is the same)
def are_segments_connected(line1, line2):
    coords1 = np.array(line1.coords)
    coords2 = np.array(line2.coords)
    
    # Check if the start or end points of the segments are the same
    if np.all(coords1[0] == coords2[0]) or np.all(coords1[0] == coords2[-1]) or \
       np.all(coords1[-1] == coords2[0]) or np.all(coords1[-1] == coords2[-1]):
        return True
    return False

# Function to reorder segments based on the turning point
def reorder_segments_by_turning_point(segments, turning_point_index):
    # Reorder segments starting from the identified turning point
    reordered_segments = segments[turning_point_index:] + segments[:turning_point_index]
    return reordered_segments

# Main function: Process each parcel_id group and return a new GeoDataFrame
def process_parcel_segments(parcel_seg):
    merged_segments = []  # List to store the reordered segments

    # Group the parcel segments by parcel_id and process each group
    for object_id, group in parcel_seg.groupby('parcel_id'):
        segments = group['geometry'].tolist()  # Get the list of line segments for the current group
        original_indices = group.index.tolist()  # Preserve the original indices
        turning_points = []

        # Loop through all adjacent segments to calculate angle differences
        for i in range(1, len(segments)):
            if are_segments_connected(segments[i-1], segments[i]):
                angle_diff = calculate_angle_difference(segments[i-1], segments[i])
                if angle_diff > 15:  # If angle difference is greater than 15 degrees, mark it as a turning point
                    turning_points.append(i)

        # If there are turning points, reorder the segments starting from the first turning point
        if turning_points:
            turning_point_index = turning_points[0]
            reordered_segments = reorder_segments_by_turning_point(segments, turning_point_index)
            reordered_original_indices = reorder_segments_by_turning_point(original_indices, turning_point_index)
        else:
            # If no turning points, retain the original order
            reordered_segments = segments
            reordered_original_indices = original_indices

        # Store the reordered segments and their attributes
        for j, (line, original_index) in enumerate(zip(reordered_segments, reordered_original_indices)):
            row = group.iloc[0].copy()  # Copy the first row's attributes
            row['geometry'] = line
            row['original_index'] = original_index  # Preserve the original index
            row['new_index'] = j  # Assign the new index based on the reordered list
            merged_segments.append(row)

    # Create a new GeoDataFrame for the reordered segments
    updated_gdf = gpd.GeoDataFrame(merged_segments, columns=parcel_seg.columns.tolist() + ['original_index', 'new_index'])
    updated_gdf = updated_gdf.reset_index(drop=True)

    return updated_gdf

# Run the main function and get the new GeoDataFrame
updated_parcel_seg = process_parcel_segments(parcel_seg)
parcel_seg = updated_parcel_seg


# Group parcel_seg by parcel_id and process each group
merged_segments = []

for object_id, group in parcel_seg.groupby('parcel_id'):
    # Get the list of geometries in the current group
    segments = group.geometry.tolist()
    # Start with the first segment
    merged_lines = [segments[0]]  # Start with the first segment
    
    for i in range(1, len(segments)):
        connected = False
        
        # Always compare the current segment with the previous one
        if are_segments_connected(segments[i-1], segments[i]):
            # Calculate the angle difference between the current segment and the previous one
            angle_diff = calculate_angle_difference(segments[i-1], segments[i])
            
            # If the angle difference is less than 15 degrees, merge the adjacent line segments
            if angle_diff < 15:
                # Merge the current and previous segments
                merged_result = linemerge([merged_lines[-1], segments[i]])
                
                # Check if the result is a MultiLineString, if so, skip the merge
                if isinstance(merged_result, LineString):
                    merged_lines[-1] = merged_result
                    connected = True
                else:
                    # Skip the merge if it's a MultiLineString
                    continue
        
        # If no connected segment is found or the angle difference is too large, add the current segment as a new one
        if not connected:
            merged_lines.append(segments[i])
    
    # Keep the merged results and add other attributes
    for line in merged_lines:
        row = group.iloc[0].copy()  # Copy the first attribute row from the group
        row['geometry'] = line
        merged_segments.append(row)

# Create a new GeoDataFrame from the merged line segments
parcel_seg = gpd.GeoDataFrame(merged_segments, columns=parcel_seg.columns)

# Check for MultiLineString geometries and explode them into LineString
exploded_segments = []

for index, row in parcel_seg.iterrows():
    geom = row['geometry']
    
    if isinstance(geom, MultiLineString):
        # Explode the MultiLineString into individual LineStrings
        for line in geom:
            new_row = row.copy()
            new_row['geometry'] = line
            exploded_segments.append(new_row)
    else:
        # Keep the original LineString geometries
        exploded_segments.append(row)

# Create a new GeoDataFrame from the exploded segments
parcel_seg = gpd.GeoDataFrame(exploded_segments, columns=parcel_seg.columns)

# extract useful columns
parcel_seg.drop(columns=['original_index', 'new_index'], inplace=True)
# Reset the index of the final GeoDataFrame
parcel_seg = parcel_seg.reset_index(drop=True)



edge_counts = parcel_seg.groupby('parcel_id').size()
parcel_seg['edge_num'] = parcel_seg['parcel_id'].map(edge_counts)

# Function to create tangent lines at both ends of a line segment
def create_tangents(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Create tangents at the start and end of the line segment
    start_tangent = LineString([coords[0], coords[1]])
    end_tangent = LineString([coords[-2], coords[-1]])
    
    return start_tangent, end_tangent

# Function to filter curve segments based on angle difference of tangents > 30 degrees
def filter_curve_segments(parcel_seg, angle_threshold=30):
    filtered_segments = []
    non_filtered_segments = []
    
    for idx, row in parcel_seg.iterrows():
        line = row['geometry']
        start_tangent, end_tangent = create_tangents(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            row_dict = row.to_dict()  # Convert the entire row to a dictionary
            row_dict['index'] = idx  # Preserve the original index
            
            if angle_diff > angle_threshold:
                # Add the entire row to the filtered list
                filtered_segments.append(row_dict)
            else:
                # Add the entire row to the non-filtered list
                non_filtered_segments.append(row_dict)
    
    # Create DataFrames with the filtered and non-filtered results if data exists
    if filtered_segments:
        filtered_df = pd.DataFrame(filtered_segments).set_index('index')
        filtered_gdf = gpd.GeoDataFrame(filtered_df, crs=parcel_seg.crs, geometry=filtered_df['geometry'])
    else:
        # Initialize an empty GeoDataFrame with the same structure if no data
        filtered_gdf = gpd.GeoDataFrame(columns=parcel_seg.columns, crs=parcel_seg.crs)
    
    if non_filtered_segments:
        non_filtered_df = pd.DataFrame(non_filtered_segments).set_index('index')
        non_filtered_gdf = gpd.GeoDataFrame(non_filtered_df, crs=parcel_seg.crs, geometry=non_filtered_df['geometry'])
    else:
        # Initialize an empty GeoDataFrame with the same structure if no data
        non_filtered_gdf = gpd.GeoDataFrame(columns=parcel_seg.columns, crs=parcel_seg.crs)
    
    return filtered_gdf, non_filtered_gdf

# Call the function to filter curve segments and create two GeoDataFrames
filtered_parcel_seg, non_filtered_parcel_seg = filter_curve_segments(parcel_seg[parcel_seg['edge_num'] == 3])

# Function to create tangent lines and reverse the line if necessary
def create_tangents_with_reversal(line):
    coords = list(line.coords)
    if len(coords) < 2:
        return None, None  # Skip invalid geometries
    
    # Find the points with the smallest and largest y-coordinate (latitude)
    if coords[0][1] < coords[-1][1]:  # If the first point's y is smaller, it's the start point
        start_point = coords[0]
        end_point = coords[-1]
    else:  # Otherwise, the last point is the start point
        start_point = coords[-1]
        end_point = coords[0]

    # Reverse the line if start_point is not the same as coords[0]
    if start_point != coords[0]:
        coords.reverse()  # Reverse the order of coordinates
    
    # Now create tangents based on the (possibly reversed) coordinates
    start_tangent = LineString([coords[0], coords[1]])  # Tangent from the first to the second point
    end_tangent = LineString([coords[-2], coords[-1]])  # Tangent from the second last to the last point

    return start_tangent, end_tangent, LineString(coords)  # Return the tangents and the (possibly reversed) line

# Function to calculate the split point based on the 4/5 rule
def calculate_split_point(line, start_tangent, end_tangent, angle_diff, angle_fraction=0.5):
    coords = list(line.coords)

    # Iterate through the line and find the point where the angle difference is approximately 4/5
    for i in range(1, len(coords) - 1):
        intermediate_tangent = LineString([coords[i - 1], coords[i]])
        current_angle_diff = calculate_angle_difference(start_tangent, intermediate_tangent)
        
        if current_angle_diff >= angle_diff * angle_fraction:
            return coords[i]  # Return the split point

    return coords[-1]  # If no point found, return the endpoint

# Function to process each segment in filtered_parcel_seg
def process_filtered_parcel_seg(filtered_parcel_seg, angle_threshold=30, angle_fraction=0.5):
    new_data = []
    
    for idx, row in filtered_parcel_seg.iterrows():
        line = row['geometry']
        
        # Apply the tangent and reversal function
        start_tangent, end_tangent, adjusted_line = create_tangents_with_reversal(line)
        
        if start_tangent and end_tangent:
            angle_diff = calculate_angle_difference(start_tangent, end_tangent)
            
            if angle_diff > angle_threshold:
                # Calculate the split point based on the angle difference and fraction
                split_point = calculate_split_point(adjusted_line, start_tangent, end_tangent, angle_diff, angle_fraction)
                
                # Add split point to row's data
                row_dict = row.to_dict()
                row_dict['split_point'] = Point(split_point)  # Store the split point as geometry
                row_dict['index'] = idx  # Store the original index
                
                new_data.append(row_dict)
            else:
                # If no split needed, just keep the original row
                row_dict = row.to_dict()
                row_dict['split_point'] = None  # No split point, store None
                row_dict['index'] = idx  # Store the original index
                
                new_data.append(row_dict)

    # Convert the processed data back into a GeoDataFrame
    new_df = pd.DataFrame(new_data).set_index('index')  # Use original index
    new_gdf = gpd.GeoDataFrame(new_df, crs=parcel_seg.crs, geometry='split_point')
    
    return new_gdf

# Check if filtered_parcel_seg is non-empty before processing
if not filtered_parcel_seg.empty:
    # Call the function to process the filtered_parcel_seg
    processed_parcel_seg = process_filtered_parcel_seg(filtered_parcel_seg)
else:
    # Handle the case where filtered_parcel_seg is empty
    processed_parcel_seg = gpd.GeoDataFrame(columns=filtered_parcel_seg.columns, crs=parcel_seg.crs)

# Function to split filtered_parcel_seg using points from processed_parcel_seg
def split_lines_with_points(filtered_parcel_seg, processed_parcel_seg):
    split_segments = []

    for idx, row in filtered_parcel_seg.iterrows():
        line = row['geometry']
        split_point_geom = processed_parcel_seg.loc[idx, 'split_point']  # Get the corresponding point geometry from split_point column
        
        if isinstance(split_point_geom, Point):
            # Check if the split point is on the line
            if line.contains(split_point_geom):
                # If the point is on the line, use it directly for splitting
                split_lines = split(line, split_point_geom)
            else:
                # If the point is not on the line, find the closest point on the line
                projected_distance = line.project(split_point_geom)
                nearest_point = line.interpolate(projected_distance)
                split_lines = split(line, nearest_point)
            
            # Handle GeometryCollection by extracting valid LineString geometries
            if isinstance(split_lines, GeometryCollection):
                split_segments.extend([{
                    **row.to_dict(), 'geometry': geom
                } for geom in split_lines.geoms if isinstance(geom, LineString)])
                continue  # Skip to the next iteration

        # If no valid split point or GeometryCollection, add the original row
        split_segments.append(row.to_dict())
    
    # Convert split_segments to a GeoDataFrame and return
    split_gdf = gpd.GeoDataFrame(split_segments, crs=parcel_seg.crs, geometry='geometry')
    return split_gdf

# Check if both filtered_parcel_seg and processed_parcel_seg are non-empty before processing
if not filtered_parcel_seg.empty and not processed_parcel_seg.empty:
    # Call the function to split lines based on points
    split_parcel_seg = split_lines_with_points(filtered_parcel_seg, processed_parcel_seg)
else:
    # Handle the case where one or both GeoDataFrames are empty
    split_parcel_seg = gpd.GeoDataFrame(columns=filtered_parcel_seg.columns, crs=parcel_seg.crs)

# Function to combine split_parcel_seg and non_filtered_parcel_seg, ensuring parcel_id proximity
def combine_parcel_segs(split_parcel_seg, non_filtered_parcel_seg):
    # Ensure both datasets contain the 'parcel_id' column
    if 'parcel_id' not in split_parcel_seg.columns or 'parcel_id' not in non_filtered_parcel_seg.columns:
        raise ValueError("Both datasets must contain the 'parcel_id' column.")
    
    # Convert parcel_id to string to avoid type errors during sorting
    split_parcel_seg['parcel_id'] = split_parcel_seg['parcel_id'].astype(str)
    non_filtered_parcel_seg['parcel_id'] = non_filtered_parcel_seg['parcel_id'].astype(str)
    
    # Concatenate the two GeoDataFrames and ensure 'crs' and 'geometry' are set
    combined_parcel_seg = gpd.GeoDataFrame(
        pd.concat([split_parcel_seg, non_filtered_parcel_seg], ignore_index=True),
        crs=parcel_seg.crs,  # Use the crs from one of the input GeoDataFrames
        geometry='geometry'  # Ensure the geometry column is correctly set
    )
    
    # Sort by 'parcel_id' to ensure similar parcel_id are together
    combined_parcel_seg_sorted = combined_parcel_seg.sort_values(by='parcel_id')
    
    return combined_parcel_seg_sorted

# Check if both split_parcel_seg and non_filtered_parcel_seg are non-empty before processing
if not split_parcel_seg.empty and not non_filtered_parcel_seg.empty:
    # Call the function to combine the datasets
    reconstr_seg = combine_parcel_segs(split_parcel_seg, non_filtered_parcel_seg)
else:
    # Handle the case where one or both GeoDataFrames are empty
    reconstr_seg = gpd.GeoDataFrame(columns=split_parcel_seg.columns, crs=parcel_seg.crs)


# Check if reconstr_seg is non-empty before concatenating
if not reconstr_seg.empty:
    parcel_seg = pd.concat([parcel_seg[parcel_seg['edge_num'] != 3], reconstr_seg], ignore_index=True).reset_index(drop=True)

parcel_seg = parcel_seg.drop(columns=['edge_num'])
parcel_seg = parcel_seg.set_crs(parcel.crs, allow_override=True)

# %%
def normalize_linestring(line):
    # Ensure the coordinates are in a consistent direction (smallest point first)
    if isinstance(line, LineString):
        coords = list(line.coords)
        if coords[0] > coords[-1]:
            coords.reverse()  # Reverse the order of coordinates to normalize the direction
        return LineString(coords)
    else:
        return line  # If it's not a LineString, keep it as is
    
def check_shared_sides_normalized(parcel_seg, threshold=0.1, distance_threshold=100):
    """
    Check for shared sides in parcel_seg using cKDTree for faster neighbor searches.
    
    Parameters:
    - parcel_seg: GeoDataFrame containing parcel segments.
    - threshold: float, minimum proportion of line length overlap to consider as a shared side.
    - distance_threshold: float, maximum distance between line segment midpoints to be considered for comparison.
    
    Returns:
    - parcel_seg: GeoDataFrame with 'shared_side' column indicating whether a side is shared.
    """
    
    # Normalize all the geometry objects
    parcel_seg['normalized_geom'] = parcel_seg['geometry'].apply(normalize_linestring)
    # Extract the midpoints of each line segment to build the KDTree
    midpoints = np.array([line.interpolate(0.5, normalized=True).coords[0] for line in parcel_seg['normalized_geom']])
    # Build cKDTree with midpoints
    kdtree = cKDTree(midpoints)
    # Initialize the 'shared_side' column as False
    parcel_seg['shared_side'] = False
    
    # Loop over each line and find nearby lines using KDTree
    for i, line1 in parcel_seg.iterrows():
        # Query the KDTree for neighbors within the distance_threshold
        indices = kdtree.query_ball_point(midpoints[i], r=distance_threshold)
        
        for j in indices:
            if i != j:  # Avoid comparing the line with itself
                line2 = parcel_seg.iloc[j]
                intersection = line1['normalized_geom'].intersection(line2['normalized_geom'])
                if not intersection.is_empty:
                    # Calculate the proportion of overlap relative to the length of line1
                    overlap_ratio = intersection.length / line1['normalized_geom'].length
                    if overlap_ratio > threshold:
                        # If the overlap is greater than the threshold, mark as shared side
                        parcel_seg.at[i, 'shared_side'] = True
                        parcel_seg.at[j, 'shared_side'] = True

    # Remove the temporarily generated 'normalized_geom' column
    parcel_seg = parcel_seg.drop(columns=['normalized_geom'])
    return parcel_seg

parcel_seg = check_shared_sides_normalized(parcel_seg)

### Classify regular inside parcels
# calculate the angle between the each parcel seg and nearest road seg
regular_insid_parcel = parcel_seg[parcel_seg['parcel_label'] == 'regular inside parcel']
regular_insid_parcel['parcel_bearing'] = regular_insid_parcel['geometry'].apply(fun_bearing_ra)
regular_insid_parcel['road_bearing'] = regular_insid_parcel['road_geometry'].apply(fun_bearing_ra)
regular_insid_parcel['angle'] = regular_insid_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)
# calculate the distance between the each parcel seg and nearest road seg
regular_insid_parcel['distance_to_road'] = regular_insid_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# Group by 'parcel_id' and perform the operations within each group
def classify_sides(group):
    # Create a new column 'side'
    group['side'] = None 
    # Step 1: Find the two rows with the smallest 'angle' values
    smallest_two_angles = group.nsmallest(2, 'angle')
    if not smallest_two_angles.empty:
        # Compare 'distance_to_road' between the two rows
        idx_min_distance = smallest_two_angles['distance_to_road'].idxmin()
        idx_max_distance = smallest_two_angles['distance_to_road'].idxmax()
        group.loc[idx_min_distance, 'side'] = 'front'
        group.loc[idx_max_distance, 'side'] = 'rear'
    # Step 2: For remaining rows, find shared_side=True and mark as 'Interior side'
    shared_side_true = group[(group['side'].isnull()) & (group['shared_side'] == True)]
    group.loc[shared_side_true.index, 'side'] = 'Interior side'
    # Step 3: Label the remaining rows as 'Exterior side'
    group.loc[group['side'].isnull(), 'side'] = 'Exterior side'
    return group

# Apply the function to each group
regular_insid_parcel = regular_insid_parcel.groupby('parcel_id').apply(classify_sides)
regular_insid_parcel = regular_insid_parcel.reset_index(level=0, drop=True)

### Classify Regular corner parcels

# calculate the angle between the each parcel seg and nearest road seg
regular_corner_parcel = parcel_seg[parcel_seg['parcel_label'] == 'regular corner parcel']
regular_corner_parcel['parcel_bearing'] = regular_corner_parcel['geometry'].apply(fun_bearing_ra)
regular_corner_parcel['road_bearing'] = regular_corner_parcel['road_geometry'].apply(fun_bearing_ra)
regular_corner_parcel['angle'] = regular_corner_parcel.apply(
    lambda row: calculate_angle_difference(row['geometry'], row['road_geometry']), axis=1
)
# calculate the distance between the each parcel seg and nearest road seg
regular_corner_parcel['distance_to_road'] = regular_corner_parcel.apply(lambda row: row['geometry'].centroid.distance(row['road_geometry']), axis=1)

# Apply the function to each group
regular_corner_parcel = regular_corner_parcel.groupby('parcel_id').apply(classify_sides)
regular_corner_parcel = regular_corner_parcel.reset_index(level=0, drop=True)
