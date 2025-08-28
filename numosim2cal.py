import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import gc
from datetime import datetime, timedelta
import colorsys

pd.options.mode.copy_on_write = True
pd.options.mode.chained_assignment = None

# Data paths
TRAIN_FILE = "data/numosim/stay_points_train.parquet"
TEST_FILE = "data/numosim/stay_points_test_anomalous.parquet"
POI_FILE = "data/numosim/poi.parquet"

# Base colors for venues (in HSV format for easier darkness adjustment)
BASE_VENUE_COLORS = {
    "Restaurant": (0.0, 1.0, 1.0),      # Red (HSV)
    "Workplace": (0.6, 1.0, 1.0),       # Blue (HSV)
    "Apartment": (0.12, 1.0, 1.0),      # More yellow than orange (HSV)
    "Pub": (0.4, 1.0, 0.7)             # More distinct green (HSV)
}

# Venue type mapping based on user requirements
VENUE_TYPE_MAPPING = {
    1: "Apartment",     # Home
    2: "Workplace",     # Work
    3: "Workplace",     # School
    4: "Workplace",     # ChildCare
    5: "Restaurant",    # BuyGoods
    6: "Restaurant",    # Services
    7: "Restaurant",    # EatOut
    8: "Restaurant",    # Errands
    9: "Pub",          # Recreation
    10: "Pub",         # Exercise
    12: "Pub",         # HealthCare
    13: "Pub",         # Religious
    14: "Pub",         # SomethingElse
    # Ignored: 0: Transportation, 11: Visit, 15: DropOff
}

def load_poi_data():
    """Load POI data and create mapping from poi_id to venue type."""
    print("Loading POI data...")
    poi_df = pd.read_parquet(POI_FILE)
    
    # Create mapping from poi_id to venue type
    poi_mapping = {}
    poi_coordinates = {}
    
    for _, row in poi_df.iterrows():
        poi_id = row['poi_id']
        act_types = row['act_types']
        latitude = row['latitude']
        longitude = row['longitude']
        
        # Find the highest numbered venue type that we care about
        mapped_venue = None
        for venue_type in sorted(act_types, reverse=True):
            if venue_type in VENUE_TYPE_MAPPING:
                mapped_venue = VENUE_TYPE_MAPPING[venue_type]
                break
        
        if mapped_venue:
            poi_mapping[poi_id] = mapped_venue
            poi_coordinates[poi_id] = {'X': longitude, 'Y': latitude}
    
    return poi_mapping, poi_coordinates

def load_and_process_data(file_path, poi_mapping, poi_coordinates):
    """Load and process parquet file."""
    print(f"Loading {file_path}...")
    df = pd.read_parquet(file_path)
    
    # Convert datetime columns
    df['start_datetime'] = pd.to_datetime(df['start_datetime'])
    df['end_datetime'] = pd.to_datetime(df['end_datetime'])
    
    # Map POI IDs to venue types and coordinates
    df['VenueType'] = df['poi_id'].map(poi_mapping)
    df['X'] = df['poi_id'].map(lambda x: poi_coordinates.get(x, {}).get('X', np.nan))
    df['Y'] = df['poi_id'].map(lambda x: poi_coordinates.get(x, {}).get('Y', np.nan))
    
    # Filter out rows where we don't have venue type mapping or coordinates
    df = df.dropna(subset=['VenueType', 'X', 'Y'])
    
    # Rename columns to match pol2cal.py structure
    df = df.rename(columns={
        'agent_id': 'UserId',
        'start_datetime': 'CheckinTime'
    })
    
    return df

def calculate_venue_stays(df):
    """Calculate venue stays from the already processed stay points data."""
    # Since numosim already provides stay points with start and end times,
    # we can directly use this information
    stays = []
    
    # Sort by CheckinTime (start time)
    df = df.sort_values('CheckinTime')
    
    for _, row in df.iterrows():
        if row['VenueType'] in BASE_VENUE_COLORS:
            stays.append({
                'venue': row['VenueType'],
                'start': row['CheckinTime'],
                'end': row['end_datetime'],
                'x': row['X'],
                'y': row['Y']
            })
    
    return stays

def cluster_locations(df, distance_threshold=0.001):  # Smaller threshold for lat/lng coordinates
    """Cluster nearby locations together to handle GPS variations."""
    # Initialize clusters with first location
    clusters = []
    processed = set()
    
    # Group by venue type first
    venue_groups = df.groupby('VenueType')
    
    for venue_type, venue_df in venue_groups:
        # Get unique coordinates for this venue type
        coords = venue_df[['X', 'Y']].drop_duplicates()
        
        for _, row in coords.iterrows():
            x, y = float(row['X']), float(row['Y'])
            coord_key = (x, y)
            
            if coord_key in processed:
                continue
                
            # Start a new cluster
            cluster = {
                'venue_type': venue_type,
                'points': [(x, y)],
                'center_x': x,
                'center_y': y
            }
            
            # Find all points within threshold distance
            for _, other_row in coords.iterrows():
                other_x, other_y = float(other_row['X']), float(other_row['Y'])
                other_key = (other_x, other_y)
                
                if other_key != coord_key and other_key not in processed:
                    dist = calculate_distance(x, y, other_x, other_y)
                    if dist <= distance_threshold:
                        cluster['points'].append((other_x, other_y))
                        processed.add(other_key)
                        # Update cluster center
                        cluster['center_x'] = sum(p[0] for p in cluster['points']) / len(cluster['points'])
                        cluster['center_y'] = sum(p[1] for p in cluster['points']) / len(cluster['points'])
            
            clusters.append(cluster)
            processed.add(coord_key)
    
    return clusters

def prepare_visualization_data(df, agent_center=None, global_max_distance=None):
    """Prepare data for visualization by processing venue stays."""
    stays_by_day = {}
    
    # Calculate agent's center point if not provided
    if agent_center is None:
        agent_center = calculate_agent_center(df)
    
    # Use provided global max distance or calculate it
    if global_max_distance is None:
        max_distance = 0
        for _, row in df.iterrows():
            distance = calculate_distance(row['X'], row['Y'], 
                                       agent_center['center_x'], 
                                       agent_center['center_y'])
            max_distance = max(max_distance, distance)
    else:
        max_distance = global_max_distance
    
    # Cluster locations to handle GPS variations
    location_clusters = cluster_locations(df)
    
    # Create mapping from original coordinates to cluster centers
    coord_to_cluster = {}
    for cluster in location_clusters:
        for x, y in cluster['points']:
            coord_to_cluster[(x, y)] = {
                'center_x': cluster['center_x'],
                'center_y': cluster['center_y']
            }
    
    # Process stays directly from the data
    for stay in calculate_venue_stays(df):
        start_time = stay['start']
        end_time = stay['end']
        venue_x = stay['x']
        venue_y = stay['y']
        
        # Use cluster center if available
        if (venue_x, venue_y) in coord_to_cluster:
            cluster = coord_to_cluster[(venue_x, venue_y)]
            venue_x = cluster['center_x']
            venue_y = cluster['center_y']
        
        # Handle stays that cross midnight
        current_time = start_time
        while current_time < end_time:
            day_key = (current_time.isocalendar().week, current_time.strftime('%A')[:2].lower())
            
            if day_key not in stays_by_day:
                stays_by_day[day_key] = []
            
            # Calculate end of current day (23:59:59.999999)
            end_of_day = current_time.replace(hour=23, minute=59, second=59, microsecond=999999)
            next_day_start = current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            
            # For the current day, the stay ends either at the actual end time or end of day
            stay_end = min(end_time, end_of_day)
            
            # Add stay segment for current day with location info
            stays_by_day[day_key].append({
                'venue': stay['venue'],
                'start': current_time,
                'end': stay_end,
                'x': venue_x,
                'y': venue_y
            })
            
            # Move to the start of next day
            current_time = next_day_start
    
    return stays_by_day, agent_center, max_distance

def calculate_duration_minutes(start_time, end_time):
    """Calculate duration between two timestamps in minutes."""
    return (end_time - start_time).total_seconds() / 60

def calculate_agent_center(df):
    """Calculate the time-weighted center point of all locations an agent has visited."""
    total_duration = 0
    weighted_x = 0
    weighted_y = 0
    
    # Calculate time spent at each location
    for _, row in df.iterrows():
        duration = calculate_duration_minutes(row['CheckinTime'], row['end_datetime'])
        
        weighted_x += row['X'] * duration
        weighted_y += row['Y'] * duration
        total_duration += duration
    
    # Calculate weighted center
    if total_duration > 0:
        center_x = weighted_x / total_duration
        center_y = weighted_y / total_duration
    else:
        # Fallback to simple mean if no duration data
        center_x = df['X'].mean()
        center_y = df['Y'].mean()
    
    return {
        'center_x': center_x,
        'center_y': center_y
    }

def calculate_distance(x1, y1, x2, y2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def adjust_color_by_distance(base_hsv, distance, max_distance):
    """Adjust color darkness based on distance from center.
    Closer venues will be lighter."""
    h, s, v = base_hsv
    
    # Normalize distance to [0,1] range and clamp it
    normalized_dist = min(1.0, distance / max_distance if max_distance > 0 else 0)
    
    # Map normalized distance to brightness range [0.4, 1.0]
    # distance = 0 -> brightness = 1.0 (lightest)
    # distance = max_distance -> brightness = 0.4 (darkest)
    if abs(h - 0.12) < 0.01:  # For yellow/apartment
        # Use fixed brightness values:
        # - 1.0 for normalized_dist = 0
        # - 0.7 for normalized_dist = 1
        brightness = 1.0 - (0.3 * normalized_dist)
    else:
        # For other colors use full range:
        # - 1.0 for normalized_dist = 0
        # - 0.4 for normalized_dist = 1
        brightness = 1.0 - (0.6 * normalized_dist)
    
    # Ensure we stay within bounds
    brightness = max(0.4, min(1.0, brightness))
    
    return colorsys.hsv_to_rgb(h, s, brightness)

def get_venue_color(venue_type, x, y, agent_center, max_distance, venue_colors, debug=False):
    """Get color for venue based on its distance from agent's center."""
    distance = calculate_distance(x, y, 
                                agent_center['center_x'], 
                                agent_center['center_y'])
    base_hsv = venue_colors[venue_type]
    rgb_color = adjust_color_by_distance(base_hsv, distance, max_distance)
    return rgb_color

def visualize_calendar(df, user_id, mode="Train", agent_center=None, global_max_distance=None):
    """Create calendar visualization for a specific user."""
    day_order = ['mo', 'tu', 'we', 'th', 'fr', 'sa', 'su']
    
    # Process data into venue stays
    stays_by_day, agent_center, max_distance = prepare_visualization_data(df, agent_center, global_max_distance)
    
    # Get unique weeks
    weeks = sorted(set(week for week, _ in stays_by_day.keys()))
    week_mapping = {orig: idx for idx, orig in enumerate(weeks)}
    n_weeks = len(weeks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Draw venue rectangles for each day
    for (week, day), stays in stays_by_day.items():
        if week in week_mapping:
            cell_row = week_mapping[week]
            day_idx = day_order.index(day)
            
            # Sort stays by start time to ensure proper rendering order
            stays.sort(key=lambda x: x['start'])
            
            for stay in stays:
                # Calculate position and width of rectangle
                start_time = stay['start']
                end_time = stay['end']
                
                # Convert to fractional day position (0 to 1)
                start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1000000
                end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1000000
                
                start_frac = start_seconds / 86400
                end_frac = end_seconds / 86400
                
                # Ensure we don't exceed day boundaries
                if end_frac > 1.0:
                    end_frac = 1.0
                if start_frac < 0.0:
                    start_frac = 0.0
                
                x_pos = day_idx + start_frac
                width = end_frac - start_frac
                
                if width > 0:  # Only draw if there's a positive duration
                    # Get color based on venue distance from center
                    color = get_venue_color(
                        stay['venue'],
                        stay['x'],
                        stay['y'],
                        agent_center,
                        max_distance,
                        BASE_VENUE_COLORS
                    )
                    
                    # Draw rectangle
                    ax.add_patch(plt.Rectangle(
                        (x_pos, cell_row),
                        width,
                        1.0,  # Full height of day
                        color=color,
                        alpha=1.0
                    ))
    
    # Draw grid lines
    for row_i in range(n_weeks + 1):
        ax.axhline(y=row_i, color='black', linewidth=1.5)
    
    for day_index in range(len(day_order) + 1):
        ax.axvline(x=day_index, color='black', linewidth=1.5)
    
    # Set axis properties
    ax.set_xlim(0, len(day_order))
    ax.set_ylim(-0.5, n_weeks + 0.5)
    ax.set_xticks(np.arange(len(day_order)) + 0.5)
    ax.set_xticklabels(day_order,fontsize=18)
    
    # Set y-ticks
    ax.set_yticks(np.arange(n_weeks) + 0.5)
    week_labels = [f"Week {i+1}" for i in range(n_weeks)]
    ax.set_yticklabels(week_labels,fontsize=18)
    
    # Add title and legend
    ax.set_title(f"Venue Calendar for Agent {user_id} - {mode}",fontsize=20)
    
    # Create legend with base colors
    legend_elements = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=colorsys.hsv_to_rgb(*color),
               markersize=10, label=f"{venue}\n(darker = farther from center)")
        for venue, color in BASE_VENUE_COLORS.items()
    ]
    ax.legend(handles=legend_elements, title="Venue Types",
             loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Invert y-axis and save
    ax.invert_yaxis()
    plt.tight_layout()
    
    # Save the plot
    image_dir = f"Visualizations-numosim/Agent_{user_id}"
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"{user_id}_{mode}.png")
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

def process_data():
    """Process the numosim dataset."""
    # Load POI mapping
    poi_mapping, poi_coordinates = load_poi_data()
    print(f"Loaded {len(poi_mapping)} POI mappings")
    
    # Load train data
    train_df = load_and_process_data(TRAIN_FILE, poi_mapping, poi_coordinates)
    print(f"Loaded train data: {len(train_df)} records")
    
    # Load test data
    test_df = load_and_process_data(TEST_FILE, poi_mapping, poi_coordinates)
    print(f"Loaded test data: {len(test_df)} records")
    
    # Process each agent
    train_agents = train_df['UserId'].unique()
    test_agents = test_df['UserId'].unique()
    
    # Find agents that appear in both train and test
    common_agents = set(train_agents) & set(test_agents)
    print(f"Processing {len(common_agents)} agents that appear in both train and test data")
    
    for agent_id in sorted(common_agents):  # Process all agents
        print(f"Processing Agent {agent_id}...")
        
        # Get agent data
        agent_train = train_df[train_df['UserId'] == agent_id].copy()
        agent_test = test_df[test_df['UserId'] == agent_id].copy()
        
        if not agent_train.empty and not agent_test.empty:
            # Calculate center point and max distance using training data
            train_center = calculate_agent_center(agent_train)
            
            max_distance = 0
            for _, row in agent_train.iterrows():
                distance = calculate_distance(row['X'], row['Y'], 
                                           train_center['center_x'], 
                                           train_center['center_y'])
                max_distance = max(max_distance, distance)
            
            # Visualize train data
            visualize_calendar(agent_train, agent_id, mode="Train", 
                             agent_center=train_center, global_max_distance=max_distance)
            
            # Visualize test data using the same center and max_distance
            visualize_calendar(agent_test, agent_id, mode="Test", 
                             agent_center=train_center, global_max_distance=max_distance)
        
        del agent_train, agent_test
        gc.collect()

def main():
    """Main execution function."""
    print("Starting numosim calendar visualization...")
    process_data()
    print("Done!")

if __name__ == "__main__":
    main()