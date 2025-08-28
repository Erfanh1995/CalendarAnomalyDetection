import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.lines import Line2D
import gc
from datetime import datetime, timedelta
import colorsys

pd.options.mode.copy_on_write = True

# Data paths
TRAIN_FILE = "data/geolife/train-20-outliers-69-agents-0.8-normal-portion.tsv"
TEST_FILE = "data/geolife/test-20-outliers-69-agents-0.8-normal-portion.tsv"

# Base colors for locations (in HSV format for easier darkness adjustment)
BASE_LOCATION_COLORS = {
    "Restaurant": (0.0, 1.0, 1.0),      # Red (HSV)
    "Workplace": (0.6, 1.0, 1.0),       # Blue (HSV)
    "Apartment": (0.12, 1.0, 1.0),      # More yellow than orange (HSV)
    "Pub": (0.4, 1.0, 0.7)             # More distinct green (HSV)
}

def calculate_duration_minutes(start_time, end_time):
    """Calculate duration between two timestamps in minutes."""
    return (end_time - start_time).total_seconds() / 60

def calculate_distance(lon1, lat1, lon2, lat2):
    """Calculate Haversine distance between two points in kilometers."""
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371  # Radius of Earth in kilometers
    
    return c * r

def adjust_color_by_distance(base_hsv, distance, max_distance, agent_id=None):
    """Adjust color darkness based on distance from center.
    Closer locations will be lighter, farther locations will be darker."""
    h, s, v = base_hsv
    
    # Normalize distance to [0,1] range
    normalized_dist = distance / max_distance if max_distance > 0 else 0
    
    # Map normalized distance to brightness range
    if abs(h - 0.12) < 0.01:  # For yellow/apartment
        # Apartments: brightness range [0.3, 1.0]
        brightness = 1.0 - (0.7 * normalized_dist)
    else:
        # Other locations: brightness range [0.2, 1.0]
        brightness = 1.0 - (0.8 * normalized_dist)
    
    # Ensure brightness stays within bounds
    brightness = max(0.2, min(1.0, brightness))
    
    if agent_id == 128:  # Debug output for Agent 128
        print(f"{base_hsv[0]:.2f} color: dist={distance:.2f}km, max={max_distance:.2f}km, norm={normalized_dist:.2f}, brightness={brightness:.2f}")
    
    return colorsys.hsv_to_rgb(h, s, brightness)

def find_best_weeks(df, num_weeks=4):
    """Find the weeks with the most data points."""
    # Add week number to each row
    df['Week'] = df['ArrivingTime'].dt.isocalendar().week
    
    # Count data points per week
    week_counts = df['Week'].value_counts()
    
    # Get the top weeks
    best_weeks = week_counts.nlargest(num_weeks).index.tolist()
    
    return sorted(best_weeks)

def filter_to_best_weeks(df, best_weeks):
    """Filter dataframe to only include data from the specified weeks."""
    df['Week'] = df['ArrivingTime'].dt.isocalendar().week
    return df[df['Week'].isin(best_weeks)].copy()

def calculate_agent_center(df):
    """Calculate the time-weighted center point of all locations an agent has visited."""
    # Group by unique locations (Longitude, Latitude coordinates)
    location_groups = df.groupby(['Longitude', 'Latitude'])
    
    total_duration = 0
    weighted_x = 0
    weighted_y = 0
    
    # Calculate time spent at each unique location
    for (lon, lat), group in location_groups:
        # Sum up all durations at this location
        location_duration = group.apply(
            lambda row: calculate_duration_minutes(row['ArrivingTime'], row['LeavingTime']),
            axis=1
        ).sum()
        
        # Add weighted coordinates
        weighted_x += lon * location_duration
        weighted_y += lat * location_duration
        total_duration += location_duration
    
    # Calculate weighted center
    if total_duration > 0:
        center_x = weighted_x / total_duration
        center_y = weighted_y / total_duration
    else:
        # Fallback to simple mean if no duration data
        center_x = df['Longitude'].mean()
        center_y = df['Latitude'].mean()
    
    return {
        'center_x': center_x,
        'center_y': center_y
    }

def cluster_locations(df, distance_threshold=10.0):
    """Cluster nearby locations together to handle GPS variations."""
    # Initialize clusters with first location
    clusters = []
    processed = set()
    
    # Group by location type first
    location_groups = df.groupby('LocationType')
    
    for location_type, location_df in location_groups:
        # Get unique coordinates for this location type
        coords = location_df[['Longitude', 'Latitude']].drop_duplicates()
        
        for _, row in coords.iterrows():
            x, y = float(row['Longitude']), float(row['Latitude'])
            coord_key = (x, y)
            
            if coord_key in processed:
                continue
                
            # Start a new cluster
            cluster = {
                'location_type': location_type,
                'points': [(x, y)],
                'center_x': x,
                'center_y': y
            }
            
            # Find all points within threshold distance
            for _, other_row in coords.iterrows():
                other_x, other_y = float(other_row['Longitude']), float(other_row['Latitude'])
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

def get_location_color(location_type, x, y, agent_center, max_distance, agent_id=None):
    """Get color for location based on its distance from agent's center."""
    distance = calculate_distance(x, y, 
                                agent_center['center_x'], 
                                agent_center['center_y'])
    base_hsv = BASE_LOCATION_COLORS[location_type]
    rgb_color = adjust_color_by_distance(base_hsv, distance, max_distance, agent_id)
    return rgb_color

def load_and_process_data(file_path):
    """Load and process TSV file."""
    print(f"Loading file: {file_path}")
    
    # Read with space as separator
    df = pd.read_csv(file_path, sep=' ', engine='python')
    print("Available columns:", df.columns.tolist())
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    print("Columns after stripping whitespace:", df.columns.tolist())
    
    # Convert timestamps to datetime, replacing comma with space
    df['ArrivingTime'] = pd.to_datetime(df['ArrivingTime'].str.replace(',', ' '))
    df['LeavingTime'] = pd.to_datetime(df['LeavingTime'].str.replace(',', ' '))
    return df

def prepare_visualization_data(df, agent_center=None, global_max_distance=None):
    """Prepare data for visualization by calculating location stays."""
    stays_by_day = {}
    
    # Calculate agent's center point if not provided
    if agent_center is None:
        agent_center = calculate_agent_center(df)
    
    # Use provided global max distance or calculate it
    if global_max_distance is None:
        max_distance = 0
        for _, row in df.iterrows():
            distance = calculate_distance(row['Longitude'], row['Latitude'], 
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
    
    # Get all stays with location information
    for _, row in df.iterrows():
        if row['LocationType'] in BASE_LOCATION_COLORS:
            # Get original coordinates
            orig_x = float(row['Longitude'])
            orig_y = float(row['Latitude'])
            
            # Use cluster center if available
            if (orig_x, orig_y) in coord_to_cluster:
                cluster = coord_to_cluster[(orig_x, orig_y)]
                venue_x = cluster['center_x']
                venue_y = cluster['center_y']
            else:
                venue_x = orig_x
                venue_y = orig_y
            
            stay = {
                'venue': row['LocationType'],
                'start': row['ArrivingTime'],
                'end': row['LeavingTime'],
                'x': venue_x,
                'y': venue_y
            }
            
            # Get the week number and day for the stay
            week_num = stay['start'].isocalendar().week
            day = stay['start'].strftime('%A')[:2].lower()
            
            day_key = (week_num, day)
            if day_key not in stays_by_day:
                stays_by_day[day_key] = []
                
            stays_by_day[day_key].append(stay)
    
    return stays_by_day, agent_center, max_distance

def visualize_calendar(df, agent_id, mode, agent_center=None, global_max_distance=0):
    """Create calendar visualization for a specific agent."""
    day_order = ['mo', 'tu', 'we', 'th', 'fr', 'sa', 'su']
    
    # Calculate the actual date range for this agent
    start_date = df['ArrivingTime'].min()
    end_date = df['LeavingTime'].max()
    if agent_id == 128:  # Only print for Agent 128
        print(f"\nAgent {agent_id} {mode} visualization period:")
        print(f"From: {start_date}")
        print(f"To: {end_date}")
    
    # Process data into location stays
    stays_by_day, agent_center, max_distance = prepare_visualization_data(df, agent_center)
    
    # Get unique weeks, but now based on actual data dates
    weeks = sorted(set(week for week, _ in stays_by_day.keys()))
    if not weeks:
        if agent_id == 128:  # Only print for Agent 128
            print(f"No data to visualize for Agent {agent_id} in {mode} dataset")
        return
        
    week_mapping = {orig: idx for idx, orig in enumerate(weeks)}
    n_weeks = len(weeks)
    if agent_id == 128:  # Only print for Agent 128
        print(f"Number of weeks: {n_weeks}")
    
    # Create figure with adjusted height based on number of weeks
    fig_height = max(10, n_weeks * 0.5)  # Adjust height based on number of weeks
    fig, ax = plt.subplots(figsize=(14, fig_height))
    
    # Draw location rectangles for each day
    for (week, day), stays in stays_by_day.items():
        if week in week_mapping:
            cell_row = week_mapping[week]
            day_idx = day_order.index(day)
            
            # Sort stays by start time to ensure proper rendering order
            stays.sort(key=lambda x: x['start'])
            
            for stay in stays:
                start_time = stay['start']
                end_time = stay['end']
                
                # Calculate the fractional position within the day (0 to 1)
                start_seconds = start_time.hour * 3600 + start_time.minute * 60 + start_time.second + start_time.microsecond / 1000000
                end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1000000
                
                start_frac = start_seconds / 86400
                end_frac = end_seconds / 86400
                
                # Handle events that cross midnight
                if end_time.date() > start_time.date():
                    # If the event ends on a different day, extend it to the end of the current day
                    end_frac = 1.0
                    
                    # Create additional rectangles for subsequent days if needed
                    current_date = start_time.date() + timedelta(days=1)
                    while current_date <= end_time.date():
                        next_day = current_date.strftime('%A')[:2].lower()
                        next_week = current_date.isocalendar().week
                        if (next_week, next_day) in stays_by_day:
                            next_day_idx = day_order.index(next_day)
                            next_row = week_mapping.get(next_week)
                            if next_row is not None:
                                # Get color based on distance from center
                                color = get_location_color(
                                    stay['venue'],
                                    stay['x'],
                                    stay['y'],
                                    agent_center,
                                    global_max_distance,
                                    agent_id
                                )
                                
                                if current_date < end_time.date():
                                    # Full day
                                    ax.add_patch(plt.Rectangle(
                                        (next_day_idx, next_row),
                                        1.0,
                                        1.0,
                                        color=color,
                                        alpha=1.0
                                    ))
                                else:
                                    # Last day - partial
                                    final_end_seconds = end_time.hour * 3600 + end_time.minute * 60 + end_time.second + end_time.microsecond / 1000000
                                    final_end_frac = final_end_seconds / 86400
                                    ax.add_patch(plt.Rectangle(
                                        (next_day_idx, next_row),
                                        final_end_frac,
                                        1.0,
                                        color=color,
                                        alpha=1.0
                                    ))
                        current_date += timedelta(days=1)
                
                # Draw the rectangle for the current day
                x_pos = day_idx + start_frac
                width = end_frac - start_frac if end_time.date() == start_time.date() else 1.0 - start_frac
                
                if width > 0:  # Only draw if there's a positive duration
                    # Get color based on distance from center
                    color = get_location_color(
                        stay['venue'],
                        stay['x'],
                        stay['y'],
                        agent_center,
                        global_max_distance,
                        agent_id
                    )
                    
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
    
    # Set y-ticks with actual week dates
    ax.set_yticks(np.arange(n_weeks) + 0.5)
    week_labels = [f"Week {weeks[i]}" for i in range(n_weeks)]
    ax.set_yticklabels(week_labels,fontsize=18)
    
    # Add title and legend with date range
    duration_days = (end_date - start_date).total_seconds() / (24 * 60 * 60)
    title = f"Location Calendar for Agent {agent_id} - {mode}\n"
    title += f"Duration: {duration_days:.1f} days ({start_date.date()} to {end_date.date()})"
    ax.set_title(title,fontsize=20)
    
    # Create legend with base colors and explanation
    legend_elements = [
        Line2D([0], [0], marker='s', color='w',
               markerfacecolor=colorsys.hsv_to_rgb(*color),
               markersize=10, label=f"{venue}\n(lighter = closer to center)")
        for venue, color in BASE_LOCATION_COLORS.items()
    ]
    ax.legend(handles=legend_elements, title="Location Types",
             loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2)
    
    # Invert y-axis and save
    ax.invert_yaxis()
    plt.tight_layout()
    
    # Save the plot
    image_dir = f"Visualizations-geolife/69-0.8/Agent_{agent_id}"
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"{agent_id}_{mode}.png")
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

def calculate_duration_stats(df, dataset_name=""):
    """Calculate duration statistics for the dataset."""
    print(f"\n{dataset_name} Duration Statistics:")
    print("-" * 50)
    
    for agent_id in df['AgentID'].unique():
        agent_data = df[df['AgentID'] == agent_id]
        first_arrival = agent_data['ArrivingTime'].min()
        last_leaving = agent_data['LeavingTime'].max()
        duration = last_leaving - first_arrival
        days = duration.total_seconds() / (24 * 60 * 60)  # Convert to days
        
        print(f"Agent {agent_id}:")
        print(f"  First Arrival: {first_arrival}")
        print(f"  Last Leaving: {last_leaving}")
        print(f"  Duration: {days:.2f} days")
        print()

def analyze_data_gaps(df, agent_id):
    """Analyze gaps in the data for a specific agent."""
    agent_data = df[df['AgentID'] == agent_id].copy()
    agent_data = agent_data.sort_values('ArrivingTime')
    
    print(f"\nAnalyzing data for Agent {agent_id}:")
    print(f"Total number of records: {len(agent_data)}")
    
    # Calculate gaps between consecutive records
    gaps = []
    for i in range(len(agent_data) - 1):
        current_leaving = agent_data.iloc[i]['LeavingTime']
        next_arriving = agent_data.iloc[i + 1]['ArrivingTime']
        gap = (next_arriving - current_leaving).total_seconds() / 3600  # Convert to hours
        if gap > 1:  # Only consider gaps longer than 1 hour
            gaps.append({
                'start': current_leaving,
                'end': next_arriving,
                'duration_hours': gap
            })
    
    if gaps:
        print("\nFound gaps in data:")
        for gap in gaps:
            print(f"Gap from {gap['start']} to {gap['end']} ({gap['duration_hours']:.1f} hours)")
    else:
        print("No significant gaps found in the data")
    
    # Print time range statistics
    print("\nTime range statistics:")
    print(f"First record: {agent_data['ArrivingTime'].min()}")
    print(f"Last record: {agent_data['LeavingTime'].max()}")
    
    # Print location type distribution
    print("\nLocation type distribution:")
    print(agent_data['LocationType'].value_counts())

def process_dataset(train_path, test_path):
    """Process train and test datasets."""
    print("Processing training data...")
    train_df = load_and_process_data(train_path)
    
    print("\nProcessing test data...")
    test_df = load_and_process_data(test_path)
    
    # Process each agent in both datasets
    all_agents = set(train_df['AgentID'].unique()) | set(test_df['AgentID'].unique())
    print(f"\nTotal number of agents to process: {len(all_agents)}")
    print(f"Agent IDs: {sorted(all_agents)}")
    
    # Debug print for Agent 128
    if 128 in all_agents:
        print("\nFound Agent 128 in dataset")
        train_128 = train_df[train_df['AgentID'] == 128]
        test_128 = test_df[test_df['AgentID'] == 128]
        print(f"Training records: {len(train_128)}")
        print(f"Test records: {len(test_128)}")
    
    for agent_id in all_agents:
        print(f"\nStarting to process Agent {agent_id}...")
        
        # Get agent's training data
        agent_train_full = train_df[train_df['AgentID'] == agent_id].copy()
        
        if not agent_train_full.empty:
            # Find the 4 weeks with most data in training
            best_train_weeks = find_best_weeks(agent_train_full, num_weeks=4)
            if agent_id == 128:  # Only print for Agent 128
                print(f"Selected training weeks for visualization: {best_train_weeks}")
            
            # Filter training data to best weeks
            agent_train = filter_to_best_weeks(agent_train_full, best_train_weeks)
            
            # Calculate center point using only training data
            train_center = calculate_agent_center(agent_train)
            
            if agent_id == 128:
                print(f"\nAgent 128 Debug Info:")
                print(f"Training center: ({train_center['center_x']:.6f}, {train_center['center_y']:.6f})")
                print(f"Training data points after filtering to best weeks: {len(agent_train)}")
                print(f"Training weeks: {best_train_weeks}")
            
            # Calculate max distance using only training data
            max_distance = 0
            distances = []
            for _, row in agent_train.iterrows():
                distance = calculate_distance(row['Longitude'], row['Latitude'], 
                                           train_center['center_x'], 
                                           train_center['center_y'])
                max_distance = max(max_distance, distance)
                if agent_id == 128:
                    distances.append(distance)
            
            if agent_id == 128:
                print(f"Training distances (km): min={min(distances):.2f}, max={max_distance:.2f}, avg={sum(distances)/len(distances):.2f}")
            
            # Process and visualize train data
            if not agent_train.empty:
                visualize_calendar(agent_train, agent_id, mode="Train", 
                                 agent_center=train_center, global_max_distance=max_distance)
            
            # Process test data using the same center and max_distance
            agent_test_full = test_df[test_df['AgentID'] == agent_id].copy()
            if not agent_test_full.empty:
                # Find the 4 weeks with most data in test
                best_test_weeks = find_best_weeks(agent_test_full, num_weeks=4)
                if agent_id == 128:  # Only print for Agent 128
                    print(f"Selected test weeks for visualization: {best_test_weeks}")
                
                # Filter test data to best weeks
                agent_test = filter_to_best_weeks(agent_test_full, best_test_weeks)
                
                if agent_id == 128:
                    print(f"\nAgent 128 Test Debug Info:")
                    print(f"Test data points after filtering to best weeks: {len(agent_test)}")
                    print(f"Test weeks: {best_test_weeks}")
                    test_distances = []
                    for _, row in agent_test.iterrows():
                        distance = calculate_distance(row['Longitude'], row['Latitude'],
                                                   train_center['center_x'],
                                                   train_center['center_y'])
                        test_distances.append(distance)
                    if test_distances:
                        print(f"Test distances (km): min={min(test_distances):.2f}, max={max(test_distances):.2f}, avg={sum(test_distances)/len(test_distances):.2f}")
                    else:
                        print("No test distances calculated - empty test data")
                
                # Process and visualize test data using the same center and max_distance
                if not agent_test.empty:
                    visualize_calendar(agent_test, agent_id, mode="Test", 
                                     agent_center=train_center, global_max_distance=max_distance)
                
                del agent_test_full, agent_test
            
            del agent_train_full, agent_train
        print(f"Finished processing Agent {agent_id}")
        gc.collect()

def main():
    """Main execution function."""
    if os.path.exists(TRAIN_FILE) and os.path.exists(TEST_FILE):
        process_dataset(TRAIN_FILE, TEST_FILE)
    else:
        print(f"Error: Could not find {TRAIN_FILE} or {TEST_FILE}")

if __name__ == "__main__":
    main()
