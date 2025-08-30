import pandas as pd
import numpy as np
from datetime import timedelta
import os
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# LocationType mapping
LOCATION_TYPE_MAP = {
    'Apartment': 1,
    'Workplace': 2,
    'Restaurant': 3,
    'Pub': 4
}

# Read the TSV files - these are already properly split train and test files
train_filename = 'data/geolife/train-20-outliers-69-agents-0.8-normal-portion.tsv'
test_filename = 'data/geolife/test-20-outliers-69-agents-0.8-normal-portion.tsv'

# Read data and split the single column into multiple columns
train_df = pd.read_csv(train_filename, sep='\t', header=None)
test_df = pd.read_csv(test_filename, sep='\t', header=None)

# Split the first row to get column names
header_row = train_df.iloc[0, 0].split()
train_df.columns = [0]  # Temporary column name
test_df.columns = [0]   # Temporary column name

# Split the data into separate columns
train_data = []
for i, row in train_df.iterrows():
    if i == 0:  # Skip header row
        continue
    values = row[0].split()
    if len(values) == 6:  # Ensure we have all 6 columns
        train_data.append(values)

test_data = []
for i, row in test_df.iterrows():
    if i == 0:  # Skip header row
        continue
    values = row[0].split()
    if len(values) == 6:  # Ensure we have all 6 columns
        test_data.append(values)

# Create new dataframes with proper columns
train_df = pd.DataFrame(train_data, columns=header_row)
test_df = pd.DataFrame(test_data, columns=header_row)

# Convert numeric columns to proper types
for df in [train_df, test_df]:
    df['Latitude'] = pd.to_numeric(df['Latitude'], errors='coerce')
    df['Longitude'] = pd.to_numeric(df['Longitude'], errors='coerce')
    df['AgentID'] = pd.to_numeric(df['AgentID'], errors='coerce')

# Parse ArrivingTime and LeavingTime as datetime
for df in [train_df, test_df]:
    if not np.issubdtype(df['ArrivingTime'].dtype, np.datetime64):
        df['ArrivingTime'] = pd.to_datetime(df['ArrivingTime'], format='%Y-%m-%d,%H:%M:%S')
    if not np.issubdtype(df['LeavingTime'].dtype, np.datetime64):
        df['LeavingTime'] = pd.to_datetime(df['LeavingTime'], format='%Y-%m-%d,%H:%M:%S')

# Map LocationType to integer
for df in [train_df, test_df]:
    if df['LocationType'].dtype != int:
        df['LocationType'] = df['LocationType'].map(LOCATION_TYPE_MAP)

# Ensure output directory
os.makedirs('geolife_user_train_test', exist_ok=True)

def find_staypoints(latitudes, longitudes, durations, eps=0.001, min_duration=300):
    """
    Find staypoints using a time-aware clustering approach.
    
    Args:
        latitudes, longitudes: Location coordinates
        durations: Duration of each stay in seconds
        eps: Distance threshold for clustering (in degrees)
        min_duration: Minimum duration for a staypoint (in seconds)
    
    Returns:
        staypoint_centers_lat, staypoint_centers_lon: Centers of staypoints
        labels: Cluster labels for each point
    """
    if len(latitudes) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert to numpy arrays
    latitudes = np.array(latitudes)
    longitudes = np.array(longitudes)
    durations = np.array(durations)
    
    # Calculate pairwise distances
    coords = np.column_stack([latitudes, longitudes])
    n_points = len(coords)
    labels = np.full(n_points, -1, dtype=int)
    cluster_id = 0
    
    # First pass: group by proximity
    for i in range(n_points):
        if labels[i] == -1:  # Unassigned point
            labels[i] = cluster_id
            
            # Find all points within eps distance
            distances = np.sqrt(np.sum((coords - coords[i])**2, axis=1))
            nearby_indices = np.where(distances <= eps)[0]
            
            # Assign all nearby points to the same cluster
            for j in nearby_indices:
                if labels[j] == -1:
                    labels[j] = cluster_id
            
            cluster_id += 1
    
    # Second pass: filter clusters by minimum duration
    unique_labels = np.unique(labels)
    valid_clusters = []
    
    for label in unique_labels:
        mask = labels == label
        cluster_duration = np.sum(durations[mask])
        
        if cluster_duration >= min_duration:
            valid_clusters.append(label)
        else:
            # Mark points in invalid clusters as noise (-1)
            labels[mask] = -1
    
    # Calculate staypoint centers for valid clusters
    staypoint_centers_lat = []
    staypoint_centers_lon = []
    
    for label in valid_clusters:
        mask = labels == label
        # Weighted average by duration
        total_duration = np.sum(durations[mask])
        center_lat = np.sum(latitudes[mask] * durations[mask]) / total_duration
        center_lon = np.sum(longitudes[mask] * durations[mask]) / total_duration
        
        staypoint_centers_lat.append(center_lat)
        staypoint_centers_lon.append(center_lon)
    
    return np.array(staypoint_centers_lat), np.array(staypoint_centers_lon), labels

def find_most_eventful_period(df, agent_id, weeks=4):
    """Find the most eventful 4-week period for a given agent."""
    agent_df = df[df['AgentID'] == agent_id].copy()
    
    if agent_df.empty:
        return None, None
    
    # Sort by arriving time
    agent_df = agent_df.sort_values('ArrivingTime').reset_index(drop=True)
    
    # Calculate duration for each stay
    agent_df['Duration'] = (agent_df['LeavingTime'] - agent_df['ArrivingTime']).dt.total_seconds()
    
    # Find all possible 4-week windows
    start_time = agent_df['ArrivingTime'].min()
    end_time = agent_df['ArrivingTime'].max()
    
    if (end_time - start_time).days < weeks * 7:
        # If data spans less than 4 weeks, use all data
        return agent_df, agent_df
    
    # Try different 4-week windows and find the one with most events
    max_events = 0
    best_window_start = None
    best_window_end = None
    
    # Slide the window by 1 week increments
    for i in range(0, (end_time - start_time).days - weeks * 7 + 1, 7):
        window_start = start_time + timedelta(days=i)
        window_end = window_start + timedelta(weeks=weeks)
        
        # Count events in this window
        window_events = agent_df[(agent_df['ArrivingTime'] >= window_start) & 
                                (agent_df['ArrivingTime'] < window_end)]
        
        total_events = len(window_events)
        
        if total_events > max_events:
            max_events = total_events
            best_window_start = window_start
            best_window_end = window_end
    
    if best_window_start is None:
        return None, None
    
    # Split the best window into train (first 2 weeks) and test (last 2 weeks)
    train_start = best_window_start
    train_end = train_start + timedelta(weeks=2)
    test_start = train_end
    test_end = best_window_end
    
    train_window = agent_df[(agent_df['ArrivingTime'] >= train_start) & 
                           (agent_df['ArrivingTime'] < train_end)]
    test_window = agent_df[(agent_df['ArrivingTime'] >= test_start) & 
                          (agent_df['ArrivingTime'] < test_end)]
    
    return train_window, test_window

def process_agent_data(agent_id, train_df, test_df):
    """Process data for a single agent."""
    # Get agent data from the already-split train and test files
    agent_train_df = train_df[train_df['AgentID'] == agent_id].copy()
    agent_test_df = test_df[test_df['AgentID'] == agent_id].copy()
    
    # Calculate duration for both train and test data
    agent_train_df['Duration'] = (agent_train_df['LeavingTime'] - agent_train_df['ArrivingTime']).dt.total_seconds()
    agent_test_df['Duration'] = (agent_test_df['LeavingTime'] - agent_test_df['ArrivingTime']).dt.total_seconds()
    
    # Combine train and test data for staypoint detection
    combined_df = pd.concat([agent_train_df, agent_test_df], ignore_index=True)
    combined_latitudes = combined_df['Latitude'].values
    combined_longitudes = combined_df['Longitude'].values
    combined_durations = combined_df['Duration'].values
    
    # Find staypoints from COMBINED train and test data
    staypoint_centers_lat, staypoint_centers_lon, combined_labels = find_staypoints(
        combined_latitudes, combined_longitudes, combined_durations, 
        eps=0.005, min_duration=100  # 0.005 degrees ≈ 0.5km, 100 seconds minimum
    )
    
    # Split the combined labels back to train and test
    train_labels = combined_labels[:len(agent_train_df)]
    test_labels = combined_labels[len(agent_train_df):]
    
    # Track which staypoints were visited in train data
    train_visited_staypoints = set()
    for label in train_labels:
        if label >= 0:  # Valid staypoint
            train_visited_staypoints.add(label)
    
    # For test data, allow assignment to any staypoint (including those only visited in test)
    # But frequency will only be non-zero for staypoints visited in train
    # No need to reassign test labels - they can stay as assigned by the clustering
    
    # Calculate center from TRAIN data only (using staypoint centers if available)
    if len(staypoint_centers_lat) > 0:
        # Use staypoint centers for center calculation
        center_lat = np.mean(staypoint_centers_lat)
        center_lon = np.mean(staypoint_centers_lon)
    else:
        # Fallback to original approach
        center_lat = np.average(agent_train_df['Latitude'].values, weights=agent_train_df['Duration'].values)
        center_lon = np.average(agent_train_df['Longitude'].values, weights=agent_train_df['Duration'].values)
    
    # Calculate frequency for each staypoint using TRAIN data only
    # But only for staypoints that were visited in train data
    staypoint_freq = {}
    if len(staypoint_centers_lat) > 0:
        # Count visits to each staypoint from TRAIN data only
        # But only for staypoints that were visited in train data
        for i, label in enumerate(train_labels):
            if label >= 0 and label in train_visited_staypoints:  # Valid staypoint visited in train
                staypoint_id = f"{staypoint_centers_lat[label]:.8f}_{staypoint_centers_lon[label]:.8f}"
                staypoint_freq[staypoint_id] = staypoint_freq.get(staypoint_id, 0) + 1
        
        # Convert to frequencies
        total_staypoint_visits = sum(staypoint_freq.values())
        if total_staypoint_visits > 0:
            staypoint_freq = {k: v / total_staypoint_visits for k, v in staypoint_freq.items()}
            

    else:
        # Fallback to original approach
        agent_train_df['LocationID'] = agent_train_df['Latitude'].round(8).astype(str) + '_' + agent_train_df['Longitude'].round(8).astype(str)
        location_counts = agent_train_df['LocationID'].value_counts()
        total_visits = location_counts.sum()
        staypoint_freq = (location_counts / total_visits).to_dict()
    
    # Use the entire train and test data for the agent
    train_period_df = agent_train_df.copy()
    test_period_df = agent_test_df.copy()
    
    # Calculate duration for the periods
    train_period_df['Duration'] = (train_period_df['LeavingTime'] - train_period_df['ArrivingTime']).dt.total_seconds()
    test_period_df['Duration'] = (test_period_df['LeavingTime'] - test_period_df['ArrivingTime']).dt.total_seconds()
    
    # Create mappings from row names to labels
    train_label_map = dict(zip(agent_train_df.index, train_labels))
    test_label_map = dict(zip(agent_test_df.index, test_labels))
    
    # Make train_visited_staypoints accessible to nested functions
    train_visited_staypoints_set = train_visited_staypoints
    
    def write_agent_file(df_part, fname, is_train=True):
        """Write agent data to file in the required format."""
        if df_part.empty:
            return
        
        # Get the full date range
        start_date = df_part['ArrivingTime'].min().date()
        end_date = df_part['ArrivingTime'].max().date()
        
        # Group by day
        df_part['Day'] = df_part['ArrivingTime'].dt.date
        lines = []
        
        # Generate all days in the period
        current_date = start_date
        while current_date <= end_date:
            day_df = df_part[df_part['Day'] == current_date]
            
            tuples = []
            for _, row in day_df.iterrows():
                # Proximity (Euclidean distance in lat/lon space)
                x = np.sqrt((row['Latitude'] - center_lat) ** 2 + (row['Longitude'] - center_lon) ** 2)
                
                # Frequency: use the staypoint label for this point
                if len(staypoint_centers_lat) > 0:
                    # Get the label using the mapping
                    if is_train:
                        label = train_label_map.get(row.name, -1)
                    else:
                        label = test_label_map.get(row.name, -1)
                    
                    # If this point belongs to a valid staypoint that was visited in train, use its frequency
                    if label >= 0 and label in train_visited_staypoints_set:
                        staypoint_id = f"{staypoint_centers_lat[label]:.8f}_{staypoint_centers_lon[label]:.8f}"
                        y = staypoint_freq.get(staypoint_id, 0)
                    else:
                        y = 0
                else:
                    # Fallback to original approach
                    location_id = f"{row['Latitude']:.8f}_{row['Longitude']:.8f}"
                    y = staypoint_freq.get(location_id, 0)
                
                # LocationType
                z = row['LocationType']
                tuples.append(f"({x:.6f},{y:.4f},{z})")
            
            # Always add a line for each day (empty if no events)
            lines.append(' '.join(tuples))
            current_date += timedelta(days=1)
        
        # Write to file
        with open(fname, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    
    # Write train and test files (using most eventful periods)
    write_agent_file(train_period_df, f'geolife_user_train_test/{agent_id}_train.txt', is_train=True)
    write_agent_file(test_period_df, f'geolife_user_train_test/{agent_id}_test.txt', is_train=False)
    
    # Generate average patterns for each day of the week
    def compute_daily_averages(df_part, period_name):
        if df_part.empty:
            return {}
        
        # Create a copy to avoid SettingWithCopyWarning
        df_part_copy = df_part.copy()
        
        # Add day of week (0=Monday, 6=Sunday)
        df_part_copy['DayOfWeek'] = df_part_copy['ArrivingTime'].dt.dayofweek
        
        daily_averages = {}
        for day_of_week in range(7):
            day_df = df_part_copy[df_part_copy['DayOfWeek'] == day_of_week].copy()
            if day_df.empty:
                daily_averages[day_of_week] = []
                continue
            
            # Create intervals with durations for this day of week
            day_df['StartTime'] = day_df['ArrivingTime']
            day_df['EndTime'] = day_df['LeavingTime']
            
            # Collect all intervals for this day of week
            intervals = []
            for _, row in day_df.iterrows():
                # Proximity (Euclidean distance)
                x = np.sqrt((row['Latitude'] - center_lat) ** 2 + (row['Longitude'] - center_lon) ** 2)
                
                # Frequency: use the staypoint label for this point
                if len(staypoint_centers_lat) > 0:
                    # Get the label using the mapping
                    if period_name == 'train':
                        label = train_label_map.get(row.name, -1)
                    else:
                        label = test_label_map.get(row.name, -1)
                    
                    # If this point belongs to a valid staypoint that was visited in train, use its frequency
                    if label >= 0 and label in train_visited_staypoints_set:
                        staypoint_id = f"{staypoint_centers_lat[label]:.8f}_{staypoint_centers_lon[label]:.8f}"
                        y = staypoint_freq.get(staypoint_id, 0)
                    else:
                        y = 0
                else:
                    # Fallback to original approach
                    location_id = f"{row['Latitude']:.8f}_{row['Longitude']:.8f}"
                    y = staypoint_freq.get(location_id, 0)
                
                # LocationType
                z = row['LocationType']
                
                intervals.append({
                    'start': row['StartTime'],
                    'end': row['EndTime'],
                    'tuple': (x, y, z)
                })
            
            # Merge overlapping intervals
            if not intervals:
                daily_averages[day_of_week] = []
                continue
            
            # Sort intervals by start time
            intervals.sort(key=lambda x: x['start'])
            
            # Split overlapping intervals to create distinct time segments
            # First, collect all start and end times to find all distinct time points
            time_points = set()
            for interval in intervals:
                time_points.add(interval['start'])
                time_points.add(interval['end'])
            
            time_points = sorted(list(time_points))
            
            # For each time segment, find all intervals that overlap with it
            averaged_tuples = []
            for i in range(len(time_points) - 1):
                segment_start = time_points[i]
                segment_end = time_points[i + 1]
                
                # Find all intervals that overlap with this segment
                overlapping_intervals = []
                for interval in intervals:
                    if interval['start'] <= segment_start and interval['end'] >= segment_end:
                        overlapping_intervals.append(interval)
                
                if overlapping_intervals:
                    # Average the tuples for this time segment
                    avg_x = np.mean([interval['tuple'][0] for interval in overlapping_intervals])
                    avg_y = np.mean([interval['tuple'][1] for interval in overlapping_intervals])
                    avg_z = np.mean([interval['tuple'][2] for interval in overlapping_intervals])
                    
                    # Convert segment start to time string
                    time_str = f"{segment_start.hour:02d}:{segment_start.minute:02d}"
                    averaged_tuples.append((time_str, avg_x, avg_y, avg_z))
            
            # Sort by time
            averaged_tuples.sort(key=lambda x: int(x[0].split(':')[0]) * 60 + int(x[0].split(':')[1]))
            
            # Remove consecutive duplicate tuples
            if len(averaged_tuples) > 1:
                filtered_tuples = [averaged_tuples[0]]
                for i in range(1, len(averaged_tuples)):
                    prev_tuple = filtered_tuples[-1]
                    curr_tuple = averaged_tuples[i]
                    
                    # Check if tuples are significantly different (tolerance for floating point)
                    tolerance = 0.01
                    if (abs(prev_tuple[1] - curr_tuple[1]) > tolerance or 
                        abs(prev_tuple[2] - curr_tuple[2]) > tolerance or 
                        prev_tuple[3] != curr_tuple[3]):  # Exact match for location type
                        filtered_tuples.append(curr_tuple)
                
                averaged_tuples = filtered_tuples
            
            daily_averages[day_of_week] = averaged_tuples
        
        return daily_averages
    
    # Compute averages for train and test periods (using most eventful periods)
    train_averages = compute_daily_averages(train_period_df, 'train')
    test_averages = compute_daily_averages(test_period_df, 'test')
    
    # Write averages to file
    with open(f'geolife_user_train_test/{agent_id}_averages.txt', 'w') as f:
        f.write(f"Agent {agent_id} - Daily Averages\n")
        f.write("=" * 50 + "\n\n")
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        f.write("TRAIN PERIOD AVERAGES:\n")
        f.write("-" * 30 + "\n")
        for day_of_week in range(7):
            averaged_tuples = train_averages.get(day_of_week, [])
            f.write(f"{day_names[day_of_week]}: {len(averaged_tuples)} time-averaged tuples\n")
            for time, avg_x, avg_y, avg_z in averaged_tuples:
                f.write(f"  {time}: ({avg_x:.6f}, {avg_y:.4f}, {avg_z:.1f})\n")
            f.write("\n")
        
        f.write("TEST PERIOD AVERAGES:\n")
        f.write("-" * 30 + "\n")
        for day_of_week in range(7):
            averaged_tuples = test_averages.get(day_of_week, [])
            f.write(f"{day_names[day_of_week]}: {len(averaged_tuples)} time-averaged tuples\n")
            for time, avg_x, avg_y, avg_z in averaged_tuples:
                f.write(f"  {time}: ({avg_x:.6f}, {avg_y:.4f}, {avg_z:.1f})\n")
            f.write("\n")

# Get all unique agent IDs from both train and test data
all_agent_ids = set(train_df['AgentID'].unique()) | set(test_df['AgentID'].unique())
print(f"Found {len(all_agent_ids)} unique agents")

# Process each agent
for i, agent_id in enumerate(all_agent_ids):
    print(f"Processing agent {agent_id} ({i+1}/{len(all_agent_ids)})")
    process_agent_data(agent_id, train_df, test_df)

print('Done! Files are in geolife_user_train_test/') 