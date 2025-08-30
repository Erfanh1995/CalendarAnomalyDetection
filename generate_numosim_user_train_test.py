import pandas as pd
import numpy as np
from datetime import timedelta
import os

# Activity type to venue type mapping based on user's rules
ACTIVITY_TYPE_TO_VENUE = {
    1: 1,   # "Apartment" -> Home (1)
    2: 2,   # "Workplace" -> Work (2)
    3: 2,   # "School" -> Work (2)
    4: 2,   # "ChildCare" -> Work (2)
    5: 3,   # "BuyGoods" -> Restaurant (3)
    6: 3,   # "Services" -> Restaurant (3)
    7: 3,   # "EatOut" -> Restaurant (3)
    8: 3,   # "Errands" -> Restaurant (3)
    9: 4,   # "Recreation" -> Pub (4)
    10: 4,  # "Exercise" -> Pub (4)
    12: 4,  # "HealthCare" -> Pub (4)
    13: 4,  # "Religious" -> Pub (4)
    14: 4,  # "SomethingElse" -> Pub (4)
    # Ignored: 0: Transportation, 11: Visit, 15: DropOff
}

def get_venue_type_from_activities(act_types):
    """Convert activity types to venue type using the highest ID rule"""
    if not act_types or act_types == [0] or act_types == [11] or act_types == [15]:
        return 0  # Ignored types
    
    # Filter out ignored activity types
    valid_activities = [act for act in act_types if act in ACTIVITY_TYPE_TO_VENUE]
    
    if not valid_activities:
        return 0  # No valid activities
    
    # Get venue types for valid activities
    venue_types = [ACTIVITY_TYPE_TO_VENUE[act] for act in valid_activities]
    
    # Return the highest venue type ID (highest priority)
    return max(venue_types)

# Read the parquet files
print("Loading data...")
train_df = pd.read_parquet('data/numosim/stay_points_train.parquet')
test_df = pd.read_parquet('data/numosim/stay_points_test_anomalous.parquet')
poi_df = pd.read_parquet('data/numosim/poi.parquet')

# Parse datetime columns
train_df['start_datetime'] = pd.to_datetime(train_df['start_datetime'])
train_df['end_datetime'] = pd.to_datetime(train_df['end_datetime'])
test_df['start_datetime'] = pd.to_datetime(test_df['start_datetime'])
test_df['end_datetime'] = pd.to_datetime(test_df['end_datetime'])

# Create POI to venue type mapping
print("Creating POI to venue type mapping...")
poi_venue_mapping = {}
for _, row in poi_df.iterrows():
    venue_type = get_venue_type_from_activities(row['act_types'])
    poi_venue_mapping[row['poi_id']] = venue_type

# Add venue type to dataframes
train_df['venue_type'] = train_df['poi_id'].map(poi_venue_mapping)
test_df['venue_type'] = test_df['poi_id'].map(poi_venue_mapping)

# Calculate duration for each stay point
train_df['duration'] = (train_df['end_datetime'] - train_df['start_datetime']).dt.total_seconds()
test_df['duration'] = (test_df['end_datetime'] - test_df['start_datetime']).dt.total_seconds()

# Ensure output directory
os.makedirs('numosim_user_train_test', exist_ok=True)

# Process each agent
print("Processing agents...")
for agent_id, agent_train_df in train_df.groupby('agent_id'):
    print(f"Processing agent {agent_id}...")
    
    # Get test data for this agent
    agent_test_df = test_df[test_df['agent_id'] == agent_id]
    
    # Sort by start time
    agent_train_df = agent_train_df.sort_values('start_datetime').reset_index(drop=True)
    agent_test_df = agent_test_df.sort_values('start_datetime').reset_index(drop=True)
    
    # Calculate center from train data (weighted by duration)
    if agent_train_df.empty:
        continue
    
    # For center calculation, we need coordinates - let's use POI coordinates
    # First, get POI coordinates for this agent's POIs
    agent_pois = set(agent_train_df['poi_id'].unique())
    agent_poi_coords = poi_df[poi_df['poi_id'].isin(agent_pois)][['poi_id', 'latitude', 'longitude']]
    
    # Create POI to coordinates mapping
    poi_coords = {}
    for _, row in agent_poi_coords.iterrows():
        poi_coords[row['poi_id']] = (row['latitude'], row['longitude'])
    
    # Add coordinates to train data
    agent_train_df['latitude'] = agent_train_df['poi_id'].map(lambda x: poi_coords.get(x, (0, 0))[0])
    agent_train_df['longitude'] = agent_train_df['poi_id'].map(lambda x: poi_coords.get(x, (0, 0))[1])
    
    # Calculate weighted center
    total_duration = agent_train_df['duration'].sum()
    if total_duration > 0:
        center_lat = (agent_train_df['latitude'] * agent_train_df['duration']).sum() / total_duration
        center_lon = (agent_train_df['longitude'] * agent_train_df['duration']).sum() / total_duration
    else:
        center_lat, center_lon = 0, 0
    
    # Calculate frequency for each POI
    poi_counts = agent_train_df['poi_id'].value_counts()
    total_visits = poi_counts.sum()
    poi_freq = (poi_counts / total_visits).to_dict()
    
    # Helper to process a dataframe and write to file
    def write_user_file(df_part, fname):
        if df_part.empty:
            return
        
        # Add coordinates to the dataframe
        df_part['latitude'] = df_part['poi_id'].map(lambda x: poi_coords.get(x, (0, 0))[0])
        df_part['longitude'] = df_part['poi_id'].map(lambda x: poi_coords.get(x, (0, 0))[1])
        
        # Group by day
        df_part['day'] = df_part['start_datetime'].dt.date
        df_part['next_day'] = df_part['end_datetime'].dt.date
        
        # Sort by start time to ensure chronological order
        df_part = df_part.sort_values('start_datetime')
        
        # Track overnight stays
        overnight_stays = {}  # day -> tuple from previous day that extends into this day
        
        # First pass: identify overnight stays
        for _, row in df_part.iterrows():
            if row['day'] != row['next_day']:
                # This stay extends to next day(s)
                current_day = row['day']
                end_day = row['next_day']
                
                # Calculate tuple values once
                x = np.sqrt((row['latitude'] - center_lat) ** 2 + (row['longitude'] - center_lon) ** 2)
                y = poi_freq.get(row['poi_id'], 0)
                z = row['venue_type']
                tuple_str = f"({x:.6f},{y:.4f},{z})"
                
                # Add this tuple to all days it spans
                while current_day <= end_day:
                    if current_day != row['day']:  # Don't add to start day (it's already there)
                        overnight_stays[current_day] = tuple_str
                    current_day += timedelta(days=1)
        
        # Second pass: generate daily tuples
        lines = []
        for day, day_df in df_part.groupby('day'):
            tuples = []
            
            # Add overnight stay from previous day if exists
            if day in overnight_stays:
                tuples.append(overnight_stays[day])
            
            # Add all check-ins for this day
            for _, row in day_df.iterrows():
                x = np.sqrt((row['latitude'] - center_lat) ** 2 + (row['longitude'] - center_lon) ** 2)
                y = poi_freq.get(row['poi_id'], 0)
                z = row['venue_type']
                tuples.append(f"({x:.6f},{y:.4f},{z})")
            
            lines.append(' '.join(tuples))
        
        # Write to file
        with open(fname, 'w') as f:
            for line in lines:
                f.write(line + '\n')
    
    # Write train and test files
    write_user_file(agent_train_df, f'numosim_user_train_test/{agent_id}_train.txt')
    write_user_file(agent_test_df, f'numosim_user_train_test/{agent_id}_test.txt')
    
    # Generate average patterns for each day of the week
    def compute_daily_averages(df_part, period_name):
        if df_part.empty:
            return {}
        
        # Create a copy to avoid SettingWithCopyWarning
        df_part_copy = df_part.copy()
        
        # Add coordinates
        df_part_copy['latitude'] = df_part_copy['poi_id'].map(lambda x: poi_coords.get(x, (0, 0))[0])
        df_part_copy['longitude'] = df_part_copy['poi_id'].map(lambda x: poi_coords.get(x, (0, 0))[1])
        
        # Add day of week (0=Monday, 6=Sunday)
        df_part_copy['DayOfWeek'] = df_part_copy['start_datetime'].dt.dayofweek
        
        daily_averages = {}
        for day_of_week in range(7):
            day_df = df_part_copy[df_part_copy['DayOfWeek'] == day_of_week].copy()
            if day_df.empty:
                daily_averages[day_of_week] = []
                continue
            
            # Collect all intervals for this day of week
            intervals = []
            for _, row in day_df.iterrows():
                # Proximity (Euclidean distance)
                x = np.sqrt((row['latitude'] - center_lat) ** 2 + (row['longitude'] - center_lon) ** 2)
                # Frequency (use train freq, fallback to 0)
                y = poi_freq.get(row['poi_id'], 0)
                # VenueType
                z = row['venue_type']
                
                intervals.append({
                    'start': row['start_datetime'],
                    'end': row['end_datetime'],
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
                        prev_tuple[3] != curr_tuple[3]):  # Exact match for venue type
                        filtered_tuples.append(curr_tuple)
                
                averaged_tuples = filtered_tuples
            
            daily_averages[day_of_week] = averaged_tuples
        
        return daily_averages
    
    # Compute averages for train and test periods
    train_averages = compute_daily_averages(agent_train_df, 'train')
    test_averages = compute_daily_averages(agent_test_df, 'test')
    
    # Write averages to file
    with open(f'numosim_user_train_test/{agent_id}_averages.txt', 'w') as f:
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
    
    # Generate hourly averages
    def compute_hourly_averages(df_part, period_name):
        if df_part.empty:
            return {}
        
        # Create a copy to avoid SettingWithCopyWarning
        df_part_copy = df_part.copy()
        
        # Add coordinates
        df_part_copy['latitude'] = df_part_copy['poi_id'].map(lambda x: poi_coords.get(x, (0, 0))[0])
        df_part_copy['longitude'] = df_part_copy['poi_id'].map(lambda x: poi_coords.get(x, (0, 0))[1])
        
        # Add day of week (0=Monday, 6=Sunday)
        df_part_copy['DayOfWeek'] = df_part_copy['start_datetime'].dt.dayofweek
        
        hourly_averages = {}
        for day_of_week in range(7):
            day_df = df_part_copy[df_part_copy['DayOfWeek'] == day_of_week].copy()
            if day_df.empty:
                hourly_averages[day_of_week] = []
                continue
            
            # Collect all intervals for this day of week
            intervals = []
            for _, row in day_df.iterrows():
                # Proximity (Euclidean distance)
                x = np.sqrt((row['latitude'] - center_lat) ** 2 + (row['longitude'] - center_lon) ** 2)
                # Frequency (use train freq, fallback to 0)
                y = poi_freq.get(row['poi_id'], 0)
                # VenueType
                z = row['venue_type']
                
                intervals.append({
                    'start': row['start_datetime'],
                    'end': row['end_datetime'],
                    'tuple': (x, y, z)
                })
            
            # For each hour (0-23), find all intervals that overlap with that hour
            hourly_tuples = []
            for hour in range(24):
                # Create timezone-aware timestamp to match the data
                hour_start = pd.Timestamp('2024-01-01 00:00:00-08:00').replace(hour=hour, minute=0, second=0)
                hour_end = hour_start + pd.Timedelta(hours=1)
                
                # Find all intervals that overlap with this hour
                overlapping_intervals = []
                for interval in intervals:
                    # Check if interval overlaps with the hour
                    if interval['start'] < hour_end and interval['end'] > hour_start:
                        overlapping_intervals.append(interval)
                
                if overlapping_intervals:
                    # Average x and y values
                    avg_x = np.mean([interval['tuple'][0] for interval in overlapping_intervals])
                    avg_y = np.mean([interval['tuple'][1] for interval in overlapping_intervals])
                    
                    # Find the venue type where most time was spent in this hour
                    venue_durations = {}
                    
                    for interval in overlapping_intervals:
                        # Calculate overlap duration with this hour
                        overlap_start = max(interval['start'], hour_start)
                        overlap_end = min(interval['end'], hour_end)
                        overlap_duration = (overlap_end - overlap_start).total_seconds()
                        
                        if overlap_duration > 0:
                            venue_type = interval['tuple'][2]
                            if venue_type not in venue_durations:
                                venue_durations[venue_type] = 0
                            venue_durations[venue_type] += overlap_duration
                    
                    if venue_durations:
                        # Use the venue type where most time was spent
                        primary_venue = max(venue_durations, key=venue_durations.get)
                        avg_z = primary_venue
                    else:
                        avg_z = 0
                    
                    time_str = f"{hour:02d}:00"
                    hourly_tuples.append((time_str, avg_x, avg_y, avg_z))
                else:
                    # No data for this hour, add empty tuple
                    time_str = f"{hour:02d}:00"
                    hourly_tuples.append((time_str, 0.0, 0.0, 0))
            
            hourly_averages[day_of_week] = hourly_tuples
        
        return hourly_averages
    
    # Compute hourly averages for train and test periods
    train_hourly = compute_hourly_averages(agent_train_df, 'train')
    test_hourly = compute_hourly_averages(agent_test_df, 'test')
    
    # Write hourly averages to file
    with open(f'numosim_user_train_test/{agent_id}_hourly_averages.txt', 'w') as f:
        f.write(f"Agent {agent_id} - Hourly Averages\n")
        f.write("=" * 50 + "\n\n")
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        f.write("TRAIN PERIOD HOURLY AVERAGES:\n")
        f.write("-" * 40 + "\n")
        for day_of_week in range(7):
            hourly_tuples = train_hourly.get(day_of_week, [])
            f.write(f"{day_names[day_of_week]}:\n")
            for time, avg_x, avg_y, avg_z in hourly_tuples:
                f.write(f"  {time}: ({avg_x:.6f}, {avg_y:.4f}, {avg_z:.1f})\n")
            f.write("\n")
        
        f.write("TEST PERIOD HOURLY AVERAGES:\n")
        f.write("-" * 40 + "\n")
        for day_of_week in range(7):
            hourly_tuples = test_hourly.get(day_of_week, [])
            f.write(f"{day_names[day_of_week]}:\n")
            for time, avg_x, avg_y, avg_z in hourly_tuples:
                f.write(f"  {time}: ({avg_x:.6f}, {avg_y:.4f}, {avg_z:.1f})\n")
            f.write("\n")

print('Done! Files are in numosim_user_train_test/') 