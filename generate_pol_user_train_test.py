import pandas as pd
import numpy as np
from datetime import timedelta
import os

# VenueType mapping
VENUE_TYPE_MAP = {
    'Apartment': 1,
    'Workplace': 2,
    'Restaurant': 3,
    'Pub': 4
}

# Read the TSV file
filename = 'data/pol/checkin-atl.tsv'
df = pd.read_csv(filename, sep='\t')

# Parse CheckinTime as datetime
if not np.issubdtype(df['CheckinTime'].dtype, np.datetime64):
    df['CheckinTime'] = pd.to_datetime(df['CheckinTime'])

# Map VenueType to integer
if df['VenueType'].dtype != int:
    df['VenueType'] = df['VenueType'].map(VENUE_TYPE_MAP)

# Ensure output directory
os.makedirs('pol_user_train_test', exist_ok=True)

for user_id, user_df in df.groupby('UserId'):
    user_df = user_df.sort_values('CheckinTime').reset_index(drop=True)
    # Compute duration as time to next check-in
    user_df['Duration'] = (user_df['CheckinTime'].shift(-1) - user_df['CheckinTime']).dt.total_seconds()
    # For the last check-in, set duration to median of durations
    median_duration = user_df['Duration'].median()
    user_df.loc[user_df.index[-1], 'Duration'] = median_duration

    # Split into test (last 2 weeks) and train (4 weeks prior)
    last_time = user_df['CheckinTime'].max()
    test_start = last_time - timedelta(weeks=2)
    train_start = test_start - timedelta(weeks=4)

    test_df = user_df[user_df['CheckinTime'] >= test_start]
    train_df = user_df[(user_df['CheckinTime'] >= train_start) & (user_df['CheckinTime'] < test_start)]

    # For center and frequency, use only train data
    if train_df.empty:
        continue  # skip users with no train data

    # Weighted center (X, Y) using duration as weight
    total_duration = train_df['Duration'].sum()
    center_x = (train_df['X'] * train_df['Duration']).sum() / total_duration
    center_y = (train_df['Y'] * train_df['Duration']).sum() / total_duration

    # Frequency for each location (VenueId)
    venue_counts = train_df['VenueId'].value_counts()
    total_visits = venue_counts.sum()
    venue_freq = (venue_counts / total_visits).to_dict()

    # Helper to process a dataframe and write to file
    def write_user_file(df_part, fname):
        if df_part.empty:
            return
            
        # Add next checkin time for duration calculation
        df_part = df_part.sort_values('CheckinTime')
        
        # Group by day and calculate next day based on duration
        df_part['Day'] = df_part['CheckinTime'].dt.date
        df_part['NextDay'] = (df_part['CheckinTime'] + pd.to_timedelta(df_part['Duration'], unit='s')).dt.date
        
        # Track overnight stays
        overnight_stays = {}  # day -> tuple from previous day that extends into this day
        
        # First pass: identify overnight stays
        for _, row in df_part.iterrows():
            if row['Day'] != row['NextDay']:
                # This stay extends to next day(s)
                current_day = row['Day']
                end_day = row['NextDay']
                
                # Calculate tuple values once
                x = np.sqrt((row['X'] - center_x) ** 2 + (row['Y'] - center_y) ** 2)
                y = venue_freq.get(row['VenueId'], 0)
                z = row['VenueType']
                tuple_str = f"({x:.2f},{y:.4f},{z})"
                
                # Add this tuple to all days it spans
                while current_day <= end_day:
                    if current_day != row['Day']:  # Don't add to start day (it's already there)
                        overnight_stays[current_day] = tuple_str
                    current_day += timedelta(days=1)
        
        # Second pass: generate daily tuples
        lines = []
        for day, day_df in df_part.groupby('Day'):
            tuples = []
            
            # Add overnight stay from previous day if exists
            if day in overnight_stays:
                tuples.append(overnight_stays[day])
            
            # Add all check-ins for this day
            for _, row in day_df.iterrows():
                x = np.sqrt((row['X'] - center_x) ** 2 + (row['Y'] - center_y) ** 2)
                y = venue_freq.get(row['VenueId'], 0)
                z = row['VenueType']
                tuples.append(f"({x:.2f},{y:.4f},{z})")
            
            lines.append(' '.join(tuples))
            
        # Write to file
        with open(fname, 'w') as f:
            for line in lines:
                f.write(line + '\n')

    # Write train and test files
    write_user_file(train_df, f'pol_user_train_test/{user_id}_train.txt')
    write_user_file(test_df, f'pol_user_train_test/{user_id}_test.txt')
    
    # Generate average patterns for each day of the week
    def compute_daily_averages(df_part, period_name):
        if df_part.empty:
            return {}
        
        # Create a copy to avoid SettingWithCopyWarning
        df_part_copy = df_part.copy()
        
        # Add day of week (0=Monday, 6=Sunday)
        df_part_copy['DayOfWeek'] = df_part_copy['CheckinTime'].dt.dayofweek
        
        daily_averages = {}
        for day_of_week in range(7):
            day_df = df_part_copy[df_part_copy['DayOfWeek'] == day_of_week].copy()
            if day_df.empty:
                daily_averages[day_of_week] = []
                continue
            
            # Create intervals with durations for this day of week
            day_df['StartTime'] = day_df['CheckinTime']
            day_df['EndTime'] = day_df['CheckinTime'] + pd.to_timedelta(day_df['Duration'], unit='s')
            
            # Collect all intervals for this day of week
            intervals = []
            for _, row in day_df.iterrows():
                # Proximity (Euclidean distance)
                x = np.sqrt((row['X'] - center_x) ** 2 + (row['Y'] - center_y) ** 2)
                # Frequency (use train freq, fallback to 0)
                y = venue_freq.get(row['VenueId'], 0)
                # VenueType
                z = row['VenueType']
                
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
                        prev_tuple[3] != curr_tuple[3]):  # Exact match for venue type
                        filtered_tuples.append(curr_tuple)
                
                averaged_tuples = filtered_tuples
            
            daily_averages[day_of_week] = averaged_tuples
        
        return daily_averages
    
    # Compute averages for train and test periods
    train_averages = compute_daily_averages(train_df, 'train')
    test_averages = compute_daily_averages(test_df, 'test')
    
    # Write averages to file
    with open(f'pol_user_train_test/{user_id}_averages.txt', 'w') as f:
        f.write(f"User {user_id} - Daily Averages\n")
        f.write("=" * 50 + "\n\n")
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        f.write("TRAIN PERIOD AVERAGES:\n")
        f.write("-" * 30 + "\n")
        for day_of_week in range(7):
            averaged_tuples = train_averages.get(day_of_week, [])
            f.write(f"{day_names[day_of_week]}: {len(averaged_tuples)} time-averaged tuples\n")
            for time, avg_x, avg_y, avg_z in averaged_tuples:
                f.write(f"  {time}: ({avg_x:.2f}, {avg_y:.4f}, {avg_z:.1f})\n")
            f.write("\n")
        
        f.write("TEST PERIOD AVERAGES:\n")
        f.write("-" * 30 + "\n")
        for day_of_week in range(7):
            averaged_tuples = test_averages.get(day_of_week, [])
            f.write(f"{day_names[day_of_week]}: {len(averaged_tuples)} time-averaged tuples\n")
            for time, avg_x, avg_y, avg_z in averaged_tuples:
                f.write(f"  {time}: ({avg_x:.2f}, {avg_y:.4f}, {avg_z:.1f})\n")
            f.write("\n")
    
    # Generate hourly averages
    def compute_hourly_averages(df_part, period_name):
        if df_part.empty:
            return {}
        
        # Create a copy to avoid SettingWithCopyWarning
        df_part_copy = df_part.copy()
        
        # Add day of week (0=Monday, 6=Sunday)
        df_part_copy['DayOfWeek'] = df_part_copy['CheckinTime'].dt.dayofweek
        
        hourly_averages = {}
        for day_of_week in range(7):
            day_df = df_part_copy[df_part_copy['DayOfWeek'] == day_of_week].copy()
            if day_df.empty:
                hourly_averages[day_of_week] = []
                continue
            
            # Create intervals with durations for this day of week
            day_df['StartTime'] = day_df['CheckinTime']
            day_df['EndTime'] = day_df['CheckinTime'] + pd.to_timedelta(day_df['Duration'], unit='s')
            
            # Collect all intervals for this day of week
            intervals = []
            for _, row in day_df.iterrows():
                # Proximity (Euclidean distance)
                x = np.sqrt((row['X'] - center_x) ** 2 + (row['Y'] - center_y) ** 2)
                # Frequency (use train freq, fallback to 0)
                y = venue_freq.get(row['VenueId'], 0)
                # VenueType
                z = row['VenueType']
                
                intervals.append({
                    'start': row['StartTime'],
                    'end': row['EndTime'],
                    'tuple': (x, y, z)
                })
            
            # For each hour (0-23), find all intervals that overlap with that hour
            hourly_tuples = []
            for hour in range(24):
                hour_start = pd.Timestamp('2019-01-01').replace(hour=hour, minute=0, second=0)
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
    train_hourly = compute_hourly_averages(train_df, 'train')
    test_hourly = compute_hourly_averages(test_df, 'test')
    
    # Write hourly averages to file
    with open(f'pol_user_train_test/{user_id}_hourly_averages.txt', 'w') as f:
        f.write(f"User {user_id} - Hourly Averages\n")
        f.write("=" * 50 + "\n\n")
        
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        
        f.write("TRAIN PERIOD HOURLY AVERAGES:\n")
        f.write("-" * 40 + "\n")
        for day_of_week in range(7):
            hourly_tuples = train_hourly.get(day_of_week, [])
            f.write(f"{day_names[day_of_week]}:\n")
            for time, avg_x, avg_y, avg_z in hourly_tuples:
                f.write(f"  {time}: ({avg_x:.2f}, {avg_y:.4f}, {avg_z:.1f})\n")
            f.write("\n")
        
        f.write("TEST PERIOD HOURLY AVERAGES:\n")
        f.write("-" * 40 + "\n")
        for day_of_week in range(7):
            hourly_tuples = test_hourly.get(day_of_week, [])
            f.write(f"{day_names[day_of_week]}:\n")
            for time, avg_x, avg_y, avg_z in hourly_tuples:
                f.write(f"  {time}: ({avg_x:.2f}, {avg_y:.4f}, {avg_z:.1f})\n")
            f.write("\n")

print('Done! Files are in pol_user_train_test/') 