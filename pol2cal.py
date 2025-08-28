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
DATA_FILES = [
    "data/pol/checkin-atl.tsv" #,"checkin-nola.tsv", "checkin-atl.tsv", "checkin-fva.tsv"
]

# Base colors for venues (in HSV format for easier darkness adjustment)
BASE_VENUE_COLORS = {
    "Restaurant": (0.0, 1.0, 1.0),      # Red (HSV)
    "Workplace": (0.6, 1.0, 1.0),       # Blue (HSV)
    "Apartment": (0.12, 1.0, 1.0),      # More yellow than orange (HSV)
    "Pub": (0.4, 1.0, 0.7)             # More distinct green (HSV)
}

def load_and_process_data(file_path):
    """Load and process TSV file."""
    df = pd.read_csv(file_path, sep='\t')
    df['CheckinTime'] = pd.to_datetime(df['CheckinTime'])
    return df

def split_train_test(df, train_weeks=4):
    """Split data into training (last N weeks) and test (last 2 weeks) periods."""
    max_date = df['CheckinTime'].max()
    test_start = max_date - timedelta(days=14)
    train_start = test_start - timedelta(weeks=train_weeks)
    
    train_data = df[(df['CheckinTime'] >= train_start) & (df['CheckinTime'] < test_start)].copy()
    test_data = df[df['CheckinTime'] >= test_start].copy()
    
    return train_data, test_data

def calculate_venue_stays(df):
    """Calculate continuous stays at venues, assuming agent stays at each location until next check-in."""
    # Sort by CheckinTime
    df = df.sort_values('CheckinTime')
    
        # Initialize lists to store venue stays
    stays = []
    current_venue = None
    start_time = None
    current_x = None
    current_y = None
    
    # Process each check-in chronologically
    for i, row in df.iterrows():
        current_time = row['CheckinTime']
        
        if current_venue is None:
            # First venue - start from the actual check-in time
            # We don't assume where they were before the first check-in
            current_venue = row['VenueType']
            current_x = float(row['X'])
            current_y = float(row['Y'])
            start_time = current_time
        elif row['VenueType'] != current_venue:
            # Venue change detected - end previous stay and start new one
            if current_venue in BASE_VENUE_COLORS:
                stays.append({
                    'venue': current_venue,
                    'start': start_time,
                    'end': current_time,
                    'x': current_x,
                    'y': current_y
                })
            
            # Start new stay
            current_venue = row['VenueType']
            current_x = float(row['X'])
            current_y = float(row['Y'])
            start_time = current_time
    
    # Handle the final stay - only extend if it's a full day
    if current_venue is not None and current_venue in BASE_VENUE_COLORS:
        end_time = df['CheckinTime'].max()            
        stays.append({
            'venue': current_venue,
            'start': start_time,
            'end': end_time,
            'x': current_x,
            'y': current_y
        })

    return stays


def prepare_visualization_data(df, agent_center=None, global_max_distance=None):
    """Prepare data for visualization by calculating venue stays."""
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
    
    # Group stays by day
    for stay in calculate_venue_stays(df):
        start_time = stay['start']
        end_time = stay['end']
        venue_x = stay['x']
        venue_y = stay['y']
        
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
    # Group by unique locations (X, Y coordinates)
    location_groups = df.groupby(['X', 'Y'])
    
    total_duration = 0
    weighted_x = 0
    weighted_y = 0
    
    # Calculate time spent at each unique location
    for (x, y), group in location_groups:
        # Sort check-ins by time
        group = group.sort_values('CheckinTime')
        
        # Calculate total duration at this location
        location_duration = 0
        for i in range(len(group) - 1):
            current_checkin = group.iloc[i]
            next_checkin = group.iloc[i + 1]
            
            # Only count consecutive check-ins at the same location
            if (current_checkin['X'] == next_checkin['X'] and 
                current_checkin['Y'] == next_checkin['Y']):
                duration = calculate_duration_minutes(
                    current_checkin['CheckinTime'],
                    next_checkin['CheckinTime']
                )
                location_duration += duration
        
        # Add weighted coordinates
        weighted_x += x * location_duration
        weighted_y += y * location_duration
        total_duration += location_duration
    
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
    ax.set_title(f"Venue Calendar for User {user_id} - {mode}", fontsize=20)
    
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
    image_dir = f"Visualizations-pol/atlanta/Agent_{user_id}"
    os.makedirs(image_dir, exist_ok=True)
    image_path = os.path.join(image_dir, f"{user_id}_{mode}.png")
    plt.savefig(image_path, bbox_inches='tight')
    plt.close()

def process_file(file_path, train_weeks=4):
    """Process a single TSV file."""
    print(f"Processing {file_path}...")
    df = load_and_process_data(file_path)
    
    for user_id in df['UserId'].unique():
        print(f"Processing User {user_id}...")
        user_data = df[df['UserId'] == user_id].copy()
        
        if not user_data.empty:
            # Split into train and test
            train_data, test_data = split_train_test(user_data, train_weeks=train_weeks)
            
            # Calculate center point using only training data
            train_center = calculate_agent_center(train_data)
            
            # Calculate max distance using only training data
            max_distance = 0
            for _, row in train_data.iterrows():
                distance = calculate_distance(row['X'], row['Y'], 
                                           train_center['center_x'], 
                                           train_center['center_y'])
                max_distance = max(max_distance, distance)
            
            # Process and visualize train data
            if not train_data.empty:
                visualize_calendar(train_data, user_id, mode="Train", 
                                 agent_center=train_center, global_max_distance=max_distance)
            
            # Process and visualize test data using the same center and max_distance
            if not test_data.empty:
                visualize_calendar(test_data, user_id, mode="Test", 
                                 agent_center=train_center, global_max_distance=max_distance)
        
        del user_data
        gc.collect()

def main():
    """Main execution function."""
    train_weeks = 4  # Set to show last 4 weeks of training data
    for file_path in DATA_FILES:
        if os.path.exists(file_path):
            process_file(file_path, train_weeks=train_weeks)
        else:
            print(f"Warning: {file_path} not found")

if __name__ == "__main__":
    main() 
