"""
Trip Start Reconstruction Tool

This script detects and reconstructs missed trip beginnings due to sensor wake-up delay.
It focuses particularly on the first trips of the day from home locations.

Usage:
    python detect_missing_segs.py --input /path/to/location_data.csv --output /path/to/output_dir
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from geopy.distance import great_circle
import folium
import os
import argparse
import traceback
from datetime import datetime, timedelta


class TripStartReconstructor:
    """
    Detects and reconstructs missed beginnings of trips from GPS tracking data.
    Addresses cases where location sensors had a delayed wake-up.
    """

    def __init__(self, input_file=None, dataframe=None, output_dir='trip_reconstruction_output'):
        """Initialize with either a file path or an existing DataFrame"""
        self.output_dir = output_dir
        self.df = None

        # Create output directory
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Parameters for detection
        self.min_suspicious_speed = 2.0  # m/s (7.2 km/h) - Suspicious if first speed is already this high
        self.max_reasonable_gap = 300  # seconds (5 minutes) between stay end and trip start
        self.min_gap_distance = 100  # meters - Minimum distance to consider a gap significant
        self.home_location_name = "home"  # Name for home location if known

        # Parameters for stay/trip detection (if needed)
        self.stay_distance_threshold = 100  # meters
        self.stay_time_threshold = 300  # seconds (5 minutes)
        self.min_trip_duration = 60  # seconds (1 minute)
        self.min_trip_distance = 100  # meters

        # Load data if provided
        if input_file:
            self.load_data(input_file)
        elif dataframe is not None:
            self.df = dataframe.copy()

    def load_data(self, file_path):
        """Load data from file"""
        print(f"Loading data from {file_path}...")

        # Determine file type from extension
        if file_path.endswith('.csv'):
            try:
                self.df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading CSV, trying with different settings: {e}")
                try:
                    # Try with automatic delimiter detection
                    self.df = pd.read_csv(file_path, engine='python')
                except:
                    raise Exception(f"Failed to load {file_path} as CSV")

        elif file_path.endswith('.pkl') or file_path.endswith('.pickle'):
            self.df = pd.read_pickle(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

        # Basic required columns for any location data
        essential_columns = ['timestamp', 'latitude', 'longitude']
        missing = [col for col in essential_columns if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing essential columns: {missing}. Cannot proceed without location and time data.")

        # Ensure timestamp is datetime
        if isinstance(self.df['timestamp'].iloc[0], str):
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])

        # Check if we need to preprocess to add trip_id and stay_id
        needs_preprocessing = 'trip_id' not in self.df.columns or 'stay_id' not in self.df.columns

        # Add effective_speed if not present
        if 'effective_speed' not in self.df.columns:
            if 'speed' in self.df.columns:
                self.df['effective_speed'] = self.df['speed']
                # Fix negative speeds
                self.df.loc[self.df['effective_speed'] < 0, 'effective_speed'] = 0
            else:
                # We'll calculate it in preprocess_data
                pass

        if needs_preprocessing:
            print("Input data missing trip/stay information. Preprocessing data...")
            self.preprocess_data()

        print(
            f"Loaded {len(self.df)} data points spanning {self.df['timestamp'].min().date()} to {self.df['timestamp'].max().date()}")

        return self.df

    def preprocess_data(self):
        """Preprocess raw location data to identify stays and trips"""
        print("Preprocessing data to identify stays and trips...")

        # Sort by timestamp
        self.df = self.df.sort_values('timestamp')

        # Calculate time differences
        self.df['time_diff'] = self.df['timestamp'].diff().dt.total_seconds()
        self.df.loc[self.df.index[0], 'time_diff'] = 0  # First point

        # Calculate distances and speeds if needed
        if 'distance' not in self.df.columns or 'effective_speed' not in self.df.columns:
            print("Calculating distances and speeds...")
            self.calculate_distances()

        # Add isMoving column if it doesn't exist
        if 'isMoving' not in self.df.columns:
            # Use effective_speed to determine if moving
            self.df['isMoving'] = self.df['effective_speed'] > 0.5
            print("Created isMoving column based on speed")

        # Detect stationary periods (stays)
        self.detect_stays()

        # Segment trips between stays
        self.segment_trips()

        # Infer travel modes if possible
        if 'activityType' in self.df.columns or 'activity' in self.df.columns:
            self.infer_travel_modes_basic()
        else:
            print("Inferring basic travel modes based on speed...")
            self.infer_travel_modes_basic()

        print("Preprocessing complete.")

    def calculate_distances(self):
        """Calculate distances and speeds between consecutive points"""
        print("Calculating distances and speeds...")

        # Initialize arrays
        distances = [0]  # First point has no previous point
        speeds = [0]

        # Calculate for each consecutive pair of points
        for i in range(1, len(self.df)):
            # Get coordinates
            lat1, lon1 = self.df.iloc[i - 1]['latitude'], self.df.iloc[i - 1]['longitude']
            lat2, lon2 = self.df.iloc[i]['latitude'], self.df.iloc[i]['longitude']

            # Get time difference
            time_diff = self.df['time_diff'].iloc[i]

            # Calculate distance
            try:
                distance = great_circle((lat1, lon1), (lat2, lon2)).meters
            except:
                distance = 0

            distances.append(distance)

            # Calculate speed
            if time_diff > 0:
                speed = distance / time_diff  # m/s
            else:
                speed = 0

            speeds.append(speed)

        # Add to dataframe
        self.df['distance'] = distances
        self.df['effective_speed'] = speeds

        # Print summary
        total_distance = sum(distances) / 1000  # km
        avg_speed = sum(speeds) / len(speeds) if len(speeds) > 0 else 0
        print(f"Total distance: {total_distance:.2f} km, Average speed: {avg_speed * 3.6:.1f} km/h")

    def detect_stays(self):
        """Detect stay points (locations where the user remained stationary)"""
        print("Detecting stationary periods...")

        # Initialize stay detection
        self.df['stay_id'] = -1  # -1 means not part of any stay
        current_stay_id = 0
        current_stay_points = []
        start_time = None

        for i, row in self.df.iterrows():
            is_stationary = (not row['isMoving']) if 'isMoving' in self.df.columns else (row['effective_speed'] < 0.5)

            if is_stationary:
                # Potential stay point
                if not current_stay_points:
                    # Start a new stay
                    current_stay_points = [(row['latitude'], row['longitude'])]
                    start_time = row['timestamp']
                else:
                    # Check if this point is close to the centroid of current stay
                    stay_centroid = np.mean(current_stay_points, axis=0)
                    dist = great_circle(stay_centroid, (row['latitude'], row['longitude'])).meters

                    if dist <= self.stay_distance_threshold:
                        # Add to current stay
                        current_stay_points.append((row['latitude'], row['longitude']))
                    else:
                        # Check if previous stay was long enough
                        duration = (row['timestamp'] - start_time).total_seconds()
                        if duration >= self.stay_time_threshold:
                            # Mark previous points as a stay
                            stay_indices = self.df[(self.df['timestamp'] >= start_time) &
                                                   (self.df['timestamp'] < row['timestamp'])].index
                            self.df.loc[stay_indices, 'stay_id'] = current_stay_id
                            current_stay_id += 1

                        # Start a new potential stay
                        current_stay_points = [(row['latitude'], row['longitude'])]
                        start_time = row['timestamp']
            else:
                # Moving point - check if previous stay was long enough
                if current_stay_points and start_time:
                    duration = (row['timestamp'] - start_time).total_seconds()
                    if duration >= self.stay_time_threshold:
                        # Mark previous points as a stay
                        stay_indices = self.df[(self.df['timestamp'] >= start_time) &
                                               (self.df['timestamp'] < row['timestamp'])].index
                        self.df.loc[stay_indices, 'stay_id'] = current_stay_id
                        current_stay_id += 1

                    # Reset stay tracking
                    current_stay_points = []
                    start_time = None

        # Check final stay if it exists
        if current_stay_points and start_time:
            duration = (self.df.iloc[-1]['timestamp'] - start_time).total_seconds()
            if duration >= self.stay_time_threshold:
                stay_indices = self.df[self.df['timestamp'] >= start_time].index
                self.df.loc[stay_indices, 'stay_id'] = current_stay_id

        # Get stay statistics
        num_stays = len(self.df[self.df['stay_id'] >= 0]['stay_id'].unique())
        stay_points = self.df[self.df['stay_id'] >= 0].shape[0]
        stay_percentage = stay_points / len(self.df) * 100 if len(self.df) > 0 else 0

        print(f"Detected {num_stays} stays containing {stay_points} points ({stay_percentage:.1f}% of data).")

    def segment_trips(self):
        """Segment the data into trips between stays."""
        print("Segmenting trips...")

        # Make sure distance column exists
        if 'distance' not in self.df.columns:
            print("Warning: 'distance' column not found. Calculating distances first.")
            self.calculate_distances()

        # Initialize trip segmentation
        self.df['trip_id'] = -1  # -1 means not part of any trip
        current_trip_id = 0
        trip_in_progress = False
        trip_start_time = None

        # Get ordered list of data points
        for i, row in self.df.iterrows():
            if row['stay_id'] >= 0:
                # End trip if in progress
                if trip_in_progress:
                    # Calculate trip duration and distance
                    trip_end_time = row['timestamp']
                    trip_duration = (trip_end_time - trip_start_time).total_seconds()
                    trip_points = self.df[(self.df['timestamp'] >= trip_start_time) &
                                          (self.df['timestamp'] < trip_end_time) &
                                          (self.df['trip_id'] == current_trip_id)]

                    # Check if trip meets minimum criteria
                    trip_distance = trip_points['distance'].sum() if len(trip_points) > 0 else 0

                    if (trip_duration >= self.min_trip_duration and
                            len(trip_points) > 1 and
                            trip_distance >= self.min_trip_distance):
                        # Valid trip
                        current_trip_id += 1
                    else:
                        # Trip too short or not enough movement, remove trip label
                        self.df.loc[trip_points.index, 'trip_id'] = -1

                    trip_in_progress = False
            else:
                # Potential trip point
                if not trip_in_progress:
                    # Start new trip
                    trip_in_progress = True
                    trip_start_time = row['timestamp']

                # Mark as part of current trip
                self.df.loc[i, 'trip_id'] = current_trip_id

        # Check final trip if still in progress
        if trip_in_progress:
            trip_end_time = self.df.iloc[-1]['timestamp']
            trip_duration = (trip_end_time - trip_start_time).total_seconds()
            trip_points = self.df[(self.df['timestamp'] >= trip_start_time) &
                                  (self.df['trip_id'] == current_trip_id)]

            # Check if final trip meets minimum criteria
            trip_distance = trip_points['distance'].sum() if len(trip_points) > 0 else 0

            if not (trip_duration >= self.min_trip_duration and
                    len(trip_points) > 1 and
                    trip_distance >= self.min_trip_distance):
                # Trip too short or not enough movement, remove trip label
                self.df.loc[trip_points.index, 'trip_id'] = -1

        # Get trip statistics
        num_trips = len(self.df[self.df['trip_id'] >= 0]['trip_id'].unique())
        trip_points = self.df[self.df['trip_id'] >= 0].shape[0]
        trip_percentage = trip_points / len(self.df) * 100 if len(self.df) > 0 else 0

        print(f"Detected {num_trips} trips containing {trip_points} points ({trip_percentage:.1f}% of data).")

    def infer_travel_modes_basic(self):
        """Infer basic travel modes based on speed and activity type if available"""
        print("Inferring basic travel modes...")

        # Add predicted_mode column
        self.df['predicted_mode'] = 'unknown'

        # Set non-trip points to 'stationary'
        self.df.loc[self.df['trip_id'] == -1, 'predicted_mode'] = 'stationary'

        # Process each trip
        for trip_id in self.df['trip_id'].unique():
            if trip_id < 0:
                continue

            # Get trip data
            trip_data = self.df[self.df['trip_id'] == trip_id]

            # Try to determine mode based on activity type
            if 'activityType' in trip_data.columns:
                activity_counts = trip_data['activityType'].value_counts()
                if len(activity_counts) > 0:
                    most_common_activity = activity_counts.index[0]

                    # Map to simplified modes
                    mode_mapping = {
                        'still': 'stationary',
                        'on_foot': 'walking',
                        'walking': 'walking',
                        'running': 'walking',
                        'on_bicycle': 'bicycle',
                        'in_vehicle': 'car'
                    }

                    inferred_mode = mode_mapping.get(most_common_activity, 'unknown')
                    self.df.loc[self.df['trip_id'] == trip_id, 'predicted_mode'] = inferred_mode
                    continue

            # Fallback: Use speed-based detection
            avg_speed = trip_data['effective_speed'].mean()
            max_speed = trip_data['effective_speed'].max()

            # Simple speed-based rules
            if max_speed > 20:  # meters/second (72 km/h)
                mode = 'car'
            elif max_speed > 8:  # meters/second (28.8 km/h)
                mode = 'bicycle'
            elif max_speed > 2:  # meters/second (7.2 km/h)
                mode = 'walking'
            else:
                mode = 'walking'  # Default to walking for slow movement

            # Update predicted mode for this trip
            self.df.loc[self.df['trip_id'] == trip_id, 'predicted_mode'] = mode

    def save_data(self, file_path=None):
        """Save processed data to file"""
        if file_path is None:
            file_path = os.path.join(self.output_dir, 'reconstructed_data.pkl')

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save based on extension
        if file_path.endswith('.csv'):
            self.df.to_csv(file_path, index=False)
        else:
            self.df.to_pickle(file_path)

        print(f"Saved processed data to {file_path}")
        return file_path

    def detect_and_fix_missed_trip_starts(self):
        """
        Detects and reconstructs missed beginnings of trips, particularly when leaving home.
        This addresses the sensor wake-up delay issue that causes the beginning of trips to be missed.
        """
        print("\nAnalyzing trips for potential missed starts...")

        # Storage for detected issues
        detected_issues = []
        reconstructed_trips = 0

        # First, identify home locations (most frequent overnight stay or first location of the day)
        home_locations = self.identify_home_locations()

        # Process each day of data
        for date, day_data in self.df.groupby(self.df['timestamp'].dt.date):
            # Sort by timestamp
            day_data = day_data.sort_values('timestamp')

            # Get ordered sequence of stays and trips for the day
            day_items = []

            # Get all stays
            for stay_id in sorted(day_data['stay_id'].unique()):
                if stay_id < 0:
                    continue

                stay_data = day_data[day_data['stay_id'] == stay_id]
                if len(stay_data) < 2:
                    continue

                stay_info = {
                    'type': 'stay',
                    'id': stay_id,
                    'start_time': stay_data['timestamp'].min(),
                    'end_time': stay_data['timestamp'].max(),
                    'lat': stay_data['latitude'].mean(),
                    'lon': stay_data['longitude'].mean(),
                    'is_home': any(self.is_same_location((stay_data['latitude'].mean(), stay_data['longitude'].mean()),
                                                         home_loc, threshold=100)
                                   for home_loc in home_locations)
                }
                day_items.append(stay_info)

            # Get all trips
            for trip_id in sorted(day_data['trip_id'].unique()):
                if trip_id < 0:
                    continue

                trip_data = day_data[day_data['trip_id'] == trip_id]
                if len(trip_data) < 2:
                    continue

                trip_info = {
                    'type': 'trip',
                    'id': trip_id,
                    'start_time': trip_data['timestamp'].min(),
                    'end_time': trip_data['timestamp'].max(),
                    'start_lat': trip_data.iloc[0]['latitude'],
                    'start_lon': trip_data.iloc[0]['longitude'],
                    'end_lat': trip_data.iloc[-1]['latitude'],
                    'end_lon': trip_data.iloc[-1]['longitude'],
                    'mode': trip_data['predicted_mode'].mode().iloc[0],
                    'start_speed': trip_data.iloc[0]['effective_speed'],
                    'data': trip_data
                }
                day_items.append(trip_info)

            # Sort by start time
            day_items.sort(key=lambda x: x['start_time'])

            # Look for suspicious patterns
            for i in range(1, len(day_items)):
                current_item = day_items[i]
                previous_item = day_items[i - 1]

                # Only interested in stay -> trip transitions
                if previous_item['type'] != 'stay' or current_item['type'] != 'trip':
                    continue

                # Calculate time gap
                time_gap = (current_item['start_time'] - previous_item['end_time']).total_seconds()

                # Calculate distance between stay and trip start
                distance = great_circle((previous_item['lat'], previous_item['lon']),
                                        (current_item['start_lat'], current_item['start_lon'])).meters

                # Get trip data
                trip_data = current_item['data']
                trip_id = current_item['id']

                # Check for suspicious indicators
                is_suspicious = False
                suspicious_reasons = []

                # INDICATOR 1: Trip starts with high speed (not normal acceleration)
                if current_item['start_speed'] > self.min_suspicious_speed:
                    is_suspicious = True
                    suspicious_reasons.append(f"High initial speed ({current_item['start_speed'] * 3.6:.1f} km/h)")

                # INDICATOR 2: Large spatial gap between stay and trip start
                if distance > self.min_gap_distance:
                    is_suspicious = True
                    suspicious_reasons.append(f"Large distance from previous stay ({distance:.0f} m)")

                # INDICATOR 3: Check acceleration pattern (too abrupt)
                if len(trip_data) >= 3:
                    first_speeds = trip_data.iloc[:3]['effective_speed'].values
                    if first_speeds[0] > 0.5 and first_speeds[0] < first_speeds[1] < first_speeds[2]:
                        speed_increase_ratio = first_speeds[2] / first_speeds[0] if first_speeds[0] > 0 else 999
                        if speed_increase_ratio > 3:  # Speed triples quickly
                            is_suspicious = True
                            suspicious_reasons.append(f"Abnormal acceleration (ratio: {speed_increase_ratio:.1f})")

                # INDICATOR 4: Check if the first heading is immediately away from the stay
                if 'heading' in trip_data.columns and trip_data.iloc[0]['heading'] > 0:
                    # Calculate ideal heading from stay to first point
                    ideal_heading = self.calculate_bearing(
                        (previous_item['lat'], previous_item['lon']),
                        (current_item['start_lat'], current_item['start_lon'])
                    )

                    # Get actual initial heading
                    actual_heading = trip_data.iloc[0]['heading']

                    # Calculate difference (normalized to 0-180)
                    heading_diff = min(abs(ideal_heading - actual_heading),
                                       360 - abs(ideal_heading - actual_heading))

                    # If heading is not toward the previous stay, that's suspicious
                    if heading_diff < 45:  # Within 45 degrees of the ideal path
                        is_suspicious = True
                        suspicious_reasons.append(
                            f"Initial heading not from previous stay ({heading_diff:.0f}° difference)")

                # INDICATOR 5: Check if this is the first trip from home (regardless of time)
                is_from_home = previous_item['is_home']
                is_first_trip_of_day = True  # We'll determine if this is the first trip of the day

                # Check if this is the first trip of the day by looking at earlier items
                for earlier_item in day_items[:i]:
                    if earlier_item['type'] == 'trip':
                        is_first_trip_of_day = False
                        break

                if is_from_home and is_first_trip_of_day:
                    # First trip from home is likely to have sensor wake-up issues
                    suspicious_probability = 0.7 if is_suspicious else 0.3
                    if suspicious_probability > 0.5:
                        is_suspicious = True
                        suspicious_reasons.append("First trip from home location")
                    elif is_suspicious:
                        # If already suspicious for other reasons, add this as supporting evidence
                        suspicious_reasons.append("Departure from home location")

                # If suspicious, record for reconstruction
                if is_suspicious:
                    issue = {
                        'date': date,
                        'trip_id': trip_id,
                        'mode': current_item['mode'],
                        'stay_end': previous_item['end_time'],
                        'trip_start': current_item['start_time'],
                        'time_gap': time_gap,
                        'distance': distance,
                        'is_from_home': is_from_home,
                        'reasons': suspicious_reasons
                    }
                    detected_issues.append(issue)

                    # If we're very confident (multiple indicators), reconstruct missing data
                    if len(suspicious_reasons) >= 2 or (is_from_home):
                        self.reconstruct_trip_start(
                            trip_id=trip_id,
                            stay_end_time=previous_item['end_time'],
                            stay_location=(previous_item['lat'], previous_item['lon']),
                            trip_mode=current_item['mode'],
                            trip_start_location=(current_item['start_lat'], current_item['start_lon']),
                            trip_start_time=current_item['start_time']
                        )
                        reconstructed_trips += 1

        # Print summary
        print(f"Detected {len(detected_issues)} trips with potential missed starts")
        print(f"Reconstructed beginnings for {reconstructed_trips} trips")

        if detected_issues:
            print("\nDetailed issues:")
            for issue in detected_issues:
                print(f"  Trip {issue['trip_id']} on {issue['date']}: "
                      f"Mode: {issue['mode']}, From home: {issue['is_from_home']}")
                for reason in issue['reasons']:
                    print(f"   - {reason}")
                if issue['is_from_home']:
                    print(f"   * Trip appears to start {issue['time_gap']:.0f} seconds and "
                          f"{issue['distance']:.0f} meters from home")

        return detected_issues

    def identify_home_locations(self):
        """
        Identifies likely home locations based on overnight stays and morning departures.
        Returns a list of (lat, lon) coordinates of probable home locations.
        """
        home_candidates = []
        day_first_locations = []
        overnight_stays = []

        # Group by date
        for date, day_data in self.df.groupby(self.df['timestamp'].dt.date):
            day_data = day_data.sort_values('timestamp')

            # First location of the day (first stationary period)
            first_stationary = day_data[day_data['stay_id'] >= 0]
            if len(first_stationary) > 0:
                first_stay_id = first_stationary['stay_id'].iloc[0]
                first_stay = day_data[day_data['stay_id'] == first_stay_id]
                if len(first_stay) > 0:
                    first_lat = first_stay['latitude'].mean()
                    first_lon = first_stay['longitude'].mean()
                    day_first_locations.append((first_lat, first_lon))

            # Last location of the day
            last_stationary = day_data[day_data['stay_id'] >= 0]
            if len(last_stationary) > 0:
                last_stay_id = last_stationary['stay_id'].iloc[-1]
                last_stay = day_data[day_data['stay_id'] == last_stay_id]
                if len(last_stay) > 0:
                    last_lat = last_stay['latitude'].mean()
                    last_lon = last_stay['longitude'].mean()

                    # Check if this ends late in the day
                    last_hour = last_stay['timestamp'].iloc[-1].hour
                    if last_hour >= 22:  # After 10 PM
                        overnight_stays.append((last_lat, last_lon))

        # Cluster locations to find frequent locations
        all_candidates = day_first_locations + overnight_stays
        if not all_candidates:
            return []

        # Simple clustering - group nearby points
        clusters = []
        for loc in all_candidates:
            # Check if close to an existing cluster
            found_cluster = False
            for i, cluster in enumerate(clusters):
                if self.is_same_location(loc, cluster['center']):
                    clusters[i]['points'].append(loc)
                    # Recalculate center
                    points = clusters[i]['points']
                    clusters[i]['center'] = (
                        sum(p[0] for p in points) / len(points),
                        sum(p[1] for p in points) / len(points)
                    )
                    clusters[i]['count'] += 1
                    found_cluster = True
                    break

            if not found_cluster:
                clusters.append({
                    'center': loc,
                    'points': [loc],
                    'count': 1
                })

        # Sort by frequency
        clusters.sort(key=lambda x: x['count'], reverse=True)

        # Return the centers of the top clusters
        home_locations = [cluster['center'] for cluster in clusters[:3]]  # Top 3 candidates

        print(f"Identified {len(home_locations)} potential home locations")
        return home_locations

    def is_same_location(self, loc1, loc2, threshold=50):
        """
        Checks if two locations are within a threshold distance of each other.

        Args:
            loc1, loc2: Tuples of (latitude, longitude)
            threshold: Distance threshold in meters

        Returns:
            Boolean indicating if locations are considered the same
        """
        try:
            distance = great_circle(loc1, loc2).meters
            return distance <= threshold
        except:
            return False

    def calculate_bearing(self, point1, point2):
        """
        Calculate the bearing from point1 to point2.

        Args:
            point1, point2: Tuples of (latitude, longitude) in decimal degrees

        Returns:
            Bearing in degrees (0-360, where 0 is North)
        """
        import math

        lat1 = math.radians(point1[0])
        lat2 = math.radians(point2[0])
        diff_long = math.radians(point2[1] - point1[1])

        x = math.sin(diff_long) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1) * math.cos(lat2) * math.cos(diff_long))

        initial_bearing = math.atan2(x, y)
        initial_bearing = math.degrees(initial_bearing)

        # Normalize to 0-360
        bearing = (initial_bearing + 360) % 360

        return bearing

    def reconstruct_trip_start(self, trip_id, stay_end_time, stay_location, trip_mode,
                               trip_start_location, trip_start_time):
        """
        Reconstructs the missing beginning of a trip by inferring the most likely path
        and generates synthetic points to fill the gap.

        Args:
            trip_id: ID of the trip to reconstruct
            stay_end_time: When the previous stay ended
            stay_location: (lat, lon) of the previous stay
            trip_mode: Transportation mode of the trip ('walking', 'bicycle', 'car', etc.)
            trip_start_location: (lat, lon) where the trip recording actually started
            trip_start_time: When the trip recording actually started
        """
        print(f"Reconstructing beginning of trip {trip_id}...")

        # Get the trip data
        trip_data = self.df[self.df['trip_id'] == trip_id].copy()
        if len(trip_data) == 0:
            print(f"  Cannot reconstruct - trip {trip_id} not found")
            return

        # Calculate the time gap
        time_gap = (trip_start_time - stay_end_time).total_seconds()
        if time_gap <= 0:
            print(f"  Cannot reconstruct - no time gap detected")
            return

        # Calculate the distance gap
        stay_lat, stay_lon = stay_location
        trip_start_lat, trip_start_lon = trip_start_location
        distance_gap = great_circle(stay_location, trip_start_location).meters

        print(f"  Gap: {time_gap:.1f} seconds, {distance_gap:.1f} meters")

        # Determine reasonable speed based on mode
        avg_speeds = {
            'walking': 1.4,  # m/s (5 km/h)
            'bicycle': 4.2,  # m/s (15 km/h)
            'car': 8.3,  # m/s (30 km/h)
            'bus': 5.5,  # m/s (20 km/h)
            'train': 11.1,  # m/s (40 km/h)
            'unknown': 5.0  # m/s (default)
        }

        # Get reasonable speed for this mode
        inferred_speed = avg_speeds.get(trip_mode, avg_speeds['unknown'])

        # Calculate reasonable time to cover the distance
        reasonable_time = distance_gap / inferred_speed

        # Determine the most likely start time
        if reasonable_time > time_gap:
            # If the gap is shorter than expected, assume quicker movement
            actual_time = time_gap
            actual_speed = distance_gap / time_gap
            print(f"  Gap shorter than expected - assuming faster speed of {actual_speed * 3.6:.1f} km/h")
        else:
            # If the gap is longer than needed, assume some preparation time before departure
            prep_time = min(60, time_gap - reasonable_time)  # Up to 1 minute prep time
            actual_time = reasonable_time + prep_time
            actual_speed = distance_gap / reasonable_time
            print(f"  Assuming {prep_time:.1f} seconds of preparation time before departure")

        # Inferred departure time
        inferred_departure = trip_start_time - pd.Timedelta(seconds=actual_time)

        # Number of synthetic points to generate
        num_points = max(3, int(actual_time / 15))  # At least 3 points, approx one every 15 seconds

        # Generate synthetic points
        synthetic_points = []

        # Create acceleration curve - start slow and gradually speed up
        speeds = []
        max_speed = actual_speed * 1.1  # Allow for slightly higher peak speed

        for i in range(num_points):
            # S-curve for acceleration
            progress = i / (num_points - 1)
            if progress < 0.2:  # First 20% - slow acceleration
                speed_factor = progress * 2.5  # 0 to 0.5
            elif progress > 0.8:  # Last 20% - maintain speed
                speed_factor = 1.0
            else:  # Middle 60% - main acceleration
                speed_factor = 0.5 + (progress - 0.2) * (0.5 / 0.6)  # 0.5 to 1.0

            speeds.append(max_speed * speed_factor)

        # Calculate positions along the path
        total_distance = 0
        times = []

        # Generate times first
        for i in range(num_points):
            if i == 0:
                times.append(inferred_departure)
            else:
                # Distance covered in this segment
                segment_speed = (speeds[i - 1] + speeds[i]) / 2  # Average speed for segment
                segment_distance = segment_speed * (actual_time / num_points)
                total_distance += segment_distance

                # Proportion of total distance
                distance_proportion = total_distance / distance_gap
                time_proportion = min(1.0, distance_proportion)  # Cap at 100%

                # Calculate time for this point
                point_time = inferred_departure + pd.Timedelta(seconds=time_proportion * actual_time)
                times.append(point_time)

        # Linear interpolation for positions
        for i in range(num_points):
            progress = i / (num_points - 1)

            # Interpolate position
            lat = stay_lat + progress * (trip_start_lat - stay_lat)
            lon = stay_lon + progress * (trip_start_lon - stay_lon)

            # Create synthetic point - include only columns from original data
            point = {}

            # Use first point of trip as template
            template_point = trip_data.iloc[0]

            # Copy all columns from template
            for col in template_point.index:
                if col in ['locationId', 'userId', 'timestamp', 'latitude', 'longitude',
                           'isMoving', 'speed', 'effective_speed', 'trip_id', 'predicted_mode']:
                    # These we'll set specifically
                    continue
                else:
                    point[col] = template_point[col]

            # Add specific values
            point['locationId'] = f"synthetic_{trip_id}_{i}"
            point['userId'] = template_point['userId'] if 'userId' in template_point else None
            point['timestamp'] = times[i]
            point['latitude'] = lat
            point['longitude'] = lon
            point['isMoving'] = i > 0  # First point stationary, rest moving
            point['speed'] = speeds[i]
            point['effective_speed'] = speeds[i]
            point['trip_id'] = trip_id
            point['predicted_mode'] = trip_mode
            point['synthetic'] = True

            # Add distance for this point
            if i == 0:
                point['distance'] = 0
            else:
                # Calculate distance from previous synthetic point
                prev_lat = synthetic_points[i - 1]['latitude']
                prev_lon = synthetic_points[i - 1]['longitude']
                point['distance'] = great_circle((prev_lat, prev_lon), (lat, lon)).meters

            # Add time_diff
            if i == 0:
                point['time_diff'] = 0
            else:
                point['time_diff'] = (times[i] - times[i - 1]).total_seconds()

            synthetic_points.append(point)

        # Create DataFrame from synthetic points
        synthetic_df = pd.DataFrame(synthetic_points)

        # Make sure columns match original DataFrame
        for col in self.df.columns:
            if col not in synthetic_df.columns:
                if col == 'synthetic':
                    synthetic_df[col] = True
                else:
                    synthetic_df[col] = None

        # Add synthetic flag column if it doesn't exist
        if 'synthetic' not in self.df.columns:
            self.df['synthetic'] = False

        # Combine with original dataframe
        self.df = pd.concat([self.df, synthetic_df], ignore_index=True)

        # Sort by timestamp
        self.df = self.df.sort_values('timestamp')

        print(f"  Added {num_points} synthetic points from {inferred_departure.strftime('%H:%M:%S')} "
              f"to {trip_start_time.strftime('%H:%M:%S')}")

        # Update metadata of original trip points to note they're part of a reconstructed trip
        self.df.loc[self.df['trip_id'] == trip_id, 'has_reconstructed_start'] = True

        return synthetic_df

    def visualize_with_reconstructed_segments(self, trip_id):
        """
        Creates a specialized visualization that highlights reconstructed segments of a trip.
        """
        # Get trip data
        trip_data = self.df[self.df['trip_id'] == trip_id].copy()

        if len(trip_data) == 0:
            print(f"Trip {trip_id} not found")
            return None

        # Check if this trip has reconstructed segments
        has_synthetic = 'synthetic' in trip_data.columns and trip_data['synthetic'].any()

        if not has_synthetic:
            print(f"Trip {trip_id} doesn't have any reconstructed segments")
            return None

        # Separate original and synthetic points
        synthetic_points = trip_data[trip_data['synthetic'] == True]
        original_points = trip_data[trip_data['synthetic'] != True]

        print(
            f"Trip {trip_id} has {len(synthetic_points)} reconstructed points and {len(original_points)} original points")

        # Create map
        center_lat = trip_data['latitude'].mean()
        center_lon = trip_data['longitude'].mean()
        trip_map = folium.Map(location=[center_lat, center_lon], zoom_start=14)

        # Get trip info
        if 'predicted_mode' in trip_data.columns:
            mode = trip_data['predicted_mode'].mode().iloc[0]
        elif 'activityType' in trip_data.columns:
            mode = trip_data['activityType'].mode().iloc[0]
        else:
            mode = 'unknown'

        start_time = trip_data['timestamp'].min()
        end_time = trip_data['timestamp'].max()

        # Calculate duration
        duration_min = (end_time - start_time).total_seconds() / 60

        # Calculate distance if available
        if 'distance' in trip_data.columns:
            distance_km = trip_data['distance'].sum() / 1000
        else:
            # Approximate distance from coordinates
            coords = list(zip(trip_data['latitude'], trip_data['longitude']))
            distance_km = sum(great_circle(coords[i], coords[i + 1]).kilometers
                              for i in range(len(coords) - 1))

        # Define colors
        synthetic_color = 'orange'
        original_color = {
            'walking': 'green',
            'bicycle': 'blue',
            'car': 'red',
            'bus': 'purple',
            'train': 'brown',
            'unknown': 'gray'
        }.get(mode, 'gray')

        # Add reconstructed segment with dashed line
        if len(synthetic_points) >= 2:
            reconstructed_coords = list(zip(synthetic_points['latitude'], synthetic_points['longitude']))
            folium.PolyLine(
                reconstructed_coords,
                color=synthetic_color,
                weight=4,
                opacity=0.8,
                dash_array='5, 10',  # Dashed line
                popup=f"Reconstructed segment ({len(synthetic_points)} points)"
            ).add_to(trip_map)

            # Mark start point
            folium.Marker(
                [synthetic_points.iloc[0]['latitude'], synthetic_points.iloc[0]['longitude']],
                popup=f"Reconstructed Start: {synthetic_points.iloc[0]['timestamp'].strftime('%H:%M:%S')}",
                icon=folium.Icon(color='orange', icon='play', prefix='fa')
            ).add_to(trip_map)

        # Add original segment
        if len(original_points) >= 2:
            original_coords = list(zip(original_points['latitude'], original_points['longitude']))
            folium.PolyLine(
                original_coords,
                color=original_color,
                weight=4,
                opacity=0.8,
                popup=f"Original {mode} segment ({len(original_points)} points)"
            ).add_to(trip_map)

            # Mark original first point
            folium.Marker(
                [original_points.iloc[0]['latitude'], original_points.iloc[0]['longitude']],
                popup=f"Original First Point: {original_points.iloc[0]['timestamp'].strftime('%H:%M:%S')}",
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(trip_map)

        # Mark end point
        folium.Marker(
            [trip_data.iloc[-1]['latitude'], trip_data.iloc[-1]['longitude']],
            popup=f"End: {end_time.strftime('%H:%M:%S')}",
            icon=folium.Icon(color='red', icon='stop', prefix='fa')
        ).add_to(trip_map)

        # Add title with trip info
        title_html = f'''
            <h3 align="center">Trip {trip_id} - {mode.capitalize()} with Reconstructed Beginning</h3>
            <p align="center">
                From {start_time.strftime('%Y-%m-%d %H:%M')} to {end_time.strftime('%H:%M')}
                <br>Duration: {duration_min:.1f} min, Distance: {distance_km:.2f} km
                <br><span style="color:orange;">▪▪▪▪▪</span> Reconstructed Segment &nbsp; 
                <span style="color:{original_color};">▪▪▪▪▪</span> Original Segment
            </p>
        '''
        trip_map.get_root().html.add_child(folium.Element(title_html))

        # Save map
        maps_dir = os.path.join(self.output_dir, 'reconstructed_maps')
        if not os.path.exists(maps_dir):
            os.makedirs(maps_dir)

        map_filename = os.path.join(maps_dir, f'reconstructed_trip_{trip_id}.html')
        trip_map.save(map_filename)
        print(f"Created visualization for reconstructed trip {trip_id}")

        # Create speed profile visualization
        self.create_speed_profile_with_reconstruction(trip_id, trip_data, synthetic_points, original_points)

        return map_filename

    def create_speed_profile_with_reconstruction(self, trip_id, trip_data, synthetic_points, original_points):
        """
        Creates a speed profile plot highlighting reconstructed segments.
        """
        # Make sure we have speed data
        if 'effective_speed' not in trip_data.columns:
            print(f"Trip {trip_id} doesn't have speed data for visualization")
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot speed profile with different colors for reconstructed vs original segments
        if len(synthetic_points) > 0:
            ax.plot(synthetic_points['timestamp'], synthetic_points['effective_speed'] * 3.6,
                    'o-', color='orange', alpha=0.8, label='Reconstructed')

        if len(original_points) > 0:
            ax.plot(original_points['timestamp'], original_points['effective_speed'] * 3.6,
                    'o-', color='blue', alpha=0.8, label='Original')

        # Add vertical line at transition point
        if len(synthetic_points) > 0 and len(original_points) > 0:
            transition_time = original_points['timestamp'].min()
            ax.axvline(transition_time, color='red', linestyle='--', alpha=0.7,
                       label='Sensor Wake-up')

            # Annotate
            ax.text(transition_time, ax.get_ylim()[1] * 0.9, 'Sensor Wake-up',
                    rotation=90, verticalalignment='top', horizontalalignment='right')

        # Format axes
        ax.set_title(f"Speed Profile for Trip {trip_id} with Reconstructed Beginning")
        ax.set_ylabel("Speed (km/h)")
        ax.set_xlabel("Time")

        # Format x-axis with time
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=1))

        # Add legend
        ax.legend(loc='upper right')

        # Add grid for readability
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        plots_dir = os.path.join(self.output_dir, 'reconstructed_plots')
        if not os.path.exists(plots_dir):
            os.makedirs(plots_dir)

        plt_filename = os.path.join(plots_dir, f'speed_profile_trip_{trip_id}.png')
        plt.savefig(plt_filename, dpi=300)
        plt.close(fig)
        print(f"Created speed profile visualization for reconstructed trip {trip_id}")

        return plt_filename

    def visualize_all_reconstructed_trips(self):
        """Visualize all trips that have reconstructed segments."""
        # Check if we have the synthetic flag
        if 'synthetic' not in self.df.columns:
            print("No reconstructed trips found (missing 'synthetic' column)")
            return []

        # Find all trips with synthetic data
        synthetic_data = self.df[self.df['synthetic'] == True]
        if len(synthetic_data) == 0:
            print("No reconstructed trips found")
            return []

        trip_ids = synthetic_data['trip_id'].unique()
        print(f"Found {len(trip_ids)} trips with reconstructed segments")

        # Visualize each trip
        visualizations = []
        for trip_id in trip_ids:
            map_file = self.visualize_with_reconstructed_segments(trip_id)
            if map_file:
                visualizations.append((trip_id, map_file))

        return visualizations

    def run_analysis(self):
        """Run the complete reconstruction analysis."""
        print(f"Starting trip reconstruction analysis...")

        # Make sure data is loaded
        if self.df is None or len(self.df) == 0:
            raise ValueError("No data loaded. Please load data first with load_data()")

        # Detect and reconstruct
        issues = self.detect_and_fix_missed_trip_starts()

        # Visualize results
        visualizations = self.visualize_all_reconstructed_trips()

        # Create results summary
        num_reconstructed = len([i for i in issues if any("distance" in r for r in i['reasons'])])

        results = {
            'total_points': len(self.df),
            'detected_issues': len(issues),
            'reconstructed_trips': num_reconstructed,
            'visualizations': visualizations
        }

        # Save results
        self.save_data(os.path.join(self.output_dir, 'reconstructed_data.pkl'))

        print("\nAnalysis complete!")
        print(f"Results saved to {self.output_dir}")

        return results


def main():
    """Command-line entry point"""
    parser = argparse.ArgumentParser(description='Trip Start Reconstruction Tool')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input CSV or pickle file with GPS tracking data')
    parser.add_argument('--output', '-o', type=str, default='trip_reconstruction_output',
                        help='Output directory for results')
    parser.add_argument('--visualize-only', '-v', action='store_true',
                        help='Only visualize existing reconstructed trips, don\'t detect new ones')

    args = parser.parse_args()

    try:
        # Create reconstructor
        reconstructor = TripStartReconstructor(input_file=args.input, output_dir=args.output)

        if args.visualize_only:
            # Just visualize existing reconstructions
            reconstructor.visualize_all_reconstructed_trips()
        else:
            # Run full analysis
            reconstructor.run_analysis()

    except Exception as e:
        print(f"ERROR: {e}")
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    main()