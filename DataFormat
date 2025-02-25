# Data Requirements for Trip Start Reconstruction

The algorithm is designed to work with GPS location tracking data with these characteristics:

## Required Data Fields
- **Timestamps**: Datetime values showing when each point was recorded
- **Coordinates**: Latitude and longitude values (decimal format)

## Optional But Helpful Fields
- **Speed**: Movement speed (m/s or km/h)
- **Activity Type**: Labels like 'walking', 'in_vehicle', etc.
- **Movement Flag**: Boolean indicating if device was moving
- **Heading**: Direction of travel (degrees)
- **Accuracy**: GPS accuracy metrics (meters)

## Data Format
The algorithm expects data in CSV format (or pickle) with each row representing a single GPS point. The data does not need to be pre-processed with trip or stay information - the algorithm will handle this automatically. A typical input might look like:

```
timestamp,latitude,longitude,speed,accuracy,activityType,isMoving
2025-01-10T07:30:21.989Z,38.36723,26.13403,1.2,10.5,walking,True
2025-01-10T07:30:28.999Z,38.36716,26.13379,3.6,8.2,walking,True
```

## Data Collection Considerations
- Sampling rate should be reasonably consistent (typically every 1-30 seconds)
- Data should include both stationary periods and movement segments
- For best results, continuous tracking across multiple days helps identify home locations
- The algorithm is robust to missing data and can handle typical GPS noise

The tool is designed to be flexible and will work with data from most mobile tracking apps, fitness trackers, and custom GPS loggers.
