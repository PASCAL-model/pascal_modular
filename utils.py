"""Utility functions for PASCAL simulation."""

import numpy as np
import pandas as pd


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def flatten_list(xss):
    """Flatten a list of lists into a single list.

    Args:
        xss: A list of lists

    Returns:
        A flattened list containing all elements
    """
    return [x for xs in xss for x in xs]


def flatten_dict(data):
    """Flatten a nested dictionary structure into records.

    Args:
        data: Dictionary with nested structure

    Returns:
        List of flattened dictionaries (records)
    """
    records = []
    for key, value in data.items():
        flat = {'key': key}
        for k, v in value.items():
            if isinstance(v, dict):
                for subk, subv in v.items():
                    flat[f'{k}_{subk}'] = subv
            else:
                flat[k] = v
        records.append(flat)
    return records


def load_locations_csv(path):
    """Load a list of [lon, lat] start locations from a two-column CSV
    (header: lon,lat) - see PascalSimulation's start_locations constructor
    arg.

    Args:
        path: Path to a CSV file with 'lon' and 'lat' columns.

    Returns:
        List of [lon, lat] pairs, in file order.
    """
    df = pd.read_csv(path)
    return df[["lon", "lat"]].values.tolist()


def points_within_distance(pt_ll, list_ll, threshold):
    """Find points within a threshold distance of a reference point.

    Calculates the great circle distance (haversine formula) between a single
    point and a list of points on Earth's surface, returning a boolean array
    indicating which points are within the threshold distance.

    Args:
        pt_ll: Reference point as [longitude, latitude] in decimal degrees
        list_ll: Array of points with shape (N, 2) where columns are
                 [longitude, latitude] in decimal degrees
        threshold: Distance threshold in meters

    Returns:
        numpy boolean array of length N, True where distance <= threshold

    Notes:
        Uses the haversine formula for great circle distance calculation,
        assuming Earth radius of 6371000 meters.
    """
    # Convert to numpy arrays
    pt_ll = np.asarray(pt_ll)
    list_ll = np.asarray(list_ll)

    # Earth radius in meters
    R = 6371000.0

    # Convert to radians
    lon1 = np.radians(pt_ll[0])
    lat1 = np.radians(pt_ll[1])
    lon2 = np.radians(list_ll[:, 0])
    lat2 = np.radians(list_ll[:, 1])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))

    # Distance in meters
    distances = R * c

    # Return boolean array
    return distances <= threshold
