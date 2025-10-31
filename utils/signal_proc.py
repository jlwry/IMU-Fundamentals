from biomechzoo.biomech_ops.filter_line import filter_line
from scipy.signal import medfilt
import numpy as np

def filter_data(data : dict, cutoff: int, sample_frequency: int, sensor_type: str) -> dict:

    """
    Filters data using median and low pass filters:

    Inputs:
    - data: dict            = n x 3 dictionary containing sensor data
    - cutoff: int           = lowpass filter cutoff frequency
    - sample_frequency:int  = collection frequency
    - sensor_type: str      = specifies if data is from gyro., mag., or accel.

    Outputs:
    - filtered_data: dict   = n x 3 dictionary containing filtered sensor data
    """
    # TODO: using the median + low pass is likely over filtering the data
    sensor_channels = {
        'gyro': ['Gyr_X', 'Gyr_Y', 'Gyr_Z'],
        'accel': ['Acc_X', 'Acc_Y', 'Acc_Z'],
        'mag': ['Mag_X', 'Mag_Y', 'Mag_Z']
    }

    if sensor_type not in sensor_channels:
        raise ValueError("sensor_type must be 'gyro.', 'accel.', or 'mag.'")

    channels = sensor_channels[sensor_type]

    filter_params = {
        'type': 'butter',
        'order': 4,
        'cutoff': cutoff,
        'btype': 'low',
        'fs': sample_frequency
    }

    filtered_data = {}

    for ch in channels:
        med_filt = medfilt(data[ch]['line'], kernel_size=5)
        filt = filter_line(med_filt, filter_params)
        filtered_data[ch] = {'line': filt}

    return filtered_data

def zero_mean(data:dict) -> dict:

    """
    Removes bias from sensor data by subtracting the mean of the signal:

    Inputs:
    - data: dict            = n x 3 dictionary containing sensor data

    Outputs:
    - zero_mean_data: dict  = n x 3 dictionary containing zero-mean sensor data
    """

    zero_mean_data = {}
    for ch in data:
        signal = data[ch]['line']
        null_mean = signal - np.mean(signal)
        zero_mean_data[ch] = {'line': null_mean}
    return zero_mean_data

def integrate(data: dict, frequency: int, times: int = 1, sensor_type='gyro') -> dict:

    """
    Takes the first or second integral of a given sensor's data:

    Inputs:
    - data: dict            = n x 3 dictionary containing sensor data that you wish to integrate
    - frequency: int        = sensor sampling frequency
    - times: int            = specifies number of times to integrate (either 1 or 2)
    - sensor_type: str      = specifies if data is from gyro or accel

    Outputs:
    - integrated_data: dict = n x 3 dictionary containing integrated sensor data
    """

    dt = 1 / frequency
    integrated_data = {}

    sensor_channels = {
        'gyro': ['Gyr_X', 'Gyr_Y', 'Gyr_Z'],
        'accel': ['Acc_X', 'Acc_Y', 'Acc_Z'],
        'mag': ['Mag_X', 'Mag_Y', 'Mag_Z']
    }

    if sensor_type not in sensor_channels:
        raise ValueError("sensor_type must be 'gyro', 'accel', or 'mag'")

    channels = sensor_channels[sensor_type]

    for ch in channels:
        signal = data[ch]['line']
        first_int = np.cumsum(signal) * dt
        if times == 1:
            integrated_data[ch] = {'line': first_int}
        elif times == 2:
            second_int = np.cumsum(first_int) * dt
            integrated_data[ch] = {'line': second_int}

    return integrated_data

def acc_orient(data: dict) -> dict:

    """
    Determines orientation of the sensor using only accelerometer data:

    Inputs:
    - data: dict    = n x 3 dictionary containing sensor data

    Outputs:
    - angles: dict  = n x 3 dictionary containing sensor orientation

    equations are taken from: https://www.youtube.com/watch?v=p7tjtLkIlFo
    """

    g = 9.81

    angle_X = np.degrees(np.arctan2(data['Acc_Y']['line'], data['Acc_Z']['line']))
    angle_Y = -np.degrees(np.arcsin(data['Acc_X']['line'] / g))
    angle_Z = np.zeros(len(angle_X))

    angles = {
        'Angles_X': {'line': angle_X},
        'Angles_Y': {'line': angle_Y},
        'Angles_Z': {'line': angle_Z}
    }

    return angles

def calibrate(dynamic: dict, static: dict, sensor_type: str) -> dict:

    """
    Calibrates dynamic data. Takes a static trial, computes bias, and removes it from the dynamic data:

    Inputs:
    - dynamic: dict             = n x 3 dictionary containing dynamic data
    - static: dict              = n x 3 dictionary containing static data
    - sensor_type: str          = specifies what sensor type is being processed

    Outputs:
    - calibrated_data: dict     = n x 3 dictionary containing sensor data with bias removed
    """

    sensor_channels = {
        'gyro': ['Gyr_X', 'Gyr_Y', 'Gyr_Z'],
        'accel': ['Acc_X', 'Acc_Y', 'Acc_Z'],
        'mag': ['Mag_X', 'Mag_Y', 'Mag_Z']
    }

    if sensor_type not in sensor_channels:
        raise ValueError("sensor_type must be 'gyro', 'accel', or 'mag'")

    channels = sensor_channels[sensor_type]

    calibrated_data = {}

    for ch in channels:
        calibrated_data[ch] = {'line': dynamic[ch]['line'] - np.mean(static[ch]['line'])}

    return calibrated_data

def clip(data1: dict, data2: dict, data3: dict) -> tuple[dict, dict, dict]:

    min_len = float('inf')
    for data in [data1, data2, data3]:
        for key in data:
            if isinstance(data[key], dict) and 'line' in data[key]:
                min_len = min(min_len, len(data[key]['line']))

    for data in [data1, data2, data3]:
        for key in data:
            if isinstance(data[key], dict):
                for subkey in data[key]:
                    if isinstance(data[key][subkey], (list, np.ndarray)):
                        data[key][subkey] = data[key][subkey][:min_len]

    return data1, data2, data3