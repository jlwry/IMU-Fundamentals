from biomechzoo.conversion.csv2zoo_data import csv2zoo_data
from biomechzoo.biomech_ops.filter_line import filter_line
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import imufusion
import sys

# Functions included:
# reload
# filter_data
# zero_mean
# integrate
# acc_orient
# magdwick_filter
# plot_xyz
# calibrate

def filter_data(data : dict, cutoff: int, frequency: int, sensor_type='gyro'):

    """Filters data using median and low pass filters
    - data = dictionary containing x, y, z, sensor data
    - cutoff = lowpass cutoff frequency
    - frequency = collection frequency
    - sensor_type = specifies if data is from gyro or accel """

    sensor_channels = {
        'gyro': ['Gyr_X', 'Gyr_Y', 'Gyr_Z'],
        'accel': ['Acc_X', 'Acc_Y', 'Acc_Z'],
        'mag': ['Mag_X', 'Mag_Y', 'Mag_Z']
    }

    if sensor_type not in sensor_channels:
        raise ValueError("sensor_type must be 'gyro', 'accel', or 'mag'")

    channels = sensor_channels[sensor_type]

    filter_params = {
        'type': 'butter',
        'order': 4,
        'cutoff': cutoff,
        'btype': 'low',
        'fs': frequency
    }

    filtered_data = {}

    for ch in channels:
        med_filt = medfilt(data[ch]['line'], kernel_size=5)
        filt = filter_line(med_filt, filter_params)
        filtered_data[ch] = {'line': filt}

    return filtered_data

def zero_mean(data:dict):

    """Removes bias from data
    - data = dictionary containing x, y, z, sensor data"""

    zero_mean = {}
    for ch in data:
        signal = data[ch]['line']
        zero_mean_data = signal - np.mean(signal)
        zero_mean[ch] = {'line': zero_mean_data}
    return zero_mean

def integrate(data: dict, frequency: int, times: int = 1, sensor_type='gyro'):

    """Takes the first or second integral of data
    - data = dictionary containing x, y, z, sensor data
    - frequency = collection frequency
    - times = specifies number of times to integrate
    - sensor_type = specifies if data is from gyro or accel"""

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

def acc_orient(data: dict):

    """ Determines orientation of the sensor
    using only accelerometer data"""

    g = 9.81

    Angle_X = np.degrees(np.arctan2(data['Acc_Y']['line'], data['Acc_Z']['line']))
    Angle_Y = np.degrees(np.arcsin(data['Acc_X']['line'] / g))
    Angle_Z = np.zeros(len(Angle_X))

    Angles = {
        'Angles_X': {'line': Angle_X},
        'Angles_Y': {'line': Angle_Y},
        'Angles_Z': {'line': Angle_Z}
    }

    return Angles

def madgwick_filter(csv: str, show_plot = True):

    """ Function implementation of the open-source
    Madgwick filter found here:
    https://github.com/xioTechnologies/Fusion"""

    data = np.genfromtxt(csv, delimiter=",", skip_header=1)

    timestamp = data[:, 0]/120
    gyroscope = data[:, 8:11]
    accelerometer = data[:, 5:8]

    # Process sensor data
    ahrs = imufusion.Ahrs()
    euler = np.empty((len(timestamp), 3))

    for index in range(len(timestamp)):
        ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], 1 / 120)
        euler[index] = ahrs.quaternion.to_euler()

    if show_plot == True:

        # Plot sensor data
        _, axes = plt.subplots(nrows=3, figsize=(8, 6), sharex=True)

        axes[0].plot(timestamp, gyroscope[:, 0], "tab:red", label="X")
        axes[0].plot(timestamp, gyroscope[:, 1], "tab:green", label="Y")
        axes[0].plot(timestamp, gyroscope[:, 2], "tab:blue", label="Z")
        axes[0].set_title("Gyroscope")
        axes[0].set_ylabel("Degrees/s")
        axes[0].grid()
        axes[0].legend()

        axes[1].plot(timestamp, accelerometer[:, 0], "tab:red", label="X")
        axes[1].plot(timestamp, accelerometer[:, 1], "tab:green", label="Y")
        axes[1].plot(timestamp, accelerometer[:, 2], "tab:blue", label="Z")
        axes[1].set_title("Accelerometer")
        axes[1].set_ylabel("g")
        axes[1].grid()
        axes[1].legend()

        # Plot Euler angles
        axes[2].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
        axes[2].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
        axes[2].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
        axes[2].set_title("Euler angles")
        axes[2].set_xlabel("Seconds")
        axes[2].set_ylabel("Degrees")
        axes[2].grid()
        axes[2].legend()

    elif show_plot == False:
        pass

    roll_hind = euler[:, 0]
    pitch_hind = euler[:, 1]
    yaw_hind = euler[:, 2]

    # Roll = about X
    # Pitch = about Y
    # Yaw = about Z

    return roll_hind, pitch_hind, yaw_hind



def plot_xyz(data: dict, div_time: int, tlabel: str, ylabel: str, sensor_type='gyro'):
    """
    Makes a figure with three subplots (X, Y, Z) for a given sensor type.
    - data: dict with keys like 'Gyr_X', 'Acc_Y', etc.
    - div_time: int to convert frames into time
    - tlabel: str, label for time axis
    - ylabel: str, label for variable plotted
    - sensor_type: 'gyro', 'accel', or 'mag'
    """

    sensor_map = {
        'gyro': 'Gyr',
        'accel': 'Acc',
        'mag': 'Mag',
        'angles': 'Angles'
    }

    if sensor_type not in sensor_map:
        raise ValueError("sensor_type must be 'gyro', 'accel', or 'mag'")

    prefix = sensor_map[sensor_type]

    time = np.arange(len(data[f'{prefix}_X']['line'])) / div_time

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(time, data[f'{prefix}_X']['line'], color="red", linewidth=2)
    axs[0].set_ylabel(f'X_{ylabel}', fontsize=12)
    axs[0].grid(True, linestyle='--', alpha=0.4)

    axs[1].plot(time, data[f'{prefix}_Y']['line'], color="green", linewidth=2)
    axs[1].set_ylabel(f'Y_{ylabel}', fontsize=12)
    axs[1].grid(True, linestyle='--', alpha=0.4)

    axs[2].plot(time, data[f'{prefix}_Z']['line'], color="blue", linewidth=2)
    axs[2].set_ylabel(f'Z_{ylabel}', fontsize=12)
    axs[2].set_xlabel(f'Time ({tlabel})', fontsize=12)
    axs[2].grid(True, linestyle='--', alpha=0.4)

    fig.suptitle(f'{sensor_type.capitalize()} Data', fontsize=14, y=0.95)
    plt.tight_layout()
    plt.show()

def calibrate(dynamic: dict, static: dict, sensor_type='gyro'):

    """Calibrates the dynamic according to bias identified in static sensor data.
    - data: IMU data from a dynamic trial
    - static: trial where the sensor isn't moving"""

    sensor_channels = {
        'gyro': ['Gyr_X', 'Gyr_Y', 'Gyr_Z'],
        'accel': ['Acc_X', 'Acc_Y', 'Acc_Z'],
        'mag': ['Mag_X', 'Mag_Y', 'Mag_Z']
    }

    if sensor_type not in sensor_channels:
        raise ValueError("sensor_type must be 'gyro', 'accel', or 'mag'")

    channels = sensor_channels[sensor_type]

    bias = {}
    for ch in channels:
        bias[ch] = np.mean(static[ch]['line'])

    calibrated_data = {}
    for ch in channels:
        calibrated_data[ch] = {'line': dynamic[ch]['line'] - bias[ch]}

    return calibrated_data


def visualize(data_path: str, visualizer_path: str):
    """ Uses an open source IMU data visualizer:
    https://github.com/jlwry/imu-visualization

    data_path = path to data you wish to visualize (or list of paths)
    visualizer_path = path to visualization repo
    """

    import sys
    import matplotlib
    import matplotlib.pyplot as plt

    original_backend = matplotlib.get_backend()
    matplotlib.use('TkAgg')  # Changed from 'notebook' - animations need TkAgg

    sys.path.append(visualizer_path)
    from vis_3D_rot_scikit_V4 import main

    try:
        csv_files = [data_path] if isinstance(data_path, str) else data_path
        main(csv_files)
    except KeyboardInterrupt:
        print("Animation stopped")
    finally:
        plt.close('all')
        matplotlib.use(original_backend)

if __name__ == '__main__':
    data_gyr = csv2zoo_data('data/overnight.csv')
    filtered_gyro_data = filter_data(data_gyr, cutoff = 1, frequency = 30, sensor_type ='gyro')
    plot_xyz(filtered_gyro_data, div_time = 108000, tlabel = '(hours)', ylabel='Orientation', sensor_type='gyro')

    data_acc = csv2zoo_data('data/still_1min.csv')
    filtered_acc_data = filter_data(data_acc, cutoff = 1, frequency = 120, sensor_type ='accel')
    plot_xyz(filtered_acc_data, div_time = 120, tlabel = '(seconds)', ylabel='Acceleration (g)', sensor_type='accel')