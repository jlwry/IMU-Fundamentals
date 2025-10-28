from biomechzoo.biomech_ops.filter_line import filter_line
from scipy.spatial.transform import Rotation as R
from scipy.signal import medfilt
import matplotlib.pyplot as plt
import numpy as np
import imufusion
import sys

def filter_data(data : dict, cutoff: int, sample_frequency: int, sensor_type='gyro'):

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
        'fs': sample_frequency
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
    Angle_Y = -np.degrees(np.arcsin(data['Acc_X']['line'] / g))
    Angle_Z = np.zeros(len(Angle_X))

    Angles = {
        'Angles_X': {'line': Angle_X},
        'Angles_Y': {'line': Angle_Y},
        'Angles_Z': {'line': Angle_Z}
    }

    return Angles

def plot_xyz(data, div_time: int, tlabel: str, ylabel: str, sensor_type='gyro', label = 'data'):
    """
    Makes a figure with three subplots (X, Y, Z) for a given sensor type.
    - data: dict or list of dicts with keys like 'Gyr_X', 'Acc_Y', etc.
    - div_time: int to convert frames into time
    - tlabel: str, label for time axis
    - ylabel: str, label for variable plotted
    - sensor_type: str 'gyro', 'accel', 'mag', 'angles', or 'results'
    - label: str, label for the data lines
    """

    sensor_map = {
        'gyro': 'Gyr',
        'accel': 'Acc',
        'mag': 'Mag',
        'angles': 'Angles',
        'results': 'Euler'
    }

    if isinstance(data, dict):
        data = [data]

    if isinstance(sensor_type, str):
        sensor_type = [sensor_type]

    if isinstance(label, str):
        label = [label]

    prefixes = [sensor_map[s] for s in sensor_type]

    time = np.arange(len(data[0][f'{prefixes[0]}_X']['line'])) / div_time

    fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

    axs[0].plot(time, data[0][f'{prefixes[0]}_X']['line'], color="red", linewidth=2, label = label[0])
    axs[1].plot(time, data[0][f'{prefixes[0]}_Y']['line'], color="green", linewidth=2, label = label[0])
    axs[2].plot(time, data[0][f'{prefixes[0]}_Z']['line'], color="blue", linewidth=2, label = label[0])

    if len(data) == 2:
        axs[0].plot(time, data[1][f'{prefixes[1]}_X']['line'], color="red", linestyle='--', linewidth=2, label = label[1])
        axs[1].plot(time, data[1][f'{prefixes[1]}_Y']['line'], color="green", linestyle='--', linewidth=2, label =label[1])
        axs[2].plot(time, data[1][f'{prefixes[1]}_Z']['line'], color="blue", linestyle='--', linewidth=2, label = label[1])

    axs[0].set_ylabel(f'X_{ylabel}', fontsize=12)
    axs[1].set_ylabel(f'Y_{ylabel}', fontsize=12)
    axs[2].set_ylabel(f'Z_{ylabel}', fontsize=12)
    axs[2].set_xlabel(f'Time ({tlabel})', fontsize=12)

    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

    # if len(sensor_type) == 1:
    #     fig.suptitle(f'{sensor_type[0].capitalize()} Data', fontsize=14, y=0.95)
    # else:
    #     fig.suptitle(f'{sensor_type[0].capitalize()} vs {sensor_type[1].capitalize()} Data', fontsize=14, y=0.95)

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


def quat_to_euler(data:dict):

    keys = ['Quat_W', 'Quat_X', 'Quat_Y', 'Quat_Z']

    quats = np.column_stack([
        data['Quat_W']['line'],
        data['Quat_X']['line'],
        data['Quat_Y']['line'],
        data['Quat_Z']['line']
    ])

    r = R.from_quat(quats[:, [1, 2, 3, 0]])
    euler = r.as_euler('zyx', degrees=True)

    return {
        'Euler_X': {'line': euler[:, 0]},
        'Euler_Y': {'line': euler[:, 1]},
        'Euler_Z': {'line': euler[:, 2]}
    }

def simple_madgwick_filter(csv: str, show_plot=True):

    """ Function implementation of the open-source
    Madgwick filter found here:
    https://github.com/xioTechnologies/Fusion"""

    data = np.genfromtxt(csv, delimiter=",", skip_header=1)

    timestamp = data[:, 0] / 120
    gyroscope = data[:, 8:11]
    accelerometer = data[:, 5:8]

    # Process sensor data
    ahrs = imufusion.Ahrs()
    euler = np.empty((len(timestamp), 3))

    for index in range(len(timestamp)):
        ahrs.update_no_magnetometer(gyroscope[index], accelerometer[index], 1 / 120)
        euler[index] = ahrs.quaternion.to_euler()

    if show_plot:

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

    else:
        pass

    roll_hind = euler[:, 0]
    pitch_hind = euler[:, 1]
    yaw_hind = euler[:, 2]

    # Roll = about X
    # Pitch = about Y
    # Yaw = about Z

    return roll_hind, pitch_hind, yaw_hind

def advanced_madgwick(csv:str, show_plot=True):

    """ Function implementation of the open-source
    Madgwick filter found here:
    https://github.com/xioTechnologies/Fusion

    ** note: the advanced Madgwick filter implements mag data"""

    data = np.genfromtxt(csv, delimiter=",", skip_header=1)

    sample_rate = 120

    timestamp = (data[:, 0])/120 # frames --> seconds unit convertion
    gyroscope = data[:, 8:11]
    accelerometer = (data[:, 5:8])/9.81 # m/s^2 --> g unit convertion
    magnetometer = (data[:, 11:14]) * 100 # G --> uT unit convertion

    # Instantiate algorithms
    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()

    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NWU,  # convention
        0.5,  # gain
        2000,  # gyroscope range
        10,  # acceleration rejection
        10,  # magnetic rejection
        5 * sample_rate,  # recovery trigger period = 5 seconds
    )

    # Process sensor data
    delta_time = np.diff(timestamp, prepend=timestamp[0])

    euler = np.empty((len(timestamp), 3))
    internal_states = np.empty((len(timestamp), 6))
    flags = np.empty((len(timestamp), 4))

    for index in range(len(timestamp)):
        gyroscope[index] = offset.update(gyroscope[index])

        ahrs.update(gyroscope[index], accelerometer[index], magnetometer[index], delta_time[index])

        euler[index] = ahrs.quaternion.to_euler()

        ahrs_internal_states = ahrs.internal_states
        internal_states[index] = np.array(
            [
                ahrs_internal_states.acceleration_error,
                ahrs_internal_states.accelerometer_ignored,
                ahrs_internal_states.acceleration_recovery_trigger,
                ahrs_internal_states.magnetic_error,
                ahrs_internal_states.magnetometer_ignored,
                ahrs_internal_states.magnetic_recovery_trigger,
            ]
        )

        ahrs_flags = ahrs.flags
        flags[index] = np.array(
            [
                ahrs_flags.initialising,
                ahrs_flags.angular_rate_recovery,
                ahrs_flags.acceleration_recovery,
                ahrs_flags.magnetic_recovery,
            ]
        )

    def plot_bool(axis, x, y, label):
        axis.plot(x, y, "tab:cyan", label=label)
        plt.sca(axis)
        plt.yticks([0, 1], ["False", "True"])
        axis.grid()
        axis.legend()

    if show_plot:

        # Plot Euler angles
        figure, axes = plt.subplots(nrows=11, sharex=True,  figsize=(12, 16), gridspec_kw={"height_ratios": [6, 1, 1, 2, 1, 1, 1, 2, 1, 1, 1]})

        figure.suptitle("Euler angles, internal states, and flags")

        axes[0].plot(timestamp, euler[:, 0], "tab:red", label="Roll")
        axes[0].plot(timestamp, euler[:, 1], "tab:green", label="Pitch")
        axes[0].plot(timestamp, euler[:, 2], "tab:blue", label="Yaw")
        axes[0].set_ylabel("Degrees")
        axes[0].grid()
        axes[0].legend()

        # Plot initialising flag
        plot_bool(axes[1], timestamp, flags[:, 0], "Initialising")

        # Plot angular rate recovery flag
        plot_bool(axes[2], timestamp, flags[:, 1], "Angular rate recovery")

        # Plot acceleration rejection internal states and flag
        axes[3].plot(timestamp, internal_states[:, 0], "tab:olive", label="Acceleration error")
        axes[3].set_ylabel("Degrees")
        axes[3].grid()
        axes[3].legend()

        plot_bool(axes[4], timestamp, internal_states[:, 1], "Accelerometer ignored")

        axes[5].plot(timestamp, internal_states[:, 2], "tab:orange", label="Acceleration recovery trigger")
        axes[5].grid()
        axes[5].legend()

        plot_bool(axes[6], timestamp, flags[:, 2], "Acceleration recovery")

        # Plot magnetic rejection internal states and flag
        axes[7].plot(timestamp, internal_states[:, 3], "tab:olive", label="Magnetic error")
        axes[7].set_ylabel("Degrees")
        axes[7].grid()
        axes[7].legend()

        plot_bool(axes[8], timestamp, internal_states[:, 4], "Magnetometer ignored")

        axes[9].plot(timestamp, internal_states[:, 5], "tab:orange", label="Magnetic recovery trigger")
        axes[9].grid()
        axes[9].legend()

        plot_bool(axes[10], timestamp, flags[:, 3], "Magnetic recovery")

        plt.show(block="dont_block" not in sys.argv)  # don't block when script run by CI

    else:
        pass

    roll_hind = euler[:, 0]
    pitch_hind = euler[:, 1]
    yaw_hind = euler[:, 2]

    # Roll = about X
    # Pitch = about Y
    # Yaw = about Z

    return roll_hind, pitch_hind, yaw_hind