import matplotlib.pyplot as plt
import numpy as np
import imufusion
import sys

def simple_madg(csv: str, gyro_column: int, accel_column: int, show_plot: bool) -> dict:

    """ Function implementation of the open-source Madgwick filter found here:
    https://github.com/xioTechnologies/Fusion

    Inputs:
    - csv: str             = path to csv file you wish to visualize
    - gyro_column: int     = column number (0-index) of the first column of gyro data
    - accel_column: int    = column number (0-index) of the first column of accelerometer data
    - show_plot: bool      = True if you wish to show the plot

    Outputs:
    - angles_simple: dict   = n x 3 dictionary containing resulting angles
    """

    data = np.genfromtxt(csv, delimiter=",", skip_header=1)

    timestamp = data[:, 0] / 120
    gyroscope = data[:, gyro_column:gyro_column + 3]
    accelerometer = (data[:, accel_column:accel_column + 3])/9.81 # m/s^2 --> g unit convertion

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

    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]

    angles_simple = {
        'Angles_X': {'line': roll},
        'Angles_Y': {'line': pitch},
        'Angles_Z': {'line': yaw},
    }

    return angles_simple

def advanced_madg(csv: str, gyro_column: int, accel_column: int, mag_column: int, mag_rej: int, acc_rej:int, show_plot: bool) -> dict | tuple:

    """ Function implementation of the open-source Madgwick filter found here:
    https://github.com/xioTechnologies/Fusion

    Inputs:
    - csv: str              = path to csv file you wish to visualize
    - gyro_column: int      = column number (0-index) of the first column of gyro data
    - accel_column: int     = column number (0-index) of the first column of accelerometer data
    - mag_column: int       = column number (0-index) of the first column of magnetic data
    - show_plot: bool       = True if you wish to show the plot

    Outputs:
    - angles_simple: dict   = n x 3 dictionary containing resulting angles

    * Note: the advanced Madgwick filter uses mag. data
    """

    data = np.genfromtxt(csv, delimiter=",", skip_header=1)

    sample_rate = 120

    timestamp = (data[:, 0])/120 # frames --> seconds unit convertion
    gyroscope = data[:, gyro_column:gyro_column + 3]
    accelerometer = (data[:, accel_column:accel_column + 3])/9.81 # m/s^2 --> g unit convertion
    magnetometer = (data[:, mag_column:mag_column + 3]) * 100 # G --> uT unit convertion

    # Instantiate algorithms
    offset = imufusion.Offset(sample_rate)
    ahrs = imufusion.Ahrs()

    ahrs.settings = imufusion.Settings(
        imufusion.CONVENTION_NWU,  # convention
        0.5,  # gain
        2000,  # gyroscope range
        acc_rej,  # acceleration rejection -- base = 10
        mag_rej,  # magnetic rejection -- base = 10
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
        # TODO: get the initializing index returned
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

    roll = euler[:, 0]
    pitch = euler[:, 1]
    yaw = euler[:, 2]

    angles_adv = {
        'Angles_X': {'line': roll},
        'Angles_Y': {'line': pitch},
        'Angles_Z': {'line': yaw},
    }

    return angles_adv, flags[:, 0]