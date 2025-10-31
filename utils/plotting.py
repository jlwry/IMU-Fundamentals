import matplotlib.pyplot as plt
import numpy as np

def plot_xyz(data: dict | list[dict], div_time: int, tlabel: str, ylabel: str, sensor_type: str | list[str], label = 'data') -> None:

    """
    Makes a figure with three subplots (X, Y, Z) for a given signal:

    Inputs:
    - data: dict                = n x 3 dictionary containing sensor data
    - div_time: int             = integer by which fs is divided to get desired time
    - tlabel: str               = time units
    - ylabel: str               = label for the plot y-axis
    - sensor_type: str or list  = specifies the data type
    - label: str                = label for the plot data lines

    Outputs:
    - None // creates the figure
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

    if len(data) == 3:
        axs[0].plot(time, data[1][f'{prefixes[1]}_X']['line'], color="red", linestyle='--', linewidth=2, label = label[1])
        axs[1].plot(time, data[1][f'{prefixes[1]}_Y']['line'], color="green", linestyle='--', linewidth=2, label =label[1])
        axs[2].plot(time, data[1][f'{prefixes[1]}_Z']['line'], color="blue", linestyle='--', linewidth=2, label = label[1])
        axs[0].plot(time, data[2][f'{prefixes[2]}_X']['line'], color="red", linestyle=':', linewidth=2, label = label[2])
        axs[1].plot(time, data[2][f'{prefixes[2]}_Y']['line'], color="green", linestyle=':', linewidth=2, label =label[2])
        axs[2].plot(time, data[2][f'{prefixes[2]}_Z']['line'], color="blue", linestyle=':', linewidth=2, label = label[2])

    axs[0].set_ylabel(f'X {ylabel}', fontsize=12)
    axs[1].set_ylabel(f'Y {ylabel}', fontsize=12)
    axs[2].set_ylabel(f'Z {ylabel}', fontsize=12)
    axs[2].set_xlabel(f'Time ({tlabel})', fontsize=12)

    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

    plt.tight_layout()
    plt.show()


def visualize(data_path: str, visualizer_path: str) -> None:
    """
    Uses an open source IMU visualizer: https://github.com/jlwry/imu-visualization:

    Inputs:
    - data_path : str           = path to data file
    - visualizer_path : str     = path to visualizer file

    Outputs:
    - None
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