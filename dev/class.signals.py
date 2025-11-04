from biomechzoo.biomech_ops.filter_line import filter_line
from biomechzoo.conversion.csv2zoo_data import csv2zoo_data as csv
from scipy.signal import medfilt
import numpy as np

class SensorProcessor:
    """
    Processes sensor data (gyro, accel, mag) with filtering, calibration, and integration.
    Provides a fluent interface for chaining operations.
    """

    SENSOR_CHANNELS = {
        'gyro': ['Gyr_X', 'Gyr_Y', 'Gyr_Z'],
        'accel': ['Acc_X', 'Acc_Y', 'Acc_Z'],
        'mag': ['Mag_X', 'Mag_Y', 'Mag_Z']
    }

    def __init__(self, sensor_type: str, sample_frequency: int):
        """
        Initialize processor for a specific sensor type.

        Args:
            sensor_type: 'gyro', 'accel', or 'mag'
            sample_frequency: Sampling frequency in Hz
        """
        if sensor_type not in self.SENSOR_CHANNELS:
            raise ValueError(f"sensor_type must be one of {list(self.SENSOR_CHANNELS.keys())}")

        self.sensor_type = sensor_type
        self.sample_frequency = sample_frequency
        self.channels = self.SENSOR_CHANNELS[sensor_type]
        self.data = None
        self.dt = 1 / sample_frequency

    def load_data(self, data: dict) -> 'SensorProcessor':
        """Load sensor data. Returns self for method chaining."""
        self.data = data
        return self

    def filter(self, cutoff: int, order: int = 4, median_kernel: int = 5) -> 'SensorProcessor':
        """
        Apply median and lowpass filtering.

        Args:
            cutoff: Lowpass filter cutoff frequency
            order: Filter order
            median_kernel: Kernel size for median filter
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        filter_params = {
            'type': 'butter',
            'order': order,
            'cutoff': cutoff,
            'btype': 'low',
            'fs': self.sample_frequency
        }

        filtered_data = {}
        for ch in self.channels:
            med_filt = medfilt(self.data[ch]['line'], kernel_size=median_kernel)
            filt = filter_line(med_filt, filter_params)
            filtered_data[ch] = {'line': filt}

        self.data = filtered_data
        return self

    def zero_mean(self) -> 'SensorProcessor':
        """Remove bias by subtracting mean."""
        if self.data is None:
            raise ValueError("No data loaded.")

        zero_mean_data = {}
        for ch in self.channels:
            signal = self.data[ch]['line']
            zero_mean_data[ch] = {'line': signal - np.mean(signal)}

        self.data = zero_mean_data
        return self

    def calibrate(self, static_data: dict) -> 'SensorProcessor':
        """
        Calibrate using static trial data (removes bias).

        Args:
            static_data: Static calibration data
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        calibrated = {}
        for ch in self.channels:
            calibrated[ch] = {'line': self.data[ch]['line'] - np.mean(static_data[ch]['line'])}

        self.data = calibrated
        return self

    def integrate(self, times: int = 1) -> 'SensorProcessor':
        """
        Integrate data once or twice.

        Args:
            times: Number of integrations (1 or 2)
        """
        if self.data is None:
            raise ValueError("No data loaded.")

        if times not in [1, 2]:
            raise ValueError("times must be 1 or 2")

        integrated = {}
        for ch in self.channels:
            signal = self.data[ch]['line']
            first_int = np.cumsum(signal) * self.dt

            if times == 1:
                integrated[ch] = {'line': first_int}
            else:
                second_int = np.cumsum(first_int) * self.dt
                integrated[ch] = {'line': second_int}

        self.data = integrated
        return self

    def acc_orient(self) -> 'SensorProcessor':
        """Compute orientation from accelerometer data."""
        if self.sensor_type != 'accel':
            raise ValueError("acc_orient only works with accelerometer data")

        g = 9.81

        angle_X = np.degrees(np.arctan2(self.data['Acc_Y']['line'], self.data['Acc_Z']['line']))
        angle_Y = -np.degrees(np.arcsin(self.data['Acc_X']['line'] / g))
        angle_Z = np.zeros(len(angle_X))

        self.data = {
            'Angles_X': {'line': angle_X},
            'Angles_Y': {'line': angle_Y},
            'Angles_Z': {'line': angle_Z}
        }
        return self

    def get_data(self) -> dict:
        """Return processed data."""
        return self.data

    def reset(self) -> 'SensorProcessor':
        """Clear data."""
        self.data = None
        return self


# Usage example:
if __name__ == "__main__":
    # Single workflow
    processor = SensorProcessor('gyro', sample_frequency=100)

    data1 = csv('/Users/joshualowery/Desktop/EDKP_616/IMU-Fundamentals/data/euler_rotation.csv')
    processed = (processor
                 .load_data(data1)
                 .filter(cutoff=10)
                 .zero_mean()
                 .integrate(times=1)
                 .get_data())

    # Reuse for another dataset
    data2 = csv('/Users/joshualowery/Desktop/EDKP_616/IMU-Fundamentals/data/mag_field.csv')
    processed2 = (processor
                  .reset()
                  .load_data(data2)
                  .filter(cutoff=10)
                  .zero_mean()
                  .get_data())