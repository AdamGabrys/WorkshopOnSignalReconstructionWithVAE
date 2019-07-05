import numpy as np
import matplotlib.pyplot as plt  # To visualize the data

"""
Generates a sinus signals of different frequencies.
"""


class SignalGenerator:
    def __init__(self, mean_freq: float = 0.5, std_freq: float = 0.05,
                 mean_phase: float = 3, std_phase: float = 1.5):
        """
        Initialize generator with mean and standard deviation of Gaussian distribution from which
        multipliers of sinusoidal signals will be sampled.
        Args:
            mean: The mean of Gaussian
            std: The standard deviation of Gaussian
        """
        self._mean_freq = mean_freq
        self._std_freq = std_freq

        self._mean_phase = mean_phase
        self._std_phase = std_phase

    @property
    def mean_freq(self):
        return self._mean_freq

    @property
    def std_freq(self):
        return self._std_freq

    @property
    def mean_phase(self):
        return self._mean_phase

    @property
    def std_phase(self):
        return self._std_phase

    def generate_signal(self, length: int, noise_factor: float = 0.1) -> (float, np.ndarray):
        """
        Generate noisy signal with multiplier that controls signal frequency
        sampled from Gaussian distribution
        Args:
            length: The length of the signal to generate
            noise_factor: ammount of noise that will be applied to the signal
        Returns:
            (multiplier that controls sin frequency, y=sin(x*multiplier),
            generated signal)

        """

        multiplier = np.random.normal(self._mean_freq, self._std_freq)
        phase = np.random.normal(self._mean_phase, self._std_phase)
        signal = np.sin(np.arange(0, length) * multiplier + phase)
        noise = np.random.rand(length) * noise_factor
        return multiplier, phase, signal + noise

    def generate_signals(self, length: int, num: int) -> (np.ndarray, np.ndarray):
        """
        Generates sequence of noisy signals with multiplier sampled from Gaussian distribution
        Args:
            length: The length of the signal to generate
            num: Number of signals to generate

        Returns:
            (array(num) of multipliers,  array(num,length) of generated signals

        """

        multipliers_and_signals = [self.generate_signal(length) for _ in range(num)]
        return tuple(np.vstack(f_s) for f_s in zip(*multipliers_and_signals))

    def generate_signals_from_gaussian(self, num: int, mean_freq: float, std_freq: float,
                                       mean_phase, std_phase,
                                       length: int = 50) -> (np.ndarray, np.ndarray):
        self._mean_freq = mean_freq
        self._std_freq = std_freq

        self._mean_phase = mean_phase
        self._std_phase = std_phase
        return self.generate_signals(num=num, length=length)

    # Lets define function to plot generated data.
    def plot_signals_with_multipliers(self, multipliers: float, phase: float, signals: np.ndarray,
                                      sample_size: int = 5, bins: int = 50,
                                      xlim_radius: float = 0.4, sin_range: int = 50):
        ax1 = plt.subplot(221)
        ax1.hist(multipliers, bins=bins)
        ax1.set_title("sinus function frequency histogram")
        ax1.set_xlabel("sin frequencies")
        ax1.set_ylabel("count")
        ax1.set_xlim(self._mean_freq - xlim_radius - 2 * self.std_freq,
                     self._mean_freq + xlim_radius + 2 * self.std_freq)

        ax2 = plt.subplot(222)
        ax2.hist(phase, bins=bins)
        ax2.set_title("sinus function phase histogram")
        ax2.set_xlabel("sin phase")
        ax2.set_ylabel("count")
        ax2.set_xlim(self._mean_phase - xlim_radius - 2 * self._std_phase,
                     self._mean_phase + xlim_radius + 2 * self._std_phase)

        ax3 = plt.subplot(212)
        for i in np.arange(sample_size):
            ax3.plot(np.arange(sin_range), signals[i])
            ax3.set_xlabel("x")
            ax3.set_ylabel("y")
            ax3.set_title("Samples of generated sinus functions")

        fig = plt.gcf()
        fig.set_size_inches(12.5, 7.5)

        plt.show()
