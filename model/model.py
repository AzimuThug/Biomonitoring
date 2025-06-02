import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
from collections import deque


class Model:

    def __init__(self,
                 width: int = 640,
                 height: int = 640,
                 brightness: float = 128.0,
                 brightness_variation: float = 0.0,
                 max_outliers: float = 128.0,
                 outlier_count: int = 0,
                 amplitude: float = 50.0,
                 bpm: int = 72,
                 fps: int = 25,
                 history_seconds: int = 10,
                 low_freq_amplitude: float = 0.0,
                 low_freq_frequency: float = 0.1,
                 high_freq_amplitude: float = 0.0
                 ) -> None:
        self._ax2 = None
        self._width = width
        self._height = height
        self._brightness = brightness
        self._noise = brightness_variation
        self._max_outliers = max_outliers
        self._outlier_count = outlier_count
        self._amplitude = amplitude
        self._bpm = bpm
        self._fps = fps
        self._history_seconds = history_seconds
        self._low_freq_amplitude = low_freq_amplitude
        self._low_freq_frequency = low_freq_frequency
        self._high_freq_amplitude = high_freq_amplitude

    @staticmethod
    def generate_2d_image(width: int = 640,
                          height: int = 640,
                          brightness: float = 128.0,
                          brightness_variation: float = 0.0,
                          max_outliers: float = 128.0,
                          outlier_count: int = 0
                          ) -> np.array:
        """
        Generate model image with noise and outliers (optional)
        :return: image
        """
        image = np.full((width, height), brightness, dtype=np.float32)
        if brightness_variation == 0.0 and outlier_count == 0.0:
            return image

        image += np.random.uniform(-brightness_variation, brightness_variation, (height, width))

        for _ in range(outlier_count):
            x, y = random.randint(0, width - 1), random.randint(0, height - 1)
            image[y, x] = random.uniform(0, max_outliers)

        return np.clip(image, 0.0, 255.0)

    @staticmethod
    def generate_1d_ppg(time: float,
                        bpm: int = 60,
                        ) -> float:
        """
        Generate a ppg curve
        :param time: Current time (sec)
        :param bpm: Beats per minute (60-100)
        :return: value at point x
        """
        period = 60.0 / bpm
        phase = (time % period) / period

        # Основной пик (систола)
        if phase < 0.2:
            return np.sin((phase / 0.2) * (np.pi / 2))
        # Дикротическая выемка
        elif phase < 0.3:
            return 0.8 - 0.4 * ((phase - 0.2) / 0.1)
        # Плавный спад (диастола)
        else:
            return 0.4 * np.exp(-10.0 * (phase - 0.3))

    def update_image(self, frame, img_plot, line_plot, time_history, ppg_history):
        """Update image and curve data depend on time every frame"""

        current_time = frame / self._fps
        ppg_value = Model.generate_1d_ppg(current_time, self._bpm)

        drift = self._low_freq_amplitude * np.sin(2 * np.pi * self._low_freq_frequency * current_time)
        noise = self._high_freq_amplitude * np.random.uniform(-1, 1)
        ppg_value += drift + noise

        base_image = Model.generate_2d_image(
            width=self._width, height=self._height, brightness=self._brightness,
            brightness_variation=self._noise, max_outliers=self._max_outliers, outlier_count=self._outlier_count
        )

        modulation_factor = 1 + (self._amplitude / self._brightness) * ppg_value
        modulated_image = base_image * modulation_factor
        modulated_image = np.clip(modulated_image, 0, 255)

        img_plot.set_array(modulated_image)

        time_history.append(current_time)
        ppg_history.append(np.mean(modulated_image))

        while len(time_history) > self._history_seconds * self._fps:
            time_history.popleft()
            ppg_history.popleft()

        line_plot.set_data(time_history, ppg_history)

        self._ax2.set_xlim(max(0, current_time - self._history_seconds), max(self._history_seconds, current_time))
        self._ax2.set_ylim(self._brightness - self._amplitude, self._brightness + self._amplitude)

        return img_plot, line_plot

    def display(self):
        """Draw image and ppg curve plots"""

        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

        # Image plot
        ax1 = fig.add_subplot(gs[0])
        img_plot = ax1.imshow(np.zeros((self._height, self._width)), cmap='gray', vmin=0, vmax=255)
        plt.colorbar(img_plot, ax=ax1, label='Яркость')
        ax1.set_title(f"2D визуализация пульса (ЧСС: {self._bpm} уд/мин)")

        # PPG plot
        self._ax2 = fig.add_subplot(gs[1])
        time_history = deque(maxlen=self._history_seconds * self._fps)
        ppg_history = deque(maxlen=self._history_seconds * self._fps)
        line_plot, = self._ax2.plot([], [], 'r-', linewidth=2)
        self._ax2.set_title("Средняя яркость изображения")
        self._ax2.set_xlabel("Время (сек)")
        self._ax2.set_ylabel("Яркость")
        self._ax2.grid(True)
        self._ax2.set_xlim(0, self._history_seconds)
        self._ax2.set_ylim(self._brightness - self._amplitude, self._brightness + self._amplitude)

        ani = animation.FuncAnimation(
            fig,
            self.update_image,
            fargs=(img_plot, line_plot, time_history, ppg_history),
            frames=self._fps * self._history_seconds * 2,
            interval=1000 / self._fps,
            blit=False,
            cache_frame_data=False
        )

        plt.tight_layout()
        plt.show()
