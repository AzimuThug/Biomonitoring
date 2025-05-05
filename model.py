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
                 outlier_count: int = 0.0,
                 amplitude: float = 50.0,
                 bpm: int = 72,
                 ts: int = 50
                 ) -> None:
        self._width = width
        self._height = height
        self._brightness = brightness
        self._noise = brightness_variation
        self._max_outliers = max_outliers
        self._outlier_count = outlier_count
        self._amplitude = amplitude
        self._bpm = bpm
        self._ts = ts

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
        :param time: Текущее время (сек)
        :param bpm: Ударов в минуту (норма: 60-100)
        :return: value at point x
        """
        period = 60.0 / bpm  # Длительность одного удара (сек)
        phase = (time % period) / period  # Фаза в пределах [0, 1]

        # Основной пик (систола)
        if phase < 0.2:
            return np.sin((phase / 0.2) * (np.pi / 2))
        # Дикротическая выемка
        elif phase < 0.3:
            return 0.8 - 0.4 * ((phase - 0.2) / 0.1)
        # Плавный спад (диастола)
        else:
            return 0.4 * np.exp(-10.0 * (phase - 0.3))

    @staticmethod
    def update_image(frame, img_plot, line_plot, time_history, ppg_history,
                     width, height, base_brightness, brightness_variation,
                     max_outliers, outlier_count, amplitude, bpm, fps, history_seconds):
        """Обновление данных для анимации с перегенерацией изображения"""
        current_time = frame / fps
        ppg_value = Model.generate_1d_ppg(current_time, bpm)

        # Генерация нового изображения в каждом кадре
        base_image = Model.generate_2d_image(
            width, height, base_brightness,
            brightness_variation, max_outliers, outlier_count
        )

        # Модулируем яркость изображения
        modulated_image = base_brightness + amplitude * ppg_value * (base_image / 255.0)
        modulated_image = np.clip(modulated_image, 0.0, 255.0)

        # Обновляем изображение
        img_plot.set_array(modulated_image)

        # Сохраняем историю
        time_history.append(current_time)
        ppg_history.append(np.mean(modulated_image))  # Средняя яркость

        # Ограничиваем историю
        while len(time_history) > history_seconds * fps:
            time_history.popleft()
            ppg_history.popleft()

        # Обновляем график
        line_plot.set_data(time_history, ppg_history)

        # Автомасштабирование
        ax2.set_xlim(max(0, current_time - history_seconds), max(history_seconds, current_time))
        ax2.set_ylim(base_brightness - amplitude, base_brightness + amplitude)

        return img_plot, line_plot

    @staticmethod
    def display(width=640, height=640,
                base_brightness=128.0, brightness_variation=10.0,
                max_outliers=255.0, outlier_count=50,
                bpm=72, amplitude=50.0,
                fps=30, history_seconds=15):
        """Отображение анимации с динамически генерируемым изображением"""
        global ax2

        # Настройка фигуры
        fig = plt.figure(figsize=(12, 6))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1])

        # График изображения (первоначально пустой)
        ax1 = fig.add_subplot(gs[0])
        img_plot = ax1.imshow(np.zeros((height, width)), cmap='gray', vmin=0, vmax=255)
        plt.colorbar(img_plot, ax=ax1, label='Яркость')
        ax1.set_title(f"Динамическая 2D визуализация пульса (ЧСС: {bpm} уд/мин)")

        # График ФПГ
        ax2 = fig.add_subplot(gs[1])
        time_history = deque(maxlen=history_seconds * fps)
        ppg_history = deque(maxlen=history_seconds * fps)
        line_plot, = ax2.plot([], [], 'r-', linewidth=2)
        ax2.set_title("Средняя яркость изображения")
        ax2.set_xlabel("Время (сек)")
        ax2.set_ylabel("Яркость")
        ax2.grid(True)
        ax2.set_xlim(0, history_seconds)
        ax2.set_ylim(base_brightness - amplitude, base_brightness + amplitude)

        # Создание анимации
        ani = animation.FuncAnimation(
            fig,
            Model.update_image,
            fargs=(img_plot, line_plot, time_history, ppg_history,
                   width, height, base_brightness, brightness_variation,
                   max_outliers, outlier_count, amplitude, bpm, fps, history_seconds),
            frames=fps * history_seconds * 2,
            interval=1000 / fps,
            blit=False,
            cache_frame_data=False  # Важно для динамической генерации!
        )

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":

    Model.display(
        width=640,
        height=640,
        base_brightness=128.0,
        brightness_variation=100.0,
        max_outliers=255.0,
        outlier_count=300,
        bpm=75,
        amplitude=60.0,
        fps=25,
        history_seconds=10
    )