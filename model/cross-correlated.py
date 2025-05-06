import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from model import Model


if __name__ == "__main__":
    duration = 20.0  # Длительность в секундах
    fs = 1000  # Частота дискретизации (точек в секунду)
    bpm = 72  # Частота пульса (уд/мин)
    freq = 72  # Частота синусоиды (Гц)
    brightness = 128.0

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)


    curve = [Model.generate_1d_ppg(time=time, bpm=freq) for time in t]
    ppg_wave = np.array([np.mean(Model.generate_2d_image(640,640, brightness*(ti+1),
                                                         50.0,128.0, 100)) for ti in curve])

    correlation = signal.correlate(sine_wave, ppg_wave, mode='full', method='auto')
    corr_t = np.linspace(-duration / 2, duration / 2, len(correlation))

    # Нормализация корреляции
    correlation = correlation / np.max(correlation)

    # Построение графиков
    plt.figure(figsize=(15, 10))

    # График синусоиды
    plt.subplot(3, 1, 1)
    plt.plot(t, sine_wave, label=f'Синусоида {freq} Гц')
    plt.title('Синусоидальный сигнал')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()

    # График пульсовой волны
    plt.subplot(3, 1, 2)
    plt.plot(t, ppg_wave, 'r', label=f'Пульсовая волна ({bpm} уд/мин)')
    plt.title('Фотоплетизмограмма (ФПГ)')
    plt.xlabel('Время (с)')
    plt.ylabel('Амплитуда')
    plt.grid(True)
    plt.legend()

    # График кросс-корреляции
    plt.subplot(3, 1, 3)
    plt.plot(corr_t, correlation, 'g', label='Кросс-корреляция')
    plt.title('Кросс-корреляция между сигналами')
    plt.xlabel('Задержка (с)')
    plt.ylabel('Коэффициент корреляции')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()