import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from model import Model


if __name__ == "__main__":
    duration = 20.0  # Длительность в секундах
    fs = 10  # Частота дискретизации (точек в секунду)
    bpm = 72  # Частота пульса (уд/мин)
    freq = 1  # Частота синусоиды (Гц)
    brightness = 128.0

    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    sine_wave = np.sin(2 * np.pi * freq * t)

    curve = [Model.generate_1d_ppg(time=time, bpm=bpm) for time in t]

    low_freq = 0.05  # Частота паразитной составляющей (0.05 Гц)
    low_freq_amplitude = 0.3  # Амплитуда низкочастотных колебаний (30% от сигнала)

    parasitic_component = low_freq_amplitude * np.sin(2 * np.pi * low_freq * t)

    curve_with_parasitic = curve * (1 + parasitic_component)

    ppg_wave = np.array([np.mean(Model.generate_2d_image(640,640, brightness*(ti+1),
                                                         150.0,128.0, 100)) for ti in curve_with_parasitic])

    ppg_wave_new = ppg_wave * (1 + parasitic_component)

    correlation = signal.correlate(sine_wave, ppg_wave, mode='full', method='auto')
    corr_t = np.linspace(-duration / 2, duration / 2, len(correlation))

    # Нормализация корреляции
    correlation = correlation / np.max(correlation)

    n = len(correlation)
    freqs = np.fft.fftfreq(n, d=1 / fs)[:n // 2]  # Положительные частоты
    spectrum = np.abs(np.fft.fft(correlation)[:n // 2])  # Амплитудный спектр

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
    plt.semilogy(freqs, spectrum)  # Логарифмическая шкала по Y
    plt.title('Спектр кросс-корреляции')
    plt.xlabel('Частота (Гц)')
    plt.ylabel('Амплитуда спектра (логарифмическая шкала)')
    plt.grid()
    plt.xlim(0, 10)  # Показываем только интересующий диапазон частот

    plt.tight_layout()
    plt.show()