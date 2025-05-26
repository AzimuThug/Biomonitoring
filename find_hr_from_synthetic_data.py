import time as pytime
import argparse
import numpy as np
from processing.extract_ppg import extract_ppg_crosscorr, extract_ppg_autocorr, extract_ppg_fourier
from model.model import Model


if __name__ == "__main__":
    """
    Пример выделения зашумленной кривой фпг методом авто или кросс-корреляции
    args:
    data - Режим генерации данных (1d - кривая, 2d - изображение)
    mode - Режим обработки данных (auto - через авто-корреляцию, cross - через кросс-корреляцию
    """

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, default='1d', help='Режим генерации данных')
    parser.add_argument('--mode', type=str, default='fourier', help='Режим обработки данных')
    args = parser.parse_args()

    fs = 16  # Частота дискретизации для ФПГ (Sampling rate)
    time = 10  # Время измерения (сек)
    freq = 2.4  # Частота пульса (Гц)
    t = np.linspace(0, time, time * fs)
    curve = [Model.generate_1d_ppg(time=time, bpm=int(freq * 60)) for time in t]
    noise = 0.4 * np.random.normal(size=len(t))
    powerline_noise = 0.3 * np.sin(2 * np.pi * 50 * t)  # Высокочастотная наводка
    motion_artifact = 0.8 * np.sin(2 * np.pi * 0.2 * t)  # Низкочастотный артефакт
    curve_with_parasitic = curve + noise + motion_artifact + powerline_noise

    if args.data.lower() == '1d':
        ppg_wave = curve_with_parasitic

    elif args.data.lower() == '2d':
        brightness = 128.0
        ppg_wave = np.array([np.mean(Model.generate_2d_image(width=640, height=640, brightness=brightness * ti,
                                                             brightness_variation=100.0, max_outliers=128.0,
                                                             outlier_count=100)) for ti in curve_with_parasitic])

    else:
        raise ValueError(f"Incorrect value of data synthesis: it should be '1d' or '2d' not {args.data}")
    start_time = pytime.time()
    if args.mode.lower() == 'cross':
        hr = extract_ppg_crosscorr(noisy_signal=ppg_wave, fs=fs, plot=True)
    elif args.mode.lower() == 'auto':
        hr = extract_ppg_autocorr(noisy_signal=ppg_wave, fs=fs, plot=True)
    elif args.mode.lower() == 'fourier':
        hr = extract_ppg_fourier(noisy_signal=ppg_wave, fs=fs, plot=True)
    else:
        raise ValueError(f"Incorrect value of processing mode: it should be 'auto' or 'cross'")
    end_time = pytime.time()
    execution_time = end_time - start_time

    print(f"Оценка ЧСС: {hr:.1f} уд/мин")
    print(f"{execution_time:.4f}")

