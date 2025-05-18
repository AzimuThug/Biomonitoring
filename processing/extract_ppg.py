import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def extract_ppg_autocorr(noisy_signal: np.array, fs: int = 250, plot: bool = True) -> float:
    """
    Выделение ФПГ-сигнала из зашумленных данных методом авто или кросс-корреляции
    param:
        noisy_signal - зашумленный ФПГ-сигнал (numpy array)
        fs - частота дискретизации (Гц)
        mode - метод выделения сигнала (кросс или авто-корреляция)
        plot - визуализировать процесс (True/False)

    return:
        clean_ppg - очищенный ФПГ-сигнал (Optional)
        heart_rate - оценка ЧСС (уд/мин)
    """
    # 1. Предварительная фильтрация (полосовой фильтр 0.8-2.5 Гц)
    nyquist = 0.5 * fs
    low = 0.8 / nyquist
    high = 2.5 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, noisy_signal)

    # 2. Автокорреляция для оценки периода
    corr = signal.correlate(filtered, filtered, mode='full', method='auto')
    corr = corr[len(corr) // 2:]  # Только положительные задержки

    # 3. Поиск пиков (исключая первый пик при нулевой задержке)
    peaks, _ = signal.find_peaks(corr, distance=int(fs / 2.5))  # Минимальное расстояние для 2.5 Гц

    if len(peaks) < 2:
        raise ValueError("Не удалось обнаружить периодичность в сигнале")

    # 4. Оценка ЧСС по среднему расстоянию между пиками
    mean_peak_distance = np.mean(np.diff(peaks)) / fs  # В секундах
    heart_rate = 60 / mean_peak_distance  # Преобразуем в уд/мин

    if plot:
        t = np.arange(len(noisy_signal)) / fs
        lags = np.arange(len(corr)) / fs  # Временные задержки в секундах
        plt.figure(figsize=(15, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t, noisy_signal, label='Исходный сигнал')
        plt.title(f'Зашумленный ФПГ (оценка ЧСС: {heart_rate:.1f} уд/мин)')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t[:len(filtered)], filtered, 'g', label='После полосовой фильтрации')
        plt.title('После предварительной фильтрации (0.8-2.5 Гц)')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(lags, corr, 'r', label='Автокорреляция')
        plt.plot(peaks/fs, corr[peaks], 'bo', label='Обнаруженные пики')
        plt.title('Автокорреляционная функция')
        plt.xlabel('Задержка (с)')
        plt.ylabel('Нормированная корреляция')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    return heart_rate


def extract_ppg_crosscorr(noisy_signal: np.array, fs: int = 250, plot: bool = True) -> float:
    """
    Выделение ФПГ-сигнала из зашумленных данных методом кросс-корреляции
    param:
        noisy_signal - зашумленный ФПГ-сигнал (numpy array)
        fs - частота дискретизации (Гц)
        mode - метод выделения сигнала (кросс или авто-корреляция)
        plot - визуализировать процесс (True/False)

    return:
        clean_ppg - очищенный ФПГ-сигнал (Optional)
        heart_rate - оценка ЧСС (уд/мин)
    """
    # 1. Предварительная фильтрация (полосовой фильтр 0.8-2.5 Гц)
    nyquist = 0.5 * fs
    low = 0.8 / nyquist
    high = 2.5 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, noisy_signal)

    # 2. Подготовка тестовых частот
    test_freqs = np.linspace(0.8, 2.5, 20)  # 20 частот в диапазоне 0.8-2.5 Гц
    t = np.arange(len(filtered)) / fs

    # 3. Цикл кросс-корреляции
    max_corr = -np.inf
    best_freq = 0.8
    corr_results = []

    for freq in test_freqs:
        # Генерация тестового синуса
        test_wave = np.sin(2 * np.pi * freq * t)

        # Кросс-корреляция
        corr = signal.correlate(filtered, test_wave, mode='same')
        corr_peak = np.max(corr)
        corr_results.append(corr_peak)

        # Поиск максимальной корреляции
        if corr_peak > max_corr:
            max_corr = corr_peak
            best_freq = freq

    # 4. Оценка ЧСС
    heart_rate = best_freq * 60  # Преобразуем Гц в уд/мин

    if plot:
        t = np.arange(len(noisy_signal)) / fs
        plt.figure(figsize=(15, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t, noisy_signal, label='Исходный сигнал')
        plt.title(f'Зашумленный ФПГ (оценка ЧСС: {heart_rate:.1f} уд/мин)')
        plt.xlabel('Время, с')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t[:len(filtered)], filtered, 'g', label='После полосовой фильтрации')
        plt.title('После предварительной фильтрации (0.8-2.5 Гц)')
        plt.xlabel('Время, с')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(test_freqs, corr_results, 'r')
        plt.title('Кросс-корреляция')
        plt.xlabel('Частота, Гц')
        plt.ylabel('Амплитуда')

        plt.tight_layout()
        plt.show()

    return heart_rate

def extract_ppg_fourier(noisy_signal: np.array, fs: int = 256, plot: bool = True) -> float:
    """
    Выделение ФПГ-сигнала из зашумленных данных через спектральный анализ
    param:
        noisy_signal - зашумленный ФПГ-сигнал (numpy array)
        fs - частота дискретизации (Гц)
        plot - визуализировать процесс (True/False)

    return:
        heart_rate - оценка ЧСС (уд/мин)
    """

    # 1. Предварительная фильтрация (полосовой фильтр 0.8-2.5 Гц)
    nyquist = 0.5 * fs
    low = 0.8 / nyquist
    high = 2.5 / nyquist
    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, noisy_signal)

    n = filtered.size
    spectrum = np.fft.fft(filtered)
    amplitudes = np.abs(spectrum) / n
    frequencies = np.fft.fftfreq(n, d=1/fs)[:n//2]
    amplitudes = amplitudes[:n//2]

    heart_rate = frequencies[np.argmax(amplitudes)] * 60

    if plot:
        t = np.arange(len(noisy_signal)) / fs
        plt.figure(figsize=(15, 8))

        plt.subplot(3, 1, 1)
        plt.plot(t, noisy_signal, label='Исходный сигнал')
        plt.title(f'Зашумленный ФПГ (оценка ЧСС: {heart_rate:.1f} уд/мин)')
        plt.xlabel('Время, с')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(t[:len(filtered)], filtered, 'g', label='После полосовой фильтрации')
        plt.title('После предварительной фильтрации (0.8-2.5 Гц)')
        plt.xlabel('Время, с')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(frequencies, amplitudes, 'r')
        plt.title('Спектр отфильтрованного сигнала')
        plt.xlabel('Частота, Гц')
        plt.ylabel('Амплитуда')
        plt.xlim(0.8, 2.5)

        plt.tight_layout()
        plt.show()

    return heart_rate

