import numpy as np
from scipy import signal
import matplotlib.pyplot as plt


def extract_ppg(noisy_signal, fs=1000, plot=True):
    """
    Выделение ФПГ-сигнала из зашумленных данных

    param:
        noisy_signal - зашумленный ФПГ-сигнал (numpy array)
        fs - частота дискретизации (Гц)
        plot - визуализировать процесс (True/False)

    return:
        clean_ppg - очищенный ФПГ-сигнал
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
    corr = corr[len(corr) // 2:]  # Берем только положительные задержки

    # 3. Поиск пиков (исключая первый пик при нулевой задержке)
    peaks, _ = signal.find_peaks(corr, distance=int(fs / 2.5))  # Минимальное расстояние для 2.5 Гц

    if len(peaks) < 2:
        raise ValueError("Не удалось обнаружить периодичность в сигнале")

    # 4. Оценка ЧСС по среднему расстоянию между пиками
    mean_peak_distance = np.mean(np.diff(peaks)) / fs  # В секундах
    heart_rate = 60 / mean_peak_distance  # Преобразуем в уд/мин

    # 5. Синхронное накопление для улучшения SNR
    period_samples = int(mean_peak_distance * fs)
    num_periods = len(filtered) // period_samples

    if num_periods < 1:
        return filtered, heart_rate

    segmented = filtered[:num_periods * period_samples].reshape(num_periods, period_samples)
    clean_ppg = np.mean(segmented, axis=0)

    # 6. Визуализация при необходимости
    if plot:
        t = np.arange(len(noisy_signal)) / fs
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
        plt.plot(peaks, 'r', label='Проход')
        plt.title('Выделенный ФПГ (один период)')
        plt.xlabel('Время (с)')
        plt.ylabel('Амплитуда')
        plt.legend()

        plt.tight_layout()
        plt.show()

    return clean_ppg, heart_rate


# Пример использования
if __name__ == "__main__":
    # Генерация тестового сигнала ФПГ с ЧСС 72 уд/мин (1.2 Гц)
    fs = 250  # Типичная частота дискретизации для ФПГ
    t = np.linspace(0, 10, 10 * fs)  # 10 секунд данных
    clean_ppg = 0.5 * np.sin(2 * np.pi * 1.2 * t) + 0.3 * np.sin(2 * np.pi * 2.4 * t)  # Основная + гармоника

    # Добавляем шум и артефакты
    noise = 0.4 * np.random.normal(size=len(t))
    motion_artifact = 0.8 * np.sin(2 * np.pi * 0.2 * t)  # Низкочастотный артефакт
    powerline_noise = 0.3 * np.sin(2 * np.pi * 50 * t)  # Наводка 50 Гц

    noisy_signal = clean_ppg + noise + motion_artifact + powerline_noise

    # Обработка
    extracted_ppg, hr = extract_ppg(noisy_signal, fs=fs, plot=True)
    print(f"Оценка ЧСС: {hr:.1f} уд/мин")