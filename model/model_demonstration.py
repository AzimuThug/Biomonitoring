import argparse
from model import Model


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--width',
        type=int,
        default=640,
        help='Ширина изображения в пикселях'
    )

    parser.add_argument(
        '--height',
        type=int,
        default=640,
        help='Высота изображения в пикселях'
    )

    parser.add_argument(
        '--brightness',
        type=float,
        default=128.0,
        help='Базовая яркость изображения (0-255)'
    )

    parser.add_argument(
        '--noise',
        type=float,
        default=100.0,
        help='Уровень шума (вариация яркости)'
    )

    args = parser.parse_args()

    ppg_model = Model(
        width=640,
        height=640,
        brightness=128.0,
        brightness_variation=100.0,
        max_outliers=255.0,
        outlier_count=300,
        bpm=75,
        amplitude=60.0,
        fps=25,
        history_seconds=10
    )

    ppg_model.display()