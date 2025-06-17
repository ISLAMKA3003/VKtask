import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from skimage.measure import compare_ssim as ssim
from skimage.transform import resize
from tqdm import tqdm


def extract_key_frames(video_path, interval=5):
    """
    Извлечение ключевых кадров из видео

    :param video_path: путь к видеофайлу
    :param interval: интервал извлечения кадров в секундах
    :return: список ключевых кадров и их временных меток
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    key_frames = []
    timestamps = []

    for i in tqdm(range(0, frame_count, int(fps * interval))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            key_frames.append(frame)
            timestamps.append(i / fps)

    cap.release()
    return key_frames, timestamps


def calculate_ssim(frame1, frame2):
    """
    Вычисление структурного сходства между кадрами

    :param frame1: первый кадр
    :param frame2: второй кадр
    :return: значение SSIM
    """
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    gray1 = resize(gray1, (256, 256))
    gray2 = resize(gray2, (256, 256))

    return ssim(gray1, gray2)


def find_similar_frames(key_frames, threshold=0.95):
    """
    Поиск похожих кадров

    :param key_frames: список ключевых кадров
    :param threshold: порог схожести
    :return: список пар похожих кадров
    """
    similar_pairs = []

    for i in tqdm(range(len(key_frames))):
        for j in range(i + 1, len(key_frames)):
            ssim_value = calculate_ssim(key_frames[i], key_frames[j])
            if ssim_value > threshold:
                similar_pairs.append((i, j, ssim_value))

    return similar_pairs


def merge_intervals(intervals):
    """
    Объединение перекрывающихся интервалов

    :param intervals: список интервалов
    :return: объединенные интервалы
    """
    merged = []
    for interval in sorted(intervals):
        if not merged or merged[-1][1] < interval[0]:
            merged.append(list(interval))
        else:
            merged[-1][1] = max(merged[-1][1], interval[1])
    return merged


def find_opening_credits(video_path, threshold=0.95, interval=5):
    """
    Основной алгоритм поиска заставки

    :param video_path: путь к видеофайлу
    :param threshold: порог схожести
    :param interval: интервал извлечения кадров
    :return: список интервалов заставки
    """
    key_frames, timestamps = extract_key_frames(video_path, interval)
    similar_pairs = find_similar_frames(key_frames, threshold)

    # Преобразование пар в интервалы
    intervals = [(timestamps[pair[0]], timestamps[pair[1]]) for pair in similar_pairs]

    # Объединение перекрывающихся интервалов
    merged_intervals = merge_intervals(intervals)

    return merged_intervals


def visualize_results(video_path, intervals):
    """
    Визуализация результатов

    :param video_path: путь к видеофайлу
    :param intervals: найденные интервалы
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    fig, ax = plt.subplots()
    ax.set_title('Временные интервалы заставки')
    ax.set_xlabel('Время (секунды)')
    ax.set_ylabel('Интенсивность')

    # Чтение всего видео для построения графика
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    intensity = []
    times = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            intensity.append(np.mean(gray))
            times.append(i / fps)

    # Построение графика
    ax.plot(times, intensity, label='Интенсивность кадров')

    # Отображение найденных интервалов
    for start, end in intervals:
        ax.axvspan(start, end, alpha=0.3, color='red', label='Заставка')

    # Показываем легенду только один раз
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())

    plt.show()
    cap.release()


def save_results(video_path, intervals):
    """
    Сохранение результатов в файл

    :param video_path: путь к видеофайлу
    :param intervals: найденные интервалы
    """
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    output_file = f"{base_name}_credits.txt"

    with open(output_file, 'w') as f:
        for start, end in intervals:
            f.write(f"Начало: {start:.2f} секунд, Конец: {end:.2f} секунд\n")

    print(f"Результаты сохранены в {output_file}")


def main():
    video_path = 'path/to/your/video.mp4'

    # Поиск заставки
    credits_intervals = find_opening_credits(video_path, threshold=0.95, interval=5)

    # Визуализация результатов
    visualize_results(video_path, credits_intervals)

    # Сохранение результатов
    save_results(video_path, credits_intervals)


if __name__ == "__main__":
    main()
