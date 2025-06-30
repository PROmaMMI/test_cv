import cv2
import os
import argparse

def extract_frames(video_path: str, output_dir: str, frame_step: int = 30) -> None:

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Не удалось открыть видео {video_path}")

    frame_idx = 0
    saved = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            filename = f"{basename}_frame{frame_idx}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved += 1
        frame_idx += 1

    cap.release()
    print(f"Извлечено и сохранено {saved} кадров из {basename}")


def main():
    parser = argparse.ArgumentParser(
        description="Извлечение кадров из всех видео в папке"
    )
    parser.add_argument(
        "-i", "--input_dir", required=True,
        help="Путь к папке с исходными видео"
    )
    parser.add_argument(
        "-o", "--output_dir", required=True,
        help="Путь к папке для сохранения кадров"
    )
    parser.add_argument(
        "-s", "--step", type=int, default=30,
        help="Шаг между кадрами (по умолчанию 30)"
    )
    args = parser.parse_args()

    for fname in os.listdir(args.input_dir):
        if fname.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            video_path = os.path.join(args.input_dir, fname)
            extract_frames(video_path, args.output_dir, args.step)


if __name__ == "__main__":
    main()
