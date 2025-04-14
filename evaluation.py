import os
import argparse
from PIL import Image
import torchvision.transforms as transforms
import pyiqa


def calculate_metrics(directory_mapping, resize_flag):
    niqe = pyiqa.create_metric("niqe_matlab")

    print(f"\nNIQE (lower better): {niqe.lower_better} (range): {niqe.score_range}")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    for dir_path, test_data_path in directory_mapping.items():
        print(f"\nProcessing directory: {dir_path}")
        results = []

        for filename in os.listdir(dir_path):
            main_img_path = os.path.join(dir_path, filename)
            test_img_path = os.path.join(test_data_path, filename)

            with Image.open(main_img_path) as main_img, Image.open(test_img_path) as test_img:
                test_width, test_height = test_img.size
                if resize_flag:
                    main_img_resized = main_img.resize((test_width, test_height), Image.LANCZOS)
                else:
                    main_img_resized = main_img

                img_tensor = transform(main_img_resized).unsqueeze(0)

                niqe_score = niqe(img_tensor)

                results.append((filename, niqe_score.item(), ))

        average_niqe_score = sum(score[1] for score in results) / len(results)

        print(f"Avg NIQE: {round(average_niqe_score, 3)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resize", action="store_true", help="匹配")
    args = parser.parse_args()

    directory_mapping = {
        "./output/DICM": "./input/DICM",
    }
    calculate_metrics(directory_mapping, args.resize)
