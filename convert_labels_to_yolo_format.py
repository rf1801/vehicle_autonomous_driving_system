import os


def convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir, img_width=1242, img_height=375):
    if not os.path.exists(yolo_label_dir):
        os.makedirs(yolo_label_dir)

    for kitti_label_file in os.listdir(kitti_label_dir):
        with open(os.path.join(kitti_label_dir, kitti_label_file), 'r') as f:
            lines = f.readlines()

        yolo_lines = []
        for line in lines:
            data = line.split()
            class_name = data[0]
            class_id = 0 if class_name == "Pedestrian" else 1  # Example class mapping

            # Extract the bounding box coordinates
            xmin, ymin, xmax, ymax = float(data[4]), float(data[5]), float(data[6]), float(data[7])

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2 / img_width
            y_center = (ymin + ymax) / 2 / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            yolo_lines.append(f"{class_id} {x_center} {y_center} {width} {height}\n")

        # Save the YOLO formatted labels
        with open(os.path.join(yolo_label_dir, kitti_label_file), 'w') as f:
            f.writelines(yolo_lines)


# Example usage
kitti_label_dir = 'C:/Datasets/kitti/training/label_2'
yolo_label_dir = 'C:/Datasets/kitti/training/label_yolo_format'
convert_kitti_to_yolo(kitti_label_dir, yolo_label_dir)
