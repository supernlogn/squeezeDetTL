#!/usr/bin/python
BASE_DIR=/media/terabyte/projects/datasets/COCO
ANN_DIR=$BASE_DIR/annotations
python ../../src/datasets/creation/create_coco_tf_record.py --data_dir=$BASE_DIR --set=train --output_filepath=$BASE_DIR/coco_train.record --shuffle_imgs=False
python ../../src/datasets/creation/create_coco_tf_record.py --data_dir=$BASE_DIR --set=val --output_filepath=$BASE_DIR/coco_val.record --shuffle_imgs=False
# --train_image_dir=$BASE_DIR/train --val_image_dir=$BASE_DIR/val --test_image_dir=$BASE_DIR/test2017 --train_annotations_file=$ANN_DIR/instances_train2017.json --val_annotations_file=$ANN_DIR/instances_val2017.json --testdev_annotations_file=$ANN_DIR/ --output-path=$BASE_DIR &
