import pascal_voc
import kitti
import ms_coco
import creation

# def create_dataset(as_dataset):
#   dataset_creators = {
#   "KITTI"      : creation.create_kitti_tf_record.convert_kitti_to_tfrecords,
#   "PASCAL_VOC" : creation.create_pascal_tf_record.,
#   "COCO"       : creation.create_coco_tf_record.create_coco_detection_dataset}

def get_dataset(name):
  all_datasets = {
  "KITTI"      : kitti.kitti,
  "PASCAL_VOC" : pascal_voc.pascal_voc,
  "COCO"       : ms_coco.coco}
  if(name in all_datasets):
    return all_datasets[name]
  else:
    return pascal_voc.pascal_voc

def get_evaluation_func(name):
  all_datasets = {
  "KITTI"      : kitti.evaluate_detections,
  "PASCAL_VOC" : pascal_voc.evaluate_detections,
  "COCO"       : ms_coco.evaluate_detections}
  if(name in all_datasets):
    return all_datasets[name]
  else:
    return pascal_voc.evaluate_detections