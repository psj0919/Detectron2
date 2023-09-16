!python -m pip install pyyaml==5.1
import sys, os, distutils.core
# Note: This is a faster way to install detectron2 in Colab, but it does not include all functionalities (e.g. compiled operators).
# See https://detectron2.readthedocs.io/tutorials/install.html for full installation instructions
!git clone 'https://github.com/facebookresearch/detectron2'
# 'https://github.com/facebookresearch/detectron2' github에 있는 것들을 불러옴
dist = distutils.core.run_setup("./detectron2/setup.py")
!python -m pip install {' '.join([f"'{x}'" for x in dist.install_requires])}
sys.path.insert(0, os.path.abspath('./detectron2'))

# Properly install detectron2. (Please do not install twice in both ways)
# !python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

import torch, detectron2
!nvcc --version
TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
CUDA_VERSION = torch.__version__.split("+")[-1]
print("torch: ", TORCH_VERSION, "; cuda: ", CUDA_VERSION) #torch와 cuda의 버전을 확인함
print("detectron2:", detectron2.__version__) # detectron2의 버전을 확인함.

# Some basic setup:
# Setup detectron2 logger 
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random
from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

#-----------------------------------------------------------------
!wget http://images.cocodataset.org/val2017/000000439715.jpg -q -O input.jpg # http://images.cocodataset.org/val2017/000000439715.jpg 이곳에서 이미지를 다운받음
im = cv2.imread("./input.jpg") #cv2를 이용하여 이미지를 읽음
cv2_imshow(im) # 이미지 출력
# -----------------------------------------------------------------

cfg = get_cfg() #get_cfg 클래스 호출
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
predictor = DefaultPredictor(cfg)
outputs = predictor(im)

# look at the outputs. See https://detectron2.readthedocs.io/tutorials/models.html#model-output-format for specification
print(outputs["instances"].pred_classes) # output이 예측된 class출력
print(outputs["instances"].pred_boxes) # output이 예측한 Boxes 출력

# We can use `Visualizer` to draw the predictions on the image.
# 입력의 사진에서 예측된것을 바운딩 박스와 그 score로 하여 그림을 visulize 함
v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2) 
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
cv2_imshow(out.get_image()[:, :, ::-1])




# if your dataset is in COCO format, this cell can be replaced by the following three lines:
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("my_dataset_train", {}, "json_annotation_train.json", "path/to/image/dir")
# register_coco_instances("my_dataset_val", {}, "json_annotation_val.json", "path/to/image/dir")

from detectron2.structures import BoxMode

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json") #img_dir이라는 변수 + via_region_data.json을 합침
    with open(json_file) as f:# file을 오픈
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {} #딕셔너리 형태
        
        filename = os.path.join(img_dir, v["filename"]) #img_dir이라는 변수 + v["filename"]을 합침
        height, width = cv2.imread(filename).shape[:2] #위에 코드에서 합친 path를 열어 height와 width로 초기화
        
        record["file_name"] = filename # 딕셔너리형태로 file_name이라는 key값을 filename value로 저장
        record["image_id"] = idx # 딕셔너리 형태로 image_id라는 key값을 idx라는 value값으로 저장
        record["height"] = height # 딕셔너리 형태로 height라는 key값을 height라는 height값으로 저장
        record["width"] = width # 딕셔너리 형태로 width라는 key값을 width라는 value값으로 저장
      
        annos = v["regions"]
        objs = []
        # box를 계산하기 위한 반복문
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]
      
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

for d in ["train", "val"]:
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")


# ----------위의 Visualizer와 get_ballon_dicts함수를 이용하는 코드----------
dataset_dicts = get_balloon_dicts("balloon/train") # get_ballon_dicts함수에 img_dir로 "ballon/train"을 매게변수로 호출
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"]) #d["file_name"]에 있는 이미지를 읽어옴
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5) #Visualizer라는 함수를 호출
    out = visualizer.draw_dataset_dict(d) 
    cv2_imshow(out.get_image()[:, :, ::-1]) # bbox와 이미지를 같이 출력
# ---------------------------------------------------------------------------



# ---------------------------------------------------training---------------------------------------------------------------------------------------
from detectron2.engine import DefaultTrainer

cfg = get_cfg() #get_cfg를 호출
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("balloon_train",) #train으로 "ballon_train"을 매게변수로 함수호출
cfg.DATASETS.TEST = () #test
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR, Learning rate
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []        # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train() # 학습 실행
# ------------------------------------------------------------------------------------------------------------------------------------------------


# ---------------------------------------------------inference & evaluation---------------------------------------------------------------------------------------
# Inference should use the config with parameters that are used in training
# cfg now already contains everything we've set previously. We changed it a little bit for inference:
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode
dataset_dicts = get_balloon_dicts("balloon/val")
for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                   metadata=balloon_metadata, 
                   scale=0.5, 
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels. This option is only available for segmentation models
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2_imshow(out.get_image()[:, :, ::-1])


from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator("balloon_val", output_dir="./output")
val_loader = build_detection_test_loader(cfg, "balloon_val")
print(inference_on_dataset(predictor.model, val_loader, evaluator))
# another equivalent way to evaluate the model is to use `trainer.test`
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------

