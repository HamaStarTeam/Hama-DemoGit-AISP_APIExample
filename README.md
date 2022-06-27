Copy from https://github.com/ultralytics/yolov5 and edit for myself
## <div align="center">Quick Start Examples</div>
<details close>
<summary>Loacl Predict</summary>

Use detect.pt to predict ex:python D:/GithubYolo5/yolov5-master/detect.py --weights D:/GithubYolo5/yolov5-master/runs/train/exp4/weights/best.pt --conf-thres 0.25

see below find more parameter 
```bash
parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
```
</details>


<details close>
<summary>Local Train</summary>
1.Change or add xxx.yaml to ./data. Yaml example can see coco.yaml or see below

```bash
   path: D:/Data/PanJuDataYolo  # dataset root dir
   train: train/images  # train images (relative to 'path') 128 images
   val: valid/images  # val images (relative to 'path') 128 images
   test: test/images 
   nc: 1  # number of classes
   names: ['offset']  # class names
```
2.# Train YOLOv5s on xxx.yaml for 3 epochs

```bash
    python train.py --img 640 --batch 16 --epochs 3 --data xxx.yaml --weights yolov5s.pt
```
more detail see [PyTorch Hub](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)
</details>

<details close>
    <summary>Local Predict by code</summary>
    PanJuPoc.py is simple code for predict 
</details>

## <div align="center">Note</div>
<details close>
    <summary>Environment</summary>
    Use pip install -r requirements.txt is not enough for cuda. We need uninstall torch and install torch with cuda after install requirements.txt.But some version is not math. I already known below:
    
Useful:   
    
```bash
    1.pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio===0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
```
    
Useless:
    
```bash
    1.pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```
</details>
<details close>
    <summary>Notice Memory</summary>
    If we use cuda, it can cause out of memory for GPU device. If use CPU only, it can cause out of memory for DDP mode. So we can set batch size and worker size when we train. EX: 
    
    python train.py --batch 16 --epochs 200 --data PanJu.yaml --weights yolov5m.pt --device cpu --workers 0
</details>