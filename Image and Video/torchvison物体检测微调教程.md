# torch视觉物体监测模型微调教程
`为了更好的使用此教程，我们推荐使用Colab Version,这可以让你体验到下面所提及的内容`
在这个教程中，我们会修改一个提前训练好的Mask R-CNN模型来进行一个数据集的训练，更多关于这个数据集请点击[这里](https://www.cis.upenn.edu/~jshi/ped_html/)。此数据集包含了170个图片，这些图片又包含了345个行人实例，我们将会使用这个模型来展示如何使用torchversion的新特性，完成使用自定义数据集训练一个分割模型。  

## 定义数据集
你可以在下列被用于训练对象检测，实例分割和人员关键点的参考脚本中很容易的添加自定义数据集。这些数据集应该继承自标准的torch.utils.data.Dataset类，并且要实现__len__和__getitem__  
	
数据集__getitem__的返回值应该要注意以下几点  
- image: 一个具有尺寸的PIL对象 (H, W)
- target: 包含了以下属性的字典
>- boxes (FloatTensor[N, 4]): N个边界框的坐标，要以[x0, y0, x1, y1] 为格式，x属于[0,W],y属于[0,H]
>- labels (Int64Tensor[N]):  边界框的标签，一般用0代表背景
>- image_id (Int64Tensor[1]):  一个图片标识符，在此图片所在的数据集中，他应该是唯一的，并且在评估中要使用
>- area (Tensor[N]): 边界框的区域。在使用COCO指标进行评估时使用，将小中大不同尺寸边框的度量分数区分开
>- iscrowd (UInt8Tensor[N]): 具有iscrowd=True标记的实例将会在评估阶段忽略
>- (可选) masks (UInt8Tensor[N, H, W]): 每一个对象的分割蒙版
>- (可选) keypoints (FloatTensor[N, K, 3]): 在N个对象中，每一个车对象都包含了K个关键点，以 [x, y, visibility]为格式定义一个对象， visibility=0表示此关键点不可见注意在数据声明中，翻转关键点的概念取决于数据表示，并且你可能需要修改 references/detection/transforms.py 来适应你的新关键点表示。
如果你的模型拥有以上方法，那么你的模型将会在训练和评估中正常工作，并且会使用到评估脚本pycocotools ，你可以使用pip install pycocotools 来安装这个脚本。
`在windows下请在命令行下从gautamchitnis获取`
运行以下命令
`pycocotools pip install git+https://github.com/gautamchitnis/cocoapi.git@cocodataset-master#subdirectory=PythonAPI`
使用labels时注意：模型会将0类认为是背景。如果你的数据集不包含背景类，那么在你的labels中就不应该包含0。举个例子，将设你只有2种类型，分别是*猫*和*狗*。你可以用1（而不是0）代表*猫*，用2代表*狗*，所以在实例中，如果一个图片同时拥有2个类型，那么你的labels张量应该是[1,2]  
此外，如果要在训练过程中使用宽高比分组（以便每个分组仅包含具有相似长宽比的图像），则建议您还实现get_height_and_width方法，该方法返回图像的高度和宽度。 如果未提供此方法，我们将通过__getitem__查询数据集的所有元素，该方法会将图像加载到内存中，并且比提供自定义方法慢。  
# 为Pennfudan编写一个自定义数据集
让我们为Pennfudan编写一个自定义数据集，[在下载并解压了压缩文件后](https://www.cis.upenn.edu/~jshi/ped_html/PennFudanPed.zip)我们就会得到如下的目录结构。
PennFudanPed/

>PedMasks/
>>FudanPed00001_mask.png
>>FudanPed00002_mask.png
>>FudanPed00003_mask.png
>>FudanPed00004_mask.png
>>...

>PNGImages/
>>FudanPed00001.png
>>FudanPed00002.png
>>FudanPed00003.png
>>FudanPed00004.png


这里是一副图像和蒙版
![图像](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image01.png)
![蒙版](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image02.png)
可以看到每一个图片都有对应的蒙版，同时每一个不同的颜色代表了一个不同的实体。让我们为这个数据集编写一个torch.utils.data.Dataset类  
```python
import os
import numpy as np
import torch
from PIL import Image


class PennFudanDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        #加载所有图片并确保他们对其
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __getitem__(self, idx):
        # 加载图片和蒙版
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        # 注意我们没有吧蒙版转换为rgb，因为每一个颜色代表一个实体，0代表背景
        mask = Image.open(mask_path)
        # 将PIL图像转换为numpy数组
        mask = np.array(mask)
        # 每一个实例被编码为不同的颜色
        obj_ids = np.unique(mask)
        # 第一个id是背景，故将其移除
        obj_ids = obj_ids[1:]

        # 将颜色编码的蒙版分成一系列二进制蒙版
        masks = mask == obj_ids[:, None, None]

        # 为每一个蒙版得到边界框
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        # 将所有对象转换为torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 这里只有一个类
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        #确保所有实例都不拥挤
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

```
这就是数据集所有要做的，现在让我们定义一个可以在此数据集上预测的模型吧
# 定义你的模型
在这篇教程中，我们将会使用 Mask R-CNN 模型，这个模型基于 Faster R-CNN ， Faster R-CNN是一个既可以为图片中的可能的对象预测边界框和类得分的模型。  
![图像](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image03.png)
Mask R-CNN在Faster R-CNN中增加了一个分支。这让它也可以为每一个实列预测分割蒙版。
![图像](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image04.png)
在两种常见情况下，可能要修改Torchvision modelzoo中的可用模型之一。 首先是当我们想从已经训练的模型开始，并且微调最后一层。 其次是当我们要替换主干模型（例如为了更快的预测）。
接下来让我们看看如何实现以上两点。
## 微调已有模型
让我们试想你想从COCO上的一个已有模型开始，并且想要为了你的模型进行微调，下面的方法也许可行：
```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# load a model pre-trained pre-trained on COCO
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

# replace the classifier with a new one, that has
# num_classes which is user-defined
num_classes = 2  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

```
## 修改模型以添加一个不同的主干
```python
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

# load a pre-trained model for classification and return
# only the features
backbone = torchvision.models.mobilenet_v2(pretrained=True).features
# FasterRCNN needs to know the number of
# output channels in a backbone. For mobilenet_v2, it's 1280
# so we need to add it here
backbone.out_channels = 1280

# let's make the RPN generate 5 x 3 anchors per spatial
# location, with 5 different sizes and 3 different aspect
# ratios. We have a Tuple[Tuple[int]] because each feature
# map could potentially have different sizes and
# aspect ratios
anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                   aspect_ratios=((0.5, 1.0, 2.0),))

# let's define what are the feature maps that we will
# use to perform the region of interest cropping, as well as
# the size of the crop after rescaling.
# if your backbone returns a Tensor, featmap_names is expected to
# be [0]. More generally, the backbone should return an
# OrderedDict[Tensor], and in featmap_names you can choose which
# feature maps to use.
roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                output_size=7,
                                                sampling_ratio=2)

# put the pieces together inside a FasterRCNN model
model = FasterRCNN(backbone,
                   num_classes=2,
                   rpn_anchor_generator=anchor_generator,
                   box_roi_pool=roi_pooler)
```
## PennFudan 数据集的一个划分模型实例
在此案例中，我们想要微调一个已有的模型，鉴于我们的数据集非常小，所以我们会采用上文中的第一种方法。  
此处我们仍任想要计算实例的划分蒙版，所以我们将会使用 Mask R-CNN模型
```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,hidden_layer,num_classes)

    return model
```
这样一来，model就已经准备好在你的数据集上进行训练以及评估。  
## 将所有的放在一起
在 references/detection/中, 我们已经有了一系列辅助方法来简化训练和评估检测模型。在此处，我们将会使用 references/detection/engine.py, references/detection/utils.py ， references/detection/transforms.py.你只需将所有内容拷贝到 references/detection 目录下并且在此处使用他们。  
现在让我们编写一些用于数据扩展和转换的辅助函数。  
```python
import transforms as T

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
```
## 测试forward()方法（可选）
在对数据集进行迭代之前，最好看看模型在训练和推断过程中对样本数据有什么要求。
```python
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
data_loader = torch.utils.data.DataLoader(
 dataset, batch_size=2, shuffle=True, num_workers=4,
 collate_fn=utils.collate_fn)
# For Training
images,targets = next(iter(data_loader))
images = list(image for image in images)
targets = [{k: v for k, v in t.items()} for t in targets]
output = model(images,targets)   # Returns losses and detections
# For inference
model.eval()
x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
predictions = model(x)           # Returns predictions
```
现在让我们编写进行训练和验证的主函数
```python
from engine import train_one_epoch, evaluate
import utils


def main():
    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    num_classes = 2
    # use our dataset and defined transformations
    dataset = PennFudanDataset('PennFudanPed', get_transform(train=True))
    dataset_test = PennFudanDataset('PennFudanPed', get_transform(train=False))

    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    dataset = torch.utils.data.Subset(dataset, indices[:-50])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes)

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

    print("That's it!")
```
在第一轮你应该可以得到以下输出
```
Epoch: [0]  [ 0/60]  eta: 0:01:18  lr: 0.000090  loss: 2.5213 (2.5213)  loss_classifier: 0.8025 (0.8025)  loss_box_reg: 0.2634 (0.2634)  loss_mask: 1.4265 (1.4265)  loss_objectness: 0.0190 (0.0190)  loss_rpn_box_reg: 0.0099 (0.0099)  time: 1.3121  data: 0.3024  max mem: 3485
Epoch: [0]  [10/60]  eta: 0:00:20  lr: 0.000936  loss: 1.3007 (1.5313)  loss_classifier: 0.3979 (0.4719)  loss_box_reg: 0.2454 (0.2272)  loss_mask: 0.6089 (0.7953)  loss_objectness: 0.0197 (0.0228)  loss_rpn_box_reg: 0.0121 (0.0141)  time: 0.4198  data: 0.0298  max mem: 5081
Epoch: [0]  [20/60]  eta: 0:00:15  lr: 0.001783  loss: 0.7567 (1.1056)  loss_classifier: 0.2221 (0.3319)  loss_box_reg: 0.2002 (0.2106)  loss_mask: 0.2904 (0.5332)  loss_objectness: 0.0146 (0.0176)  loss_rpn_box_reg: 0.0094 (0.0123)  time: 0.3293  data: 0.0035  max mem: 5081
Epoch: [0]  [30/60]  eta: 0:00:11  lr: 0.002629  loss: 0.4705 (0.8935)  loss_classifier: 0.0991 (0.2517)  loss_box_reg: 0.1578 (0.1957)  loss_mask: 0.1970 (0.4204)  loss_objectness: 0.0061 (0.0140)  loss_rpn_box_reg: 0.0075 (0.0118)  time: 0.3403  data: 0.0044  max mem: 5081
Epoch: [0]  [40/60]  eta: 0:00:07  lr: 0.003476  loss: 0.3901 (0.7568)  loss_classifier: 0.0648 (0.2022)  loss_box_reg: 0.1207 (0.1736)  loss_mask: 0.1705 (0.3585)  loss_objectness: 0.0018 (0.0113)  loss_rpn_box_reg: 0.0075 (0.0112)  time: 0.3407  data: 0.0044  max mem: 5081
Epoch: [0]  [50/60]  eta: 0:00:03  lr: 0.004323  loss: 0.3237 (0.6703)  loss_classifier: 0.0474 (0.1731)  loss_box_reg: 0.1109 (0.1561)  loss_mask: 0.1658 (0.3201)  loss_objectness: 0.0015 (0.0093)  loss_rpn_box_reg: 0.0093 (0.0116)  time: 0.3379  data: 0.0043  max mem: 5081
Epoch: [0]  [59/60]  eta: 0:00:00  lr: 0.005000  loss: 0.2540 (0.6082)  loss_classifier: 0.0309 (0.1526)  loss_box_reg: 0.0463 (0.1405)  loss_mask: 0.1568 (0.2945)  loss_objectness: 0.0012 (0.0083)  loss_rpn_box_reg: 0.0093 (0.0123)  time: 0.3489  data: 0.0042  max mem: 5081
Epoch: [0] Total time: 0:00:21 (0.3570 s / it)
creating index...
index created!
Test:  [ 0/50]  eta: 0:00:19  model_time: 0.2152 (0.2152)  evaluator_time: 0.0133 (0.0133)  time: 0.4000  data: 0.1701  max mem: 5081
Test:  [49/50]  eta: 0:00:00  model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)  time: 0.0735  data: 0.0022  max mem: 5081
Test: Total time: 0:00:04 (0.0828 s / it)
Averaged stats: model_time: 0.0628 (0.0687)  evaluator_time: 0.0039 (0.0064)
Accumulating evaluation results...
DONE (t=0.01s).
Accumulating evaluation results...
DONE (t=0.01s).
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.606
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.984
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.780
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.313
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.582
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.612
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.270
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.672
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.755
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.664
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.704
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.979
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.871
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.325
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.727
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.316
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.748
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.749
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.650
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.673
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.758

```
在第一轮训练过后，我们得到的COCO-style的mAP值为60.6，蒙版的mAP值为70.4
在10轮的训练后，我得到的模型水准如下：
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.935
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.349
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.592
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.831
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.324
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.844
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.844
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.777
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.870
IoU metric: segm
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.761
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.969
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.919
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.341
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.464
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.788
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.303
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.799
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.400
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.769
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.818
```
但是预测怎么样呢？让我们用数据集中的一张图片验证一下：  
![图片](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image05.png "训练")
我们训练的模型在这张图片中发现了9人，让我们看看其中的2个  
![图片](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image06.png "训练")
![图片](https://pytorch.org/tutorials/_static/img/tv_tutorial/tv_image07.png "训练")
结果看上去还不错！  
## 总结
在本教程中，你已经学习了怎么使用你自己的数据集创建你自己的划分模型训练流，为此你编写了一会返回图片、ground truth boxes和划分蒙版的torch.utils.data.Dataset类你还使用了一个 为了在新数据集上实现转换学习而使用COCO train2017 预训练的Mask R-CNN模型。
为了使例子更加完整，使其包含多机器/多GPU训练，请检查references/detection/train.py，它位于torchvision 的仓库中。
你可以点击[这里](https://pytorch.org/tutorials/_static/tv-training-code.py)来下载完整的示例代码。





















