## X-PDNet: Accurate Joint Plane Instance Segmentation and Monocular Depth Estimation with Cross-Task Distillation and Boundary Correction (BMVC 2023)
This is an implementation for X-PDNet: a multi-task learning framework for joint plane instance segmentation and depth estimation

The official paper can be found at [paper](https://arxiv.org/pdf/2309.08424v2.pdf). Thank the [PlaneRecNet](https://github.com/EryiXie/PlaneRecNet) for a great baseline implementation

![Network Architecture](/images/X_PDNet.png)



### How to run the inference?
1. Use conda to create an env:
```
conda env create -f environment.yml
```

2. Create a folder "weights", download resnet and X-PDNet checkpoint via this [link](https://drive.google.com/drive/folders/1Y5cIHAKG44lEz0O6eNOfxoc7oRKMEFzL?usp=sharing) and put on "weights" folder.
3. Inference a single image (*.png or *.jpg for mat):
```
python3 simple_inference.py --config=XPDNet_101_config --trained_model=weights/XPDNet_101_9_125000.pth  --image=example
_images/scene0134_01_frame_color_756.jpg

```
4. Inference a folder:
```
python3 simple_inference.py --config=XPDNet_101_config --trained_model=weights/XPDNet_101_9_125000.pth --images=input_folder:output_folder

```