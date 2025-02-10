# multi-track
This repo contains a model pipeline to detect objects within a video or series of frames, and then track their movements throughout the video

## Goal

Implement an algorithm which tracks a vehicle through a set of images in `/data`. 
- write a model/pipeline/algorithm that scales well in inference and training
- interested more in laying groundwork for sophstication/long term performance and less interested in current results
- can use pretrained weights

## Data

- A sequence of images is given in `/data`, which is drawn from the VOT 2014 challenge dataset. You have been provided with annotations for the sequence.

- Bounding boxes are given in `groundtruth.txt`. Each bounding box is a comma separated list of 8 floating point numbers, organized in (x,y) order:
    - first corner 
    - second corner 
    - third corner 
    - fourth corner

In addition to the bounding box, each image is provided with a number of per-frame attributes, in the `.label` files. 

Unfortunately no key was givien with this data so while some may be obvious others are a bit of a mystery.

- occlusion.label: Is the target partially or fully blocked by objects like trees, etc
- illum_change.label: where illumination changes
- camera_motion.label: is the camera moving
- motion_change.label: does the target appear to be accelerating
- size_change.label: apparent target size change due to perspective (size of bounding boxes?)

## Approach

As an AI engineer who prioritizes production code and getting MVP prototypes I was thinking about this less in what models can I train and more what models can I utilize day 1. I'm going to pull down some established traditional ML/DL models from huggingface/torch hub to use to determine current pipeline capabilities. As we develop more test cases, interest in other objects, etc we can finetune these models for our usecases. My first pipeline looked like this:

### vehicle tracker
- input video or frames for processing
- use pretrained YOLO for vehicle detection 
- use pretrained resnet18 as feature extractor
- use kalman filter with history to track objects and maintain context between frames
- output frames with objects marked by ID and bounding box

### Benefits
- no training data necessary
- tracking algorithm is efficient
- ML models are easy to finetune
- feature extractor 