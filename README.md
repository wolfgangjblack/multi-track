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
    - first corner (x1, y2)
    - second corner (x1, y1)
    - third corner  (x2, y1)
    - fourth corner (x2, y2)

Readers should note, $x_1 < x_2$ and $y1 < y2$
In addition to the bounding box, each image is provided with a number of per-frame attributes, in the `.label` files. 

Unfortunately no key was givien with this data so while some may be obvious others are a bit of a mystery.

- occlusion.label: Is the target partially or fully blocked by objects like trees, etc
- illum_change.label: where illumination changes
- camera_motion.label: is the camera moving
- motion_change.label: does the target appear to be accelerating
- size_change.label: apparent target size change due to perspective (size of bounding boxes?)

## Approach

As an AI engineer who prioritizes production code and getting MVP prototypes I was thinking about this less in what models can I train and more what models can I utilize day 1. I'm going to pull down some established traditional ML/DL models from huggingface/torch hub to use to determine current pipeline capabilities. As we develop more test cases, interest in other objects, etc we can finetune these models for our usecases. My mvp looks like this:

### vehicle tracker

1. video_to_frames.py
    - Desc: input video and extract frames
    - Usage: `python src/video_to_frames.py --config "config/video.json"`
    - Note: I shared this step to show preliminary data processing. Ideal we'd have done this in a way where we could train all the models used downstream in the pipeline. This was an extra step, while I wont use other videos/sequences of images to train I AM curious about how well this will perform on various sources of data. To prep this, I ended up writing a script to transform videos into frames for processing. 
2. inference.py
    Pipeline Flow:
    1. Model Loading
        - Load pretrained YOLO model
        - Initialize VehicleTracker
    
    2. Frame Processing Loop
        - Load frames sequentially
        - YOLO detection → bounding boxes
        - Feature extraction from boxes
        - Use previous position and current position to get velocity, velocity can be used to predict next position
        - use similarity between detected objects using object identification (bb from yolo) and predicted bb (velocity prediction), if similarity is high we track object
        - Visualization and saving
    
    Output:
        - Annotated frames with tracked objects
        - Optional bbox.txt with tracking data
    Usage: `python src/inference.py --conifg "config/inference.json"`
3. eval.py
    - Desc: If we have ground truth labels we can get an IoU evaluation. Do our boxes overlap the object. We'll accept 85% and above 
    - Usage `python src/eval.py --config "config/eval.json"`
    - Note: This currently only works with single_class examples. Can scale this up to multiclass examples

I also include some pseudocode for model training. This is pseudo-code but could be quickly verified to work to help use improve our model. 
- train.py
    - Desc: this pulls training code for yolo provided by ultralytics and my own training code for the backbone feature extractor (resnet18)
        - yolo reference: [ultralyrics training code](https://docs.ultralytics.com/modes/train/#train-settings)
        - resnet18 reference: [Resnet traning code I wrote for another project](https://github.com/wolfgangjblack/multimodal-moderation-pipeline/blob/main/src/resnet_training_utils.py)
        - transformer training reference: [Vision Transformer finetuning code I wrote](https://github.com/wolfgangjblack/multimodal-moderation-pipeline/blob/main/src/vit_training_utils.py)
    - shortcomings: we really need to flesh out the data utils for how the frames save data to be used for training. ideally we'd save the bounding boxes and their cropped images by class. This could be used by both models. We'd maintain the yolo class indexes, but would have to modify them for the resnet/whatever backbone model we used
- I also added a pytest framework for unittests. to run locally try `pytest --cov=project --cov-report=term-missing`

### Drawbacks and other ideas
1. I considered using a segmentation model (ClipSeg or SAM) instead of a YOLO model, but these id models don't output bounding boxes natively
    - what's interesting is we can utilize these models in concert with a VLM and sort of interrogate the images. We can capture all the objects in a sequence of frames and ask the VLMs to describe the motion, behavior, changes, etc. This would be a great project but beyond the scope of this quick interview
2. For tracking I did implement Kalman (filterpy) - then i saw it wasn't allowed. 
3. We have some short comings when trees/objects block the car. I beleive this is largely in part due to the yolo model. I'd like to explore using other detection models to determine if we can improve this. AS an aside the YOLO model implementation can totally drop or displace smaller objects like motorcycles. 
4. I'd like to explore using a more sophisticated model for the feature extractor. The feature extractor I'd consider using with more data is a DeiT. These transformer models have better feature extraction techniques, but might be over kill for a single car method
5. I choose the YOLO and resnet18 models for a number of reasons:
 - YOLO:
    - it's already trained to detect vehicles
    - It's fast and consistently supported/update. 
    - supports cuda/batching
    - can do streaming, work on front end, is free
 - Resnet18 as feature extractor:
    - lightweight
    - trained on ImageNet which has cars
    - can be used in Onnx/TensorRT
    - supports batching
6. Does not really handle occlusion, blurring, or light. We can improve this utilizing all of this - but since I'm not a perception engineer, rather more an ML/GenAI engineer I'd have to do more research there. 

I also tend to start projects with simple models over more complex models, just so we can understand the limits to the data/performance as well as try to save costs. 