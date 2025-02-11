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
    - Usage: `python src/video_to_frames.py --video_uri <local path to video> --output_dir <output path for frames>`
    - Note: This was an extra step, while I wont use other videos/sequences of images to train I AM curious about how well this will perform on various sources of data. To prep this, I ended up writing a script to transform videos into frames for processing. 
2. inference.py
    Pipeline Flow:
    1. Model Loading
        - Load pretrained YOLO model
        - Initialize VehicleTracker
    
    2. Frame Processing Loop
        - Load frames sequentially
        - YOLO detection → bounding boxes
        - Feature extraction from boxes
        - Track updating and ID maintenance
        - Visualization and saving
    
    Output:
        - Annotated frames with tracked objects
        - Optional bbox.txt with tracking data
    Usage: `python src/inference.py --data_dir <path> --output_dir <path> --save_text <bool>`
3. eval.py
    -Usage `python src/eval.py --ground_truth_dir <path> --predicted_bb_dir <path>`
    -Note: This currently only works with single_class examples. Can scale this up to multiclass examples

### Benefits
- no training data necessary for prototype
    - training can improve results, especially on blurry images and new classes/cars
    - can explore better architectures
- consistent id maintanence across frames
- handles occlusions via kalman prediction
- class-aware tracking

### 