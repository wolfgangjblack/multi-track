from typing import List

def read_positions_from_file(filename: str,
                             indices: List[int]= [0,5,4,1]):
    """
    Reads in x1, y1, x2, y2 from a text file given the textfile location
    the indices list maps the variable to the correct x,y value
    - for groundtruth use [0,5,4,1]
    - for predictions from our feature tracker use [1,2,3,4] 
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    output = []
    for line in lines:
        line = [i for i in line.split('\n')[0].split(',')[:6]]        
        output.append(
            [
                float(line[indices[0]]), 
                float(line[indices[1]]), 
                float(line[indices[2]]),
                float(line[indices[3]])
            ]
        )

    return output

def calculate_iou(box1, box2):
    # Determine the coordinates of the intersection rectangle
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the area of intersection rectangle
    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)

    # Calculate the area of both bounding boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate the IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)

    return iou