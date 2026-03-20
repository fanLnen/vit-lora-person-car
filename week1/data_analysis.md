# COCO Person-Car Dataset Analysis

## Category IDs

person: 1  
car: 3  

## Instance Statistics(train)

person instances: 262465  
car instances: 43867  

## Image Statistics

only person images: 55596  
only car images: 3732  
person + car images: 8519  

total valid images: 67847  

## Instance Statistics(val)

person instances: 11004
car instances: 1932

## Image Statistics

only person images: 2334
only car images: 176
person + car images: 359

total valid images: 2869


## Visualization

Random images from the dataset were visualized using pycocotools and matplotlib.
Bounding boxes for person and car instances were drawn to verify annotation correctness.

## Dataset Construction

The COCO detection dataset was converted into a multi-label image classification dataset.

Each image is assigned a label vector:

[is_person, is_car]

Examples:

[1,0] → person only  
[0,1] → car only  
[1,1] → person and car