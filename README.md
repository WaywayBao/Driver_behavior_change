## Risk Object Identification

This work was done when Chengxi Li was an intern at Honda Research Institute, San Jose, CA

#### Environment
The code is developed with CUDA 10.2, ***Python 3.6***, ***PyTorch = 1.0.0***


### How to get tracker
Step 1: Use Detectron.pytorch to generate detection.txt for each clip
Step 2: Use deep_sort to generate detection features
Step 3: Use deep_sort to generate tracking.txt
Step 4: Use script 'refine_tracking' to generate refined_tracking.txt


