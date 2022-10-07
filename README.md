# fine_grained_image_classification 
Recognizing fine-grained categories highly relies on discriminative part localization and part based fine-grained feature learning.

The intuition behind the Bilinear CNN’s can be understood as simple parallel CNN’s each trying to identify different feature of same image. Crudely putting it, in identification of a particular bird species, 2 parallel CNN’s can be used one would identify beak and other would identify a tail and by multiple CNN’s identify different features.

Bilinear CNN’s are simple parallel CNN’s which are combined using matrix outer product.The outputs from CNN’s are taken before the FC layers.


![1_gr4snhggh9jV1F3mnHeErg](https://user-images.githubusercontent.com/112108580/194578146-f646b290-a318-4d84-abfe-ca3581194998.png)


## Problem Objectives:
The aim of this challenge is to build a Generalised Model for the task of Image Classification. We have to also deal with Class Imbalance Problem and detect Fine Grained details.

![Screenshot 2022-10-07 161109](https://user-images.githubusercontent.com/112108580/194587792-8a2420b8-2276-4ef1-9072-a355e04c4c0b.png)


## Exprimenting with Loss Functions

| model | score|
| --- | --- |
| `ResNET-50 with Cross Entropy` | 0.8825 |
| `ResNET-50 with Focal Loss`` | 0.892 |
| `ResNET-50 with Label Smoothing` | 0.9 |
| `ResNET-50 with Focal Loss + Label Smoothing` | 0.924|

## References 
Bilinear CNN Models for Fine-grained Visual Recognition- https://doi.org/10.48550/arXiv.1504.07889
https://towardsdatascience.com/what-is-label-smoothing-108debd7ef06
