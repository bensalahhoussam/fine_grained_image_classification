# fine_grained_image_classification 
Recognizing fine-grained categories highly relies on discriminative part localization and part based fine-grained feature learning.

The intuition behind the Bilinear CNN’s can be understood as simple parallel CNN’s each trying to identify different feature of same image. Crudely putting it, in identification of a particular bird species, 2 parallel CNN’s can be used one would identify beak and other would identify a tail and by multiple CNN’s identify different features.

Bilinear CNN’s are simple parallel CNN’s which are combined using matrix outer product.The outputs from CNN’s are taken before the FC layers.


![1_gr4snhggh9jV1F3mnHeErg](https://user-images.githubusercontent.com/112108580/194578146-f646b290-a318-4d84-abfe-ca3581194998.png)
## References 
Bilinear CNN Models for Fine-grained Visual Recognition- https://doi.org/10.48550/arXiv.1504.07889
