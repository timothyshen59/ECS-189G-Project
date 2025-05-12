Dataset
There are 3 different datasets to be used for stage 3 of the project.
ORL: The ORL is about human face images. There are 40 different person in the dataset, each person we have 10 images (9 for training, 1 for testing). The images are reduced to gray scale from colored images by assigning the RGB channels with the same coding values. So, only one channel e.g., R channel, is only needed for building the CNN model. However, if you prefer to use all these RGB channels (although they have the same codes), it is also fine. Each image is a matrix of shape 112x92x3. The labels are images from {1, 2, …, 40}.
MNIST: The MNIST dataset is about hand-written digit images from 0 to 9. Each image is a matrix of shape 28x28. The labels of the images are integers from {0, 1, 2, …,9} indicating the digit of these images.
CIFAR-10: The CIFAR-10 dataset is about 10 different colored objects. The images have very low-resolutions, which are matrix of shape 32x32x3. The labels of these objects are integers from {0, 1, 2, …,9} indicating different classes of the objects.

Dataset statistics:
ORL Dataset (gray-sacle)
Training instance number: 360
Testing instance number: 40
Image size: 112x92x3 (actually the RGB codes are identical, since it is a gray-scale image. You only need to use one channel (e.g., the R channel) for building the CNN model.)
Labels: each image is associated with one single label indicating the person. The label is an integer from {1, 2, …, 40}, for the images with the same label, they are about the same person.

MNIST Dataset (gray-sacle)
Training instance number: 60000
Testing instance number: 10000
Image size: 28x28 (the images are gray-scale with only one channel)
Labels: each image is associated with one single label indicating the digit number. The label is an integer from {0, 1, 2, …,9}, for the images with the same label, they are about the same digit number.

CIFAR-10 Dataset (colored)
Training instance number: 50000
Testing instance number: 10000
Image size: 32x32x3 (they are all colored images)
Labels: each image is associated with one single label indicating the object in the image. The label is an integer from {0, 1, 2, …,9}, for the images with the same label, they are about the same object.


Dataset Loading Code
In the compressed folder, we attach a script_data_loader.py script file for loading and print the images in the dataset. The organization of the datasets are illustrated as follows. We have partitioned the datasets into training and testing sets already.

Dataset Organization Structure

loaded_dataset = {
“train”: [
# train instance 1
{‘image’: [… a matrix of the image … ]
‘label’: an integer denoting the label}
# train instance 2
{‘image’: [… a matrix of the image … ]
‘label’: an integer denoting the label}
….
],
“test”: [
# test instance 1
{‘image’: [… a matrix of the image … ]
‘label’: an integer denoting the label}
# test instance 2
{‘image’: [… a matrix of the image … ]
‘label’: an integer denoting the label}
….
]
}

Task TBD: For each dataset, based on the provided training set, please train a CNN model, and evaluate its performance on the testing set. Also try to change the configurations of the provided CNN model and report the results to see the impacts of configuration on the model performance. Write a report about your experimental process and results based on the provided report template (5 pages at the maximum).

(Optional) if you have GPUs that can run cuda, you can also try to play it on GPUs to compare its learning efficiency vs CPUs.