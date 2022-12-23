# Potato-diseases-detection
An academic deep learning project: Potato diseases have grown more present in the past years. This created the need to develop a solution that helps farmers detect the diseases in an early stage to not lose their production.

## Data collection
Our dataset is ready-made data from Kaggle named "PlantVillage". It contains 2152 images.

## Data Exploration
We conducted an exploratory data analysis that allowed us to further understand our data.
Our dataset is split into 3 classes: Healthy with 152 images, early blight and late blight each with 1000 images.

This shows a clear imbalance in our dataset!
![Dataset distibution](https://github.com/KhalilSRIDI/Potato-diseases-detection/blob/main/Readme%20images/datasetimabalance.png)

Here are some example images of our dataset
![Dataset exemples](https://github.com/KhalilSRIDI/Potato-diseases-detection/blob/main/Readme%20images/examples.png)

## Data Preprocessing
Our initial assumption is that, because we are working on images, we might need to apply some image processing techniques to help improve the model performance later on.
### Greyscale
We started by applying a greyscale filter that yielded images like the following:
![Greyscale image](https://github.com/KhalilSRIDI/Potato-diseases-detection/blob/main/Readme%20images/greyscale.png)

This showed us that there will be no clear improvement as the disease spots are not more clear in the greyscale colour space.

### Segmentation
We moved on and applied 2 segmentation techniques to our images:
The first one is a normal segmentation where we removed the background.

![Segmentation](https://github.com/KhalilSRIDI/Potato-diseases-detection/blob/main/Readme%20images/segmentation.png)

The second one is a segmentation into the HSV colour space.

![Segmentation HSV](https://github.com/KhalilSRIDI/Potato-diseases-detection/blob/main/Readme%20images/HSVSegmentation.png)

## Modeling
For the modeling phase, we worked simultaneously on a model from scratch and some pretrained models.
You can use the scripts to run all the models that we worked on.
