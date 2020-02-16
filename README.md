# Easy-garden
Machine Learning model to take care of your plants easily.
## An easy way to take care of a garden
Most of the people like having plants in their home and office, because hey are beautiful and can connect us with nature just a little bit. But most of the time we really don't take care of them, and they could get sick. Thinking of this, a system that can monitor the health of plants and tell you if one of them got a disease could be helpful. The system needs to be capturing images at real time and then classify it in diseased or healthy, in case of a disease it can notify you or even provide a treatment for the plant.
## What it does
The model takes an input image and classify it into healthy or disease.
## Data
I use datasets of plants healthy and disease found in PlantVillage and developed a machine learning model in Tensorflow using keras API, getting either healthy (1) or diseased (0).
The data set consisted in a total of 3376 images in the data set in which 1942 images are of category diseased and 1434 images are of category healthy. The size of each image is different so the image dimension. Most of the images are in jpeg but also contains some images in .png and gif.
## Preparation of the data
To feed the maching learning model, it was needed to convert each pixel of the RGB color images to a pixel with value betwwen 0 and 1, and resize all the images to a dimension of 170 x 170.

I use Tensorflow to feed the data to neural network, and created 3 datasets with different distributions of data, Training: 75%, Valid: 15% and Testing: 10%.
## Model
Testing a few differents models of convolution neural networks, ending with the model Vgg16, pre trained on imagenet. The Vgg 16 has two dense layers. Each dense layer has 'relu' activation. And it has two convolutional layers at the end.

The model was trained with training set and after each epoch the model is tested against validation set. Once the model is trained it can be tuned by retraining the model on the last two convolutional layers, using a lower value of learning rate.
## Approach
With the created model it is aime to develop an app that uses that model to identify if our plants are heathy or there are problems with them, in case there is the second option, the app will give some advises of how to treat it in real time.

Using Azure Machine Learning and Kinvy SDK we could make the camara obtein some images per a certain time and analice it with the Tensorflow model and poping up little signboards announcing good or bad news. In case the plant has a disease, the sign will contain the option to expand into some instruction to follow and take care of the plant.

## Future Work
The current idea whas just for a system to notify about a disease and provide information about it, but it can be extended to build a mechanism that automatically can take care of the plants, giving them water, nutrients, fertilizer or even light pesticides, whenever the algorithm considers it necessary.
