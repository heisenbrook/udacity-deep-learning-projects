### Fourth project - Face generation with GAN

 This project will focus on define and train a Generative Adverserial network on a dataset of faces. The goal is to get a generator network to generate new images of faces that look as realistic as possible.

 The project will be broken down into a series of tasks from defining new architectures training adversarial networks. At the end the generated samples should look like fairly realistic faces with small amounts of noise. They can be found in ``/generated_images`` repo.
 Some examples:


 ![Alt text](https://github.com/heisenbrook/udacity-deep-learning-projects/blob/main/4_GAN_project/generated_images/Image_1_epoch_1000)
 ![Alt text](https://github.com/heisenbrook/udacity-deep-learning-projects/blob/main/4_GAN_project/generated_images/Image_2_epoch_1000)
 ![Alt text](https://github.com/heisenbrook/udacity-deep-learning-projects/blob/main/4_GAN_project/generated_images/Image_3_epoch_1000)
 ![Alt text](https://github.com/heisenbrook/udacity-deep-learning-projects/blob/main/4_GAN_project/generated_images/Image_4_epoch_1000)

1. # Get the Data:

 The CelebFaces Attributes Dataset (CelebA) will be used to train the adversarial networks. It is suggested the use of a GPU for training.

2. # Pre-processed Data:

 Since the project's main focus is on building the GANs, we've done some pre-processing. Each of the CelebA images has been cropped to remove parts of the image that don't include a face, then resized down to 64x64x3 NumPy images.

 The archive containing the images has been splitted into smaller archives and are available in ``/archives`` repo.