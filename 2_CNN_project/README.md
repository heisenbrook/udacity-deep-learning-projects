### Second project - CNN - Algorithms for landmark classification

In the very first part of this project, you will create a CNN that classifies landmarks. You must create your CNN from scratch (so, you can't use transfer learning yet!), and you must attain a test accuracy of at least 50%.

Although 50% may seem low at first glance, it seems more reasonable after realizing how difficult of a problem this is. Many times, an image that is taken at a landmark captures a fairly mundane image of an animal or plant, like in the following picture.

Just by looking at that image alone, would you have been able to guess that it was taken at the Haleakalā National Park in Hawaii?

An accuracy of 50% is significantly better than random guessing, which would provide an accuracy of just 2% (100% / 50 classes). In Step 2 of this notebook, you will have the opportunity to greatly improve accuracy by using transfer learning to create a CNN.

Experiment with different architectures, hyperparameters, training strategies, and trust your intuition. And, of course, have fun!

The second part of this task will involve transfer learning. Let's see how hard it is to match that performance with transfer learning.

The third and last part will be about building a simple app that uses our exported model