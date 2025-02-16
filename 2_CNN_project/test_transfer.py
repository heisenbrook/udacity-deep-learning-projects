import torch
from src.data import get_data_loaders
from src.optimization import get_optimizer, get_loss
from src.train import one_epoch_test
from src.transfer import get_model_transfer_learning
from src.predictor import Predictor
from src.helpers import compute_mean_and_std
from src.predictor import predictor_test
from src.helpers import plot_confusion_matrix


batch_size = 64  # size of the minibatch for stochastic gradient descent (or Adam)
valid_size = 0.2  # fraction of the training data to reserve for validation
num_epochs = 50  # number of epochs for training
num_classes = 50  # number of classes. Do not change this
learning_rate = 0.001  # Learning rate for SGD (or Adam)
opt = 'adam'      # optimizer. 'sgd' or 'adam'
weight_decay = 0.0 # regularization. Increase this to combat overfitting

model_transfer = get_model_transfer_learning("resnet18", n_classes=num_classes)
# Load saved weights
data_loaders = get_data_loaders(batch_size=batch_size)
loss = get_loss()
model_transfer = torch.load('checkpoints/model_transfer.pt', 
                                          map_location=torch.device('cpu'))

one_epoch_test(data_loaders['test'], model_transfer, loss)

# First let's get the class names from our data loaders
class_names = data_loaders["train"].dataset.classes

# Then let's move the model_transfer to the CPU
# (we don't need GPU for inference)
model_transfer = model_transfer.cpu()
# Let's make sure we use the right weights by loading the
# best weights we have found during training
# NOTE: remember to use map_location='cpu' so the weights
# are loaded on the CPU (and not the GPU)
model_transfer =torch.load("checkpoints/model_transfer.pt", map_location="cpu")


# Let's wrap our model using the predictor class
mean, std = compute_mean_and_std()
predictor = Predictor(model_transfer, class_names, mean, std).cpu()

# Export using torch.jit.script
scripted_predictor = torch.jit.script(predictor)
scripted_predictor.save("checkpoints/transfer_exported.pt")

model_reloaded = torch.jit.load("checkpoints/transfer_exported.pt")

pred, truth = predictor_test(data_loaders['test'], model_reloaded)

plot_confusion_matrix(pred, truth)