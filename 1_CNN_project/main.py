from src.helpers import setup_env
from src.data import get_data_loaders
from src.train import optimize
from src.optimization import get_optimizer, get_loss
from src.model import MyModel

from src.train import one_epoch_test
from src.model import MyModel
import torch

batch_size = 32        # size of the minibatch for stochastic gradient descent (or Adam)
valid_size = 0.2       # fraction of the training data to reserve for validation
num_epochs = 40       # number of epochs for training
num_classes = 50       # number of classes. Do not change this
dropout = 0.6          # dropout for our model
learning_rate = 0.001  # Learning rate for SGD (or Adam)
opt = 'adam'            # optimizer. 'sgd' or 'adam'
weight_decay = 0.01    # regularization. Increase this to combat overfitting

# If running locally, this will download dataset 
setup_env()


# get the data loaders using batch_size and valid_size defined in the previous
# cell
# HINT: do NOT copy/paste the values. Use the variables instead
data_loaders = get_data_loaders(batch_size, valid_size,)

# instance model MyModel with num_classes and drouput defined in the previous
# cell
model = MyModel(num_classes, dropout)

# Get the optimizer using get_optimizer and the model you just created, the learning rate,
# the optimizer and the weight decay specified in the previous cell
optimizer = get_optimizer(model, opt, learning_rate, weight_decay)

# Get the loss using get_loss
loss = get_loss()

optimize(
    data_loaders,
    model,
    optimizer,
    loss,
    n_epochs=num_epochs,
    save_path="checkpoints/best_val_loss.pt",
    interactive_tracking=True
)

# YOUR CODE HERE: load the weights in 'checkpoints/best_val_loss.pt'
model = torch.load('checkpoints/best_val_loss.pt')
model.eval()

# Run test
one_epoch_test(data_loaders['test'], model, loss)