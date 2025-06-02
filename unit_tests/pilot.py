import sys
print("In module products sys.path[0], __package__ ==", sys.path[0], __package__)
from ..bc_models.bearnet import BearNet

# SETUP
# Instantiate BearNet
# random_pilot = BearNet()
# random_pilot.eval()
#
# # MAIN
# from torchinfo import summary
# model = BearNet()  # Adjust num_classes as needed
# batch_size = 1
# summary(model, input_size=(batch_size, 3, 224, 224))

