import numpy as np
from PIL import Image
from boxer_classifier import BoxerClassifier
from app import App

classifer = BoxerClassifier(224, "./data/models/boxer_cnn_model")
app = App(classifer, 224)
app.run()
