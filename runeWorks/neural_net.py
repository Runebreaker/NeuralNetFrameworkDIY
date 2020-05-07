from abc import ABC

import numpy as np
import abc
import lazy

class Shape:
    def __init__(self, a: int, b: int):
        self.axis = [[0 for x in range(a)] for y in range(b)]
        self.volume = np.linalg.det(self.axis)

class Tensor:
    def __init__(self, elements: float, shape: Shape, deltas: float):
        self.elements = elements
        self.shape = shape
        self.deltas = deltas

class Layer:
    @abc.abstractmethod
    def forward(self, inTensors: Tensor, outTensors: Tensor):
        pass
    @abc.abstractmethod
    def backward(self, outTensors: Tensor, inTensors: Tensor):
        pass
    @abc.abstractmethod
    def calculateDeltaWeights(self, outTensors: Tensor, inTensors: Tensor)):
        pass

class FullyConnectedLayer(Layer):
    def __init__(self, Weightmatrix: Tensor, Bias: Tensor, InShape: Shape, OutShape: Shape):
        self.W = Weightmatrix
        self.B = Bias
        self.InS = InShape
        self.OuS = OutShape

    def forward(self, inTensors: Tensor, outTensors: Tensor):
        outTensors = inTensors * self.W + self.B

    def backward(self, outTensors: Tensor, inTensors: Tensor):
        pass

    def calculateDeltaWeights(self, outTensors: Tensor, inTensors: Tensor):
        pass


class SoftmaxLayer(Layer):

    def forward(self, inTensors: Tensor, outTensors: Tensor):
        pass

    def backward(self, outTensors: Tensor, inTensors: Tensor):
        pass

    def calculateDeltaWeights(self, outTensors: Tensor, inTensors: Tensor):
        pass


class ActivationLayer(Layer):

    def forward(self, inTensors: Tensor, outTensors: Tensor):
        pass

    def backward(self, outTensors: Tensor, inTensors: Tensor):
        pass

    def calculateDeltaWeights(self, outTensors: Tensor, inTensors: Tensor):
        pass


class InputLayer:
    def Forward(rawData) -> Tensor:
        pass
