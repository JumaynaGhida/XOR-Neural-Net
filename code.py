import math
import random

isize = 2
hsize = 3
osize = 1
LR = 0.1
epochs = 1000

weights_ih = [[random.uniform(-0.5, 0.5) for _ in range(hsize)] for _ in range(isize)]
biases_h = [random.uniform(-0.5, 0.5) for _ in range(hsize)]
weights_ho = [[random.uniform(-0.5, 0.5) for _ in range(osize)] for _ in range(hsize)]
biases_o = [random.uniform(-0.5, 0.5) for _ in range(osize)]

training_data = [
    ([0, 0], [0]),
    ([0, 1], [1]),
    ([1, 0], [1]),
    ([1, 1], [0])
]

for epoch in range(epochs):
    for inputs, targets in training_data:
        hinputs = [0.0] * hsize
        for j in range(hsize):
            for i in range(isize):
                hinputs[j] += inputs[i] * weights_ih[i][j]
            hinputs[j] += biases_h[j]
            hinputs[j] = 1 / (1 + math.exp(-hinputs[j]))

        oinputs = [0.0] * osize
        for k in range(osize):
            for j in range(hsize):
                oinputs[k] += hinputs[j] * weights_ho[j][k]
            oinputs[k] += biases_o[k]
            oinputs[k] = 1 / (1 + math.exp(-oinputs[k]))

        oerror = [0.0] * osize
        for k in range(osize):
            oerror[k] = targets[k] - oinputs[k]

        output_deltas = [0.0] * osize
        for k in range(osize):
            output_deltas[k] = oerror[k] * oinputs[k] * (1 - oinputs[k])

        hidden_errors = [0.0] * hsize
        for j in range(hsize):
            for k in range(osize):
                hidden_errors[j] += output_deltas[k] * weights_ho[j][k]

        hiddend = [0.0] * hsize
        for j in range(hsize):
            hiddend[j] = hidden_errors[j] * hinputs[j] * (1 - hinputs[j])

        for j in range(hsize):
            for k in range(osize):
                weights_ho[j][k] += LR * output_deltas[k] * hinputs[j]
        for k in range(osize):
            biases_o[k] += LR * output_deltas[k]

        for i in range(isize):
            for j in range(hsize):
                weights_ih[i][j] += LR * hiddend[j] * inputs[i]
        for j in range(hsize):
            biases_h[j] += LR * hiddend[j]

inputs = [0, 0]
hinputs = [0.0] * hsize
for j in range(hsize):
    for i in range(isize):
        hinputs[j] += inputs[i] * weights_ih[i][j]
    hinputs[j] += biases_h[j]
    hinputs[j] = 1 / (1 + math.exp(-hinputs[j]))

oinputs = [0.0] * osize
for k in range(osize):
    for j in range(hsize):
        oinputs[k] += hinputs[j] * weights_ho[j][k]
    oinputs[k] += biases_o[k]
    oinputs[k] = 1 / (1 + math.exp(-oinputs[k]))

print(oinputs)

inputs = [0, 1]
hinputs = [0.0] * hsize
for j in range(hsize):
    for i in range(isize):
        hinputs[j] += inputs[i] * weights_ih[i][j]
    hinputs[j] += biases_h[j]
    hinputs[j] = 1 / (1 + math.exp(-hinputs[j]))

oinputs = [0.0] * osize
for k in range(osize):
    for j in range(hsize):
        oinputs[k] += hinputs[j] * weights_ho[j][k]
    oinputs[k] += biases_o[k]
    oinputs[k] = 1 / (1 + math.exp(-oinputs[k]))

print(oinputs)

inputs = [1, 0]
hinputs = [0.0] * hsize
for j in range(hsize):
    for i in range(isize):
        hinputs[j] += inputs[i] * weights_ih[i][j]
    hinputs[j] += biases_h[j]
    hinputs[j] = 1 / (1 + math.exp(-hinputs[j]))

oinputs = [0.0] * osize
for k in range(osize):
    for j in range(hsize):
        oinputs[k] += hinputs[j] * weights_ho[j][k]
    oinputs[k] += biases_o[k]
    oinputs[k] = 1 / (1 + math.exp(-oinputs[k]))

print(oinputs)

inputs = [1, 1]
hinputs = [0.0] * hsize
for j in range(hsize):
    for i in range(isize):
        hinputs[j] += inputs[i] * weights_ih[i][j]
    hinputs[j] += biases_h[j]
    hinputs[j] = 1 / (1 + math.exp(-hinputs[j]))

oinputs = [0.0] * osize
for k in range(osize):
    for j in range(hsize):
        oinputs[k] += hinputs[j] * weights_ho[j][k]
    oinputs[k] += biases_o[k]
    oinputs[k] = 1 / (1 + math.exp(-oinputs[k]))

print(oinputs)
