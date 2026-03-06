import numpy as np
import torch
import torch.nn as nn

def model_forward_single_layer(model, inputs, targets_len, num_layers):
    outputs = []
    states = [None] * len(num_layers)
    # states = [None] * num_layers
    # states = [None] 
    inputs_len = inputs.shape[1]
    
    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        output, states = model(inputs[:, i], states)
        outputs.append(output)

    for i in range(targets_len):
        output, states = model(last_input, states)
        outputs.append(output)
        last_input = output

    return outputs


def model_forward_multi_layer(model, inputs, targets_len, num_layers):
    states_down = [None] * len(num_layers)
    states_up = [None] * len(num_layers)

    outputs = []

    # print(f"shape of input in multilayer: {inputs.shape}")
    
    inputs_len = inputs.shape[1]

    last_input = inputs[:, -1]

    for i in range(inputs_len - 1):
        # print(f"\n\nshape of inputs[:, i] in multilayer: {inputs[:, i].shape}")
        output, states_down, states_up = model(inputs[:, i], states_down, states_up)
        outputs.append(output)

    for i in range(targets_len):
        # print(f"\n\nshape of last_input in multilayer: {last_input.shape}")
        output, states_down, states_up = model(last_input, states_down, states_up)
        outputs.append(output)
        last_input = output

    return outputs