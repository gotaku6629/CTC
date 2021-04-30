"""
# tuple test
for future prediction
    want to change the contents of tupel
    (class Charades ... train_datset, val_dataset)

must change once tuple to list to change the content
and then, rechange the list to tuple
"""
dataset = (0, 1, 2)

print('dataset = ', dataset)
print('type(dataset) = ', type(dataset))


"""
# one of all videos, and one picture of one video
it is result of 4 skipping one video
    train_dataset[0]
"""
import numpy as np
import torch

# 1. make train_dataset
dataset_train = []
for i in range(100):
    input_data = torch.rand(224, 224)
    s_target = 9
    v_target = torch.rand(32)
    o_target = torch.rand(38)
    if 0 <= i < 12: 
        meta = {'id': '46GP8', 'time': i}
    elif 12 <= i < 25: 
        meta = {'id': 'YH5SD', 'time': i-12}
    elif 25 <= i < 42: 
        meta = {'id': '0F4FS', 'time': i-25}
    elif 42 <= i < 63: 
        meta = {'id': 'SKJFE', 'time': i-42}
    elif 63 <= i < 89: 
        meta = {'id': '8DK4F', 'time': i-63}
    elif 89 <= i < 100: 
        meta = {'id': 'EJ2KD', 'time': i-89}
    dataset = (input_data, s_target, v_target, o_target, meta)
    dataset_train.append(dataset)

# 2. try to change dataset
dataset_train_new = []

for i in range(len(dataset_train)-1):
    current_id = list(dataset_train[i])[4]['id']
    next_id = list(dataset_train[i+1])[4]['id']
    if current_id == next_id:
        dataset_train_new.append(dataset_train[i])

print(len(dataset_train_new)) # 94  Very Good!!
