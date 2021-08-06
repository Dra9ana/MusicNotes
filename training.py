import torch
from orm_model import CTC_CNN, default_model_params
from orm_dataset import CTC_PriMuS
import os
import time

def decode(target):
    decoded = []
    decoded_2 = []
    prev = 0
    for note in target:
        if note == prev:
            continue
        else:
            decoded.append(note)
            prev = note
    for note in decoded:
        if note != 0:
            decoded_2.append(note)
    return decoded_2

data_dir = './package_aa'
dict_path = './vocabulary_semantic.txt'
img_height = 128
learning_rate = 0.04
num_epochs = 100
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

corpus_list = os.listdir(data_dir)[0:700]
primus = CTC_PriMuS(data_dir, corpus_list, dict_path, True, val_split=0.1)
params = default_model_params(img_height, primus.vocabulary_size)

# model
model = CTC_CNN(img_height, primus.vocabulary_size).to(device)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# loss
loss_func = torch.nn.CTCLoss()
loss_log = []

# loop
start_time = time.time()
for epoch in range(num_epochs):
    print(f'Epoch {epoch}/{num_epochs - 1}')
    print('-' * 10)

    epoch_time = time.time()

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            val_i = 0
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        while True:
            if phase == 'train':
                batch = primus.nextBatch(params)
                #print(primus.current_idx)
                inputs = torch.tensor(batch['inputs']).to(device)
                targets = torch.tensor(batch['targets']).to(device)
            else:
                batch = primus.getValidation(params)
                inputs = torch.tensor(batch['inputs'][val_i:val_i+params['batch_size']]).to(device)
                targets = torch.tensor(batch['targets'][val_i:val_i+params['batch_size']]).to(device)

            inputs = inputs.view(inputs.shape[0], inputs.shape[3], inputs.shape[1], inputs.shape[2])

            # zero the parameter gradients
            optimizer.zero_grad()
                
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                outputs_transposed = outputs.view(outputs.shape[1], outputs.shape[0], outputs.shape[2])
                output_lengths = [len(x) for x in outputs]
                target_lenghts = [len(x) for x in targets]

                #print(torch.argmax(outputs[5], dim=1))
                #print(targets[5])

                decoded_out = [decode(torch.argmax(out, dim=1)) for out in outputs]
                
                decoded_target = [decode(target) for target in targets]
                
                for out, target in zip(decoded_out, decoded_target):
                    if len(out) == len(target):
                        for o, t in zip(out, target):
                            if o != t:
                                break
                        else:
                            running_corrects += 1

                loss = loss_func(outputs_transposed, targets, output_lengths, target_lenghts)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics TODO: pitaj
                running_loss += loss.item() * inputs.size(0)
                #running_corrects += torch.sum(preds == labels.data).item()
            if phase == 'train':
                if primus.current_idx == 0:
                    break
            else:
                val_i += params['batch_size']
                if val_i >= len(primus.validation_list):
                    break

        if phase == 'train':
            epoch_loss = running_loss / len(primus.training_list)
            loss_log.append([epoch_loss])
        else:
            epoch_loss = running_loss / len(primus.validation_list)
            loss_log[-1].append(epoch_loss)
        #epoch_acc = float(running_corrects) / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f}')#' Acc: {epoch_acc:.4f}')
        if phase == 'train':
            print(f'{phase} Acc: {running_corrects/len(primus.training_list):.4f}')
        else:
            print(f'{phase} Acc: {running_corrects/len(primus.validation_list):.4f}')

            
        #metrics[phase+"_loss"].append(epoch_loss)
        #metrics[phase+"_acc"].append(epoch_acc)
        '''    
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())'''

        print()
    scheduler.step()
    torch.save(model, './model')
    print(time.time() - epoch_time)

time_elapsed = time.time() - start_time
print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
print(loss_log)
#print('Bestval Acc: {best_acc:4f}')