import torch
from orm_model import CTC_CNN, default_model_params
from orm_dataset import CTC_PriMuS
import os
import time
import cv2
from PIL import Image
import matplotlib as plt

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
train_loss=val_loss=[]
img_height = 128
learning_rate = 0.04
num_epochs = 500
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

corpus_list = os.listdir(data_dir)[0:700]
primus = CTC_PriMuS(data_dir, corpus_list, dict_path, True, val_split=0.1)
params = default_model_params(img_height, primus.vocabulary_size)
#print(primus.nextBatch(["targets"])[0][0][50:100])

# model
model = CTC_CNN(img_height, primus.vocabulary_size).to(device)
# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
# loss
loss_func = torch.nn.CTCLoss()
loss_log = []

#batch = primus.nextBatch(params)
#inputs = torch.tensor(batch['inputs']).to(device)
#targets = torch.tensor(batch['targets']).to(device)
#inputs=torch.transpose(inputs,1,2)
   
#inputs=torch.transpose(inputs,1,3)
            
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
                 #img = Image.fromarray(batch['inputs'][0][:][:][:]*255,'RGB')
           
                  #img.show()
            
                 inputs = torch.tensor(batch['inputs']).to(device)
                 targets = torch.tensor(batch['targets']).to(device)
                #input_image=cv2.imread("C:\\Users\\psiml\\Downloads\\MusicDetector\\package_aa\\000051650-1_1_1\\000051650-1_1_1.png")
                #inputs=torch.tensor([input_image]).to(device)
                #output_line="clef-G2	keySignature-EbM	timeSignature-3/4	note-Bb5_quarter	note-Eb5_eighth	note-Bb5_eighth	note-C6_eighth	note-Bb5_eighth	barline	note-Ab5_eighth	note-Ab5_eighth	rest-sixteenth	note-Ab5_sixteenth	note-G5_sixteenth	note-Ab5_sixteenth	note-Bb5_sixteenth	note-Ab5_sixteenth	note-G5_sixteenth	note-Ab5_sixteenth	barline	"
                #targets =  torch.tensor([output_line]).to(device)
            else:
                batch = primus.getValidation(params)
                inputs = torch.tensor(batch['inputs'][val_i:val_i+params['batch_size']]).to(device)
                targets = torch.tensor(batch['targets'][val_i:val_i+params['batch_size']]).to(device)

            #inputs = inputs.view(inputs.shape[0], inputs.shape[3], inputs.shape[1], inputs.shape[2])
            inputs=torch.transpose(inputs,1,2)
            #print(inputs.shape)
            inputs=torch.transpose(inputs,1,3)
            #print(inputs.shape)


            # zero the parameter gradients
            optimizer.zero_grad()
                
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                outputs_transposed = outputs.permute(1,0,2)
                
                output_lengths = [len(x) for x in outputs]
                target_lenghts = [len(x) for x in targets]
                #print(output_lengths)
                #print(target_lenghts)
                #print(torch.argmax(outputs[5], dim=1))
                #print(targets[5])

                decoded_out = [torch.argmax(out, dim=1) for out in outputs]
                
                decoded_target = [target for target in targets]
                
                correct = 0
                total = 0

                for out, target in zip(decoded_out, decoded_target):
                    total += len(target)
                    for o, t in zip(out, target):
                        if o == t:
                            correct += 1
                
                print(decoded_out[0])
                print(decoded_target[0])
                #print(outputs.size)

                running_corrects = correct / total

                loss = loss_func(outputs_transposed, targets, output_lengths, target_lenghts)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    #print(model.conv1.weight.grad)
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
            epoch_loss = running_loss/ len(primus.training_list)
            #loss_log.append([epoch_loss])
            train_loss.append(epoch_loss)
        else:
            epoch_loss = running_loss / len(primus.validation_list)
            #loss_log[-1].append(epoch_loss)
            val_loss.append(epoch_loss)
        #epoch_acc = float(running_corrects) / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f}')#' Acc: {epoch_acc:.4f}')
        if phase == 'train':
            print(f'{phase} Acc: {running_corrects/16:.4f}')
        else:
            print(f'{phase} Acc: {running_corrects/16:.4f}')

            
        #metrics[phase+"_loss"].append(epoch_loss)
        #metrics[phase+"_acc"].append(epoch_acc)
        '''    
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())'''

        print()
    #scheduler.step()
    torch.save(model, './model1')
    print(time.time() - epoch_time)

time_elapsed = time.time() - start_time
print(f'Training complete in {(time_elapsed // 60):.0f}m {time_elapsed % 60:.0f}s')
plt.plot(train_loss,np.linspace(0,len(train_loss),len(train_loss)),color='r')
plt.show()
plt.plot(val_loss,np.linspace(0,len(val_loss),len(val_loss)),color='g')
plt.show()
#print(loss_log)
#print('Bestval Acc: {best_acc:4f}')