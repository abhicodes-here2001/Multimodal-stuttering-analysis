#importing pytorch modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split

#standard libraries
import os
import sys
#add project root to python's search path
#this lets us import from training/and models/folders
project_root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from training.Dataset import StutterDataset
from models.WaveLm_model import WaveLmStutterClassification

#Collate function to pad variable-length audio in batch
def collate_fn(batch):
    """ Combines multiple audio samples into a batch, padding as necessary."""
    #seprate waveforms and labels
    waveforms,labels=zip(*batch)

    #find the logest waveform and create masks
    max_len=max(w.size(0) for w in waveforms)

    #pad each waveform and create masks
    padded_waveforms=[]
    attention_masks=[]

    for w in waveforms:
        #how much padding is needed?
        pad_len=max_len-w.size(0)

        #create mask: 1= real audio , 0 =padding
        mask=torch.cat([torch.ones(w.size(0)),torch.zeros(pad_len)])
        attention_masks.append(mask)

        #pad the waveform
        if pad_len>0:
            w=torch.nn.functional.pad(w,(0,pad_len))
        padded_waveforms.append(w)

    #stack into batch tensors
    waveforms_batch=torch.stack(padded_waveforms)
    labels_batch=torch.stack(list(labels))
    masks_batch=torch.stack(attention_masks)
    return waveforms_batch,labels_batch,masks_batch

#Train function
def train():
    CSV_PATH="data/SEP-28k_labels.csv"
    AUDIO_DIR="data/clips"
    LABEL_COLUMNS=['Prolongation','Block','SoundRep','WordRep','Interjection']
    BATCH_SIZE=16 #try 4,8,16,32 based on your GPU memory
    EPOCHS=15 #Start small for testing, increase later
    LEARNING_RATE=5e-5 #Try : 1e-3,1e-5

    #Use GPU if available
    if torch.backends.mps.is_available():
        device=torch.device("mps") #for APPLE
    else:
        device=torch.device("cpu")
        print("USING: CPU")
    
    #Createing Dataset
    dataset=StutterDataset(CSV_PATH,AUDIO_DIR,LABEL_COLUMNS)
    print(f"Total samples:{len(dataset)}")

    #split 80 percent train , 20 percent validation
    train_size=int(0.8*len(dataset))
    val_size=len(dataset)-train_size
    train_data,val_data=random_split(dataset,[train_size,val_size])
    print(f"Train: {len(train_data)}, Validation: {len(val_data)}")

    #Create dataloaders
    train_loader=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,collate_fn=collate_fn)
    #trainloader loads the data with batch size , stacks it up and apply padding by collate function and we also get masked tensors
    val_loader=DataLoader(val_data,batch_size=BATCH_SIZE,shuffle=False,collate_fn=collate_fn)
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    #Creating the model
    model=WaveLmStutterClassification(num_labels=len(LABEL_COLUMNS),freeze_base=True,unfreeze_last_n_layers=1)
    model.to(device) #moving to mps GPU

    #count parameters
    Trainable=sum(p.numel() for p in model.parameters() if p.requires_grad) #it gives total trainable parameters in our model
    total=sum(p.numel() for p in model.parameters())
    print(f"Model parameters: Trainable: {Trainable}, Total: {total}")

    #defining the loss function and optimizer
    loss_fn=nn.BCEWithLogitsLoss() #for binary classifcation , it uses sigmoid internally to calculate the logits
    optimizer=torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE) #optimizer is used not dropout because we are fine tuning the model, we cannot drop neuron

    #Training process starts here
    print("********Starting Training******** ")
    
    #Track best model for saving
    best_val_loss = float('inf')  # Start with infinity so first epoch is always "best"
    
    for epoch in range(EPOCHS):
        print(f"--- Epoch {epoch+1}/{EPOCHS} ---")

        #Training
        model.train() #Enable Training mode , it activates the training mode and apply dropout on internal layers so that our classifier is trained perfectly
        train_loss=0

        for batch_idx,(waveforms,labels,masks) in enumerate(train_loader):
            #move data to GPU- Apple Metal MPS
            waveforms=waveforms.to(device)
            labels=labels.to(device)
            masks=masks.to(device)

            #Forward pass
            logits=model(waveforms,masks)
            loss=loss_fn(logits,labels)

            #Backward pass and optimization
            optimizer.zero_grad() #this means we are clearing the previous gradients because pytorch accumulates the gradients, if we dont clear it , it will add up the gradients from previous batch
            loss.backward() #compute new gradients
            optimizer.step()#update the weights

            


            train_loss+=loss.item() #accumulate loss

            #Print progressin every 50 batches
            if batch_idx%50==0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        avg_train_loss=train_loss/len(train_loader)

        #Validation after each epoch
        model.eval()#Disable training mode
        val_loss=0
        correct=0
        total=0

        with torch.no_grad(): #no need to compute gradients during validation
            for waveforms,labels,masks in val_loader:
                waveforms=waveforms.to(device)
                labels=labels.to(device)
                masks=masks.to(device)

                logits=model(waveforms,masks)
                loss=loss_fn(logits,labels)
                val_loss+=loss.item()

                #Calculate accuracy
                preds=(torch.sigmoid(logits)>0.5).float() #apply sigmoid and threshold at 0.5
                correct+=(preds==labels).sum().item()
                total+=labels.numel()
        
        avg_val_loss=val_loss/len(val_loader)
        val_accuracy=correct/total*100

        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")
        
        #Save the best model (only when validation loss improves)
        os.makedirs("checkpoints", exist_ok=True)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/wavlm_stutter_classification_best.pth")
            print(f"✓ New best model saved! (Val Loss: {avg_val_loss:.4f})")
        else:
            print(f"✗ Val Loss did not improve (Best: {best_val_loss:.4f})")

    #Save final model (last epoch, regardless of performance)
    torch.save(model.state_dict(), "checkpoints/wavlm_stutter_classification_final.pth")
    print("Final model saved to checkpoints/wavlm_stutter_classification_final.pth")
    print(f"Best model (Val Loss: {best_val_loss:.4f}) saved to checkpoints/wavlm_stutter_classification_best.pth")
    print("********Training Completed********")


#entry point
if __name__=="__main__":
    train()

















