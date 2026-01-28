import torch
import torch.nn as nn
from transformers import WavLMModel

class WaveLmStutterClassification(nn.Module):
    def  __init__(self,num_labels=5,freeze_base=True,unfreeze_last_n_layers=0):
        """
        Args:
            num_labels: Number of output classes (stutter types).
        freeze_base: If True, freeze WavLM weights (only train the classifier).
        """
        super().__init__()
        #Load pre-trained WavLM
        self.wavlm=WavLMModel.from_pretrained("microsoft/wavlm-base")

        #unfreeze last encoder layer and attaching a classifier head
        if freeze_base:
            #First,freeze all parameters
            for param in self.wavlm.parameters():
                param.requires_grad=False
            #Now unfreeze last n encoder layers(last layer in our case)
            if unfreeze_last_n_layers>0:
                #wavelm has 12 encoder layers(index 0 to 11)
                total_layers=12
                start_unfreeze=total_layers-unfreeze_last_n_layers
                for i,layer in enumerate(self.wavlm.encoder.layers):
                    if i>=start_unfreeze:
                        for param in layer.parameters():
                            param.requires_grad=True #total 7.1M parameters to train
                print(f"Unfroze last {unfreeze_last_n_layers} encoder layers of WavLM")
        
        #Classifier head
        self.hidden_size=self.wavlm.config.hidden_size #768 for wavlm-base
        self.classifier=nn.Linear(self.hidden_size,num_labels) #num_labels output neurons, in our case 5
        #total parameters to train is 7.1M + (768*5 + 5) = ~7.1M
    
    def forward(self, waveform, attention_mask=None):
        """
        Args:
            waveform: (batch, samples) raw audio
            attention_mask: (batch, samples) optional mask, 1 for real audio, 0 for padding
        Returns:
            logits: (batch, num_labels) raw scores (apply sigmoid for probabilities)
        """
        # Pass through WavLM
        outputs = self.wavlm(waveform, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, time, 768)
        
        # Mean pooling across time dimension
        # If we have attention_mask, we should only average over non-padded positions
        if attention_mask is not None:
            # WavLM downsamples the audio, so we need to downsample the mask too
            # The downsampling factor is approximately 320 (for 16kHz audio)
            # But we can just use the hidden_states length
            mask_length = hidden_states.size(1)
            # Downsample attention_mask to match hidden_states length
            attention_mask_downsampled = torch.nn.functional.interpolate(
                attention_mask.unsqueeze(1).float(),  # (batch, 1, samples)
                size=mask_length,
                mode='nearest'
            ).squeeze(1)  # (batch, mask_length)
            
            # Expand mask for broadcasting: (batch, time, 1)
            mask_expanded = attention_mask_downsampled.unsqueeze(-1)
            
            # Apply mask and compute mean only over non-padded positions
            hidden_states_masked = hidden_states * mask_expanded
            sum_hidden = hidden_states_masked.sum(dim=1)  # (batch, 768)
            count = mask_expanded.sum(dim=1)  # (batch, 1)
            pooled = sum_hidden / count.clamp(min=1)  # Avoid division by zero
        else:
            # Simple mean pooling if no mask provided
            pooled = hidden_states.mean(dim=1)  # (batch, 768)
        
        # Get predictions
        logits = self.classifier(pooled)  # (batch, num_labels)
        return logits
         
        

