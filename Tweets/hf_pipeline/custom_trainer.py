import torch
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Calculate loss using class weights if provided
        #! note that we have to set the correct device, there could be an error during the train 
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights, device=model.device)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(device=model.device)

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss