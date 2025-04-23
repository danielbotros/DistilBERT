import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

################################################################################
############################# Data And Utils ###################################
################################################################################


class MaskedLMDataset(Dataset):
    """
    Example usage:
    # texts is a List[str]
    dataset = MaskedLMDataset(texts, max_length=max_len)

    collate_fn = DataCollatorForLanguageModeling(
      tokenizer=tokenizer,
      mlm=True,
      mlm_probability=mlm_prob
    )

    dataloader = DataLoader(
      dataset,
      batch_size=n,
      shuffle=True,
      collate_fn=collate_fn
    )
    """

    def __init__(self, texts, max_length=128):
        """
        texts: List of raw strings (one per example)
        max_length: Max sequence length for truncation
        """
        self.texts = texts
        self.max_length = max_length
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=False,
            return_tensors=None
        )

        return {
            "input_ids": encoding["input_ids"]
        }


def copy_every_other_layer(student, teacher):
    """
    Copy every other teacher layer into the student.
    Assumes student has half as many encoder layers as teacher.
    !!! Call after creating DistilBERT model object !!!
    """
    teacher_layers = teacher.bert.encoder.layer
    student_layers = student.distilbert.encoder_layers

    for i, layer in enumerate(student_layers):
        teacher_layer = teacher_layers[i * 2]
        layer.load_state_dict(teacher_layer.state_dict(), strict=False)


def process_batch(bert_model, data, criterion, device, val=False):
    """
    Inputs:
    data: The data in the batch to process.
    criterion: The loss function.
    val: True if processing a batch from the validation or test set.
         False if processing a batching from the training set.

    Outputs:
    Tuple of (outputs, losses)
        outputs: a dictionary containing the model outputs ('out') and predicted labels ('preds')
        metrics: a dictionary containing the model loss over the batch ('loss') and during validation (val = True),
                 the total number of examples in the batch ('batch_size') and the total number of examples whose
                 label the model predicted correctly ('num_correct')
    """

    outputs, metrics = dict(), dict()

    input_ids = data['source_ids'].to(device)
    attention_mask = data['source_mask'].to(device)
    labels = data['label'].to(device)

    model_output = bert_model(
        input_ids=input_ids, attention_mask=attention_mask).logits

    preds = torch.argmax(model_output, dim=1)

    outputs['out'] = model_output
    outputs['preds'] = preds

    loss = criterion(model_output, labels)
    metrics['loss'] = loss

    if val:
        metrics['batch_size'] = labels.size(0)
        metrics['num_correct'] = torch.sum(preds == labels)

    return outputs, metrics


def val_bert(model, val_loader, criterion, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.

    Outputs:
    Tuple of (validation loss, validation accuracy)
    """
    val_running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):

            _, batch_metrics = process_batch(
                model, data, criterion, device,  val=True)

            val_running_loss += batch_metrics['loss'].cpu().item()
            correct += batch_metrics['num_correct']
            total += batch_metrics['batch_size']

    return val_running_loss, (correct / total).item()


def train_bert(model, train_loader, val_loader, criterion, epochs, optim, lr_scheduler, device):
    """
    Inputs:
    model (torch.nn.Module): The deep learning model to be trained.
    val_data_loader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    criterion (torch.nn.Module): Loss function to compute the training loss.
    epochs: Number of epochs to train.
    optim: The optimizer for training.
    lr_scheduler: Learning rate scheduler for training.

    Outputs:
    Tuple of (train_loss_arr, val_loss_arr, val_acc_arr), an array of the training and validation
    losses and validation accuracy at each epoch
    """
    train_loss_arr = []
    val_loss_arr = []
    val_acc_arr = []
    running_loss = 0.0

    for epoch in range(epochs):
        running_loss = 0.0

        for batch_idx, data in enumerate(train_loader):

            _, metrics = process_batch(model, data, criterion, device)

            loss = metrics['loss'].cpu().item()

            optim.zero_grad()
            metrics['loss'].backward()
            optim.step()

            running_loss += loss

        val_running_loss, val_acc = val_bert(
            model, val_loader, criterion, device)
        train_loss_arr.append(running_loss)
        val_loss_arr.append(val_running_loss)
        val_acc_arr.append(val_acc)

        print("epoch:", epoch+1, "training loss:", round(running_loss, 3), 'validation loss:',
              round(val_running_loss, 3), 'validation accuracy:', round(val_acc*100, 2))

        lr_scheduler.step()

    return train_loss_arr, val_loss_arr, val_acc_arr
