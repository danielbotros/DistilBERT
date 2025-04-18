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