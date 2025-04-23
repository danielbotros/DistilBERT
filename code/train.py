import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DistilBertConfig,
    DistilBertModel
)
from datasets import load_dataset
from model.training import DistillationTrainer
from model.models import DistilBertConfig, DistilBertModel
from utils import copy_every_other_layer, MaskedLMDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained BERT
print("Loading teacher BERT model...")
teacher = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")

selected_layer_indices = list(range(0, 12, 2))  # [0, 2, 4, 6, 8, 10]

print(
    f"Selected BERT layers for DistilBERT student: {selected_layer_indices}")

# Create DistilBERT configuration
student_config = DistilBertConfig(
    vocab_size=teacher.config.vocab_size,
    max_position_embeddings=teacher.config.max_position_embeddings,
    sinusoidal_pos_embds=False,
    n_layers=len(selected_layer_indices),
    n_heads=teacher.config.num_attention_heads,
    dim=teacher.config.hidden_size,
    hidden_dim=teacher.config.intermediate_size,
    dropout=teacher.config.hidden_dropout_prob,
    attention_dropout=teacher.config.attention_probs_dropout_prob
)

# Initialize DistilBERT student model
student = DistilBertModel(student_config)


# Copy embeddings
student.embeddings.word_embeddings.weight.data = teacher.embeddings.word_embeddings.weight.data.clone()
student.embeddings.position_embeddings.weight.data[
    :student_config.max_position_embeddings] = teacher.embeddings.position_embeddings.weight.data.clone()
student.embeddings.LayerNorm.weight.data = teacher.embeddings.LayerNorm.weight.data.clone()
student.embeddings.LayerNorm.bias.data = teacher.embeddings.LayerNorm.bias.data.clone()

# Copy selected BERT encoder layers to DistilBERT transformer layers
print("Copying encoder layers...")
copy_every_other_layer(student, teacher)

print("Finished initializing student.")

teacher.to(device)
student.to(device)

# since we don't want to update the teacher's weights
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
raw_ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
train_texts = raw_ds["text"]
train_ds = MaskedLMDataset(train_texts)
collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

dataloader = DataLoader(train_ds, batch_size=8,
                        shuffle=True, collate_fn=collator)

for batch in dataloader:
    print(batch)  # This will print a batch
    break


# === Hyperparameters ===
alpha = 0.5
beta = 0.5
gamma = 1
temperature = 2.0

optimizer = optim.AdamW(params=student.parameters())
trainer = DistillationTrainer(teacher, student, optimizer)

num_epochs = 3

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    for batch in dataloader:
        loss_dict = trainer.train_step(batch, device)
        print(loss_dict)


student.save_pretrained("distilbert-mlm-distilled")
tokenizer.save_pretrained("distilbert-mlm-distilled")
