import torch
from transformers import AutoModelForMaskedLM
import torch.optim as optim
from model.training import DistillationTrainer
from model.models import DistilBertConfig, DistilBertModel


student_config = DistilBertConfig()
student = DistilBertModel(student_config)
teacher = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-uncased")
optimizer = optim.AdamW(params=student.parameters())
trainer = DistillationTrainer(teacher, student, optimizer)

num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}")
    for batch in dataloader:
        loss_dict = trainer.train_step(batch, device)
        print(loss_dict)
