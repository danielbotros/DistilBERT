import torch
import torch.nn.functional as F
from torch.nn import KLDivLoss, CosineEmbeddingLoss

################################################################################
############################ Training Class ####################################
################################################################################

# TODO: Tune hyperparameters (temperature, alpha, beta, gamma)
class DistillationTrainer:
    def __init__(self, teacher, student, optimizer, temperature=1.0, alpha=1.0, beta=1.0, gamma=1.0):
        self.teacher = teacher.eval()
        self.student = student.train()
        self.optimizer = optimizer

        self.kl_loss_fn = KLDivLoss(reduction="batchmean") # TODO: Verify batchmean
        self.cosine_loss_fn = CosineEmbeddingLoss()
        self.temp = temperature
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def distill_logits_loss(self, teacher_logits, student_logits):
        # (B, SEQ_LEN, VOCAB_SIZE)
        # Perform sotmax over dim=-1 so that we are doing it over VOCAB_SIZE 
        # dimension giving us a probability distribution for all vocabulary 
        # token for each token in our batch (B x SEQ_LEN) 
        t_probs = F.softmax(teacher_logits / self.temp, dim=-1)
        s_logp = F.log_softmax(student_logits / self.temp, dim=-1)
        return self.kl_loss_fn(s_logp, t_probs) * (self.temp ** 2)

    def distill_hidden_loss(self, teacher_hs, student_hs):
        losses = []
        teacher_layers = teacher_hs[1::2]  # take every 2nd layer starting from index 1
        student_layers = student_hs[1:]    # skip the student input embedding layer

        for t, s in zip(teacher_layers, student_layers):
            # flatten hidden state in vectors of HIDDEN_STATE_SIZE for every token
            # in the batch
            # (B, SEQ_LEN, HIDDEN_STATE_SIZE) -> (B x SEQ_LEN, HIDDEN_STATE_SIZE)
            t_flat = torch.flatten(t, start_dim=0, end_dim=1)
            s_flat = torch.flatten(s, start_dim=0, end_dim=1)
            # Create a tensor of ones (B x SEQ_LEN) as the target cosine similarity 
            # between teacher and student hidden states
            target = torch.ones(t_flat.size(0)).to(t.device)
            losses.append(self.cosine_loss_fn(s_flat, t_flat, target))
        return torch.stack(losses).mean()

    def train_step(self, batch, device):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        with torch.no_grad():
            t_out = self.teacher(input_ids=input_ids, attention_mask=attention_mask)
            t_logits = t_out.logits
            t_hs = t_out.hidden_states

        s_out = self.student(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        s_logits = s_out["logits"]
        s_hs = s_out["hidden_states"]
        mlm_loss = s_out["loss"]

        kd_loss = self.distill_logits_loss(t_logits, s_logits)
        cos_loss = self.distill_hidden_loss(t_hs, s_hs)

        total_loss = self.alpha * mlm_loss + self.beta * kd_loss + self.gamma * cos_loss
        total_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return {
            "total_loss": total_loss.item(),
            "mlm_loss": mlm_loss.item(),
            "kd_loss": kd_loss.item(),
            "cos_loss": cos_loss.item()
        }
