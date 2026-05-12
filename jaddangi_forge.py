"""
JADDANGI ALFA FORGE v1.0 (LoRA Training Infrastructure)
Designed by: Kurumalla Venkataramana | Jaddangi IT & AI Consultancy
Description: Custom Parameter-Efficient Fine-Tuning (PEFT) and Instruction Masking.
Capabilities: Trains massive models on low-VRAM GPUs by freezing 99.8% of the network.
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ====================================================================
# 1. LORA PHYSICS (Low-Rank Matrix Decomposition)
# ====================================================================
class JaddangiLoRALinear(nn.Module):
    """Surgical adapter that trains parallel tiny matrices instead of the massive base brain."""
    def __init__(self, base_linear, rank=8, alpha=16):
        super().__init__()
        # 1. Freeze the original knowledge completely
        self.base_layer = base_linear
        self.base_layer.weight.requires_grad = False
        if self.base_layer.bias is not None:
            self.base_layer.bias.requires_grad = False
            
        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features
        
        # 2. Create the tiny trainable LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        self.scaling = alpha / rank
        
        # 3. Kaiming init for A, Zero init for B (Starts with 0 impact)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # Base knowledge (frozen) + New knowledge (trainable)
        base_out = self.base_layer(x)
        lora_out = (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return base_out + lora_out

def inject_lora_into_jaddangi(model, rank=8, alpha=16):
    """Surgically replaces Attention projections with LoRA adapters across all 24 layers."""
    print("💉 Injecting LoRA adapters into Attention layers...")
    injected_count = 0
    
    for layer in model.layers:
        attn = layer.self_attn
        # Replace Query and Value projections
        attn.q_proj = JaddangiLoRALinear(attn.q_proj, rank=rank, alpha=alpha)
        attn.v_proj = JaddangiLoRALinear(attn.v_proj, rank=rank, alpha=alpha)
        injected_count += 2
        
    print(f"✅ Successfully injected {injected_count} LoRA modules.")
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    print(f"🧊 Frozen Base Parameters: {frozen:,}")
    print(f"🔥 Trainable LoRA Parameters: {trainable:,} ({(trainable/(trainable+frozen))*100:.2f}%)")
    return model

# ====================================================================
# 2. INSTRUCTION DATASET FORMATTER (-100 Masking)
# ====================================================================
class JaddangiInstructDataset(Dataset):
    """Formats raw text into tensors and masks the prompt so the AI only learns the answer."""
    def __init__(self, data, tokenizer):
        self.tokenizer = tokenizer
        self.inputs = []
        self.labels = []
        
        system_prompt = "You are Jaddangi-Alfa, a precise AI Agent. You MUST use tools to answer.\n"
        print(f"📦 Formatting {len(data)} custom instructions...")
        
        for item in data:
            prompt_text = f"{system_prompt}User: {item['user']}\nJaddangi: "
            target_text = item['jaddangi'] + "<|endoftext|>" 
            
            prompt_ids = tokenizer.encode(prompt_text)
            target_ids = tokenizer.encode(target_text)
            
            full_ids = prompt_ids + target_ids
            
            # -100 tells PyTorch's CrossEntropyLoss to IGNORE the prompt during training
            label_ids = [-100] * len(prompt_ids) + target_ids
            
            self.inputs.append(torch.tensor(full_ids, dtype=torch.long))
            self.labels.append(torch.tensor(label_ids, dtype=torch.long))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]

def collate_fn(batch, pad_token_id=0):
    inputs, labels = zip(*batch)
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=pad_token_id)
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
    return inputs_padded, labels_padded

# ====================================================================
# 3. THE TRAINING PIPELINE (The Forge)
# ====================================================================
def forge_jaddangi_agent(model, dataset, epochs=3, lr=5e-5, batch_size=4):
    """Executes the GPU training loop to rewire the model's tool-trigger instincts."""
    print("\n" + "="*50)
    print("🔥 IGNITING THE JADDANGI INSTRUCTION FORGE")
    print("="*50)
    
    device = next(model.parameters()).device
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda b: collate_fn(b, pad_token_id=0) # Update with actual pad token if available
    )
    
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        
        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x)
            logits = outputs.logits
            
            # Shift alignment: Compare what it predicted (logits) with what should come next (labels)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = y[..., 1:].contiguous()
            
            # F.cross_entropy automatically ignores the -100 tokens
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), 
                shift_labels.view(-1), 
                ignore_index=-100
            )
            
            # Backward pass (Calculate Gradients)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if step % 10 == 0:
                print(f"   Epoch {epoch+1} | Step {step} | Loss: {loss.item():.4f}")
                
        avg_loss = total_loss / len(dataloader)
        print(f"🏁 Epoch {epoch+1} Complete | Avg Loss: {avg_loss:.4f} | Time: {time.time()-start_time:.1f}s")
        
    print("\n✅ JADDANGI-ALFA AGENT FORGE COMPLETE!")
    return model
