import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

# ==========================================
# 1. Configuration & Hyperparameters (ตาม PDF)
# ==========================================
# โครงสร้าง Layer ตามที่ระบุใน Section II.C [cite: 203]
INPUT_SIZE = 6      # 5 ADLs (Sitting, Standing, Walking, Running, Jumping) + 1 History of Falls [cite: 77, 209]
RBM1_UNITS = 256    # [cite: 165]
RBM2_UNITS = 200    # [cite: 167]
RBM3_UNITS = 150    # [cite: 168]
OUTPUT_CLASSES = 3  # Low, Moderate, High Risk [cite: 74]

# Hyperparameters สำหรับการ Train [cite: 165, 167, 168]
LEARNING_RATE_PRETRAIN = 0.01
PRETRAIN_EPOCHS = 10
FINETUNE_LR = 0.001 # ค่าทั่วไปสำหรับ Adam Optimizer
FINETUNE_EPOCHS = 50 # ปรับเพิ่มเพื่อให้ Model เรียนรู้ได้ดีขึ้น

# ==========================================
# 2. RBM Layer Definition (Unsupervised)
# ==========================================
class RBM(nn.Module):
    """
    Restricted Boltzmann Machine สำหรับการ Pre-training แบบ Unsupervised
    ใช้เทคนิค Contrastive Divergence (CD) ตามที่ระบุใน Section II [cite: 21]
    """
    def __init__(self, visible_units, hidden_units):
        super(RBM, self).__init__()
        self.W = nn.Parameter(torch.randn(visible_units, hidden_units) * 0.01)
        self.h_bias = nn.Parameter(torch.zeros(hidden_units))
        self.v_bias = nn.Parameter(torch.zeros(visible_units))

    def forward(self, v):
        # Forward pass เพื่อแปลง input เป็น hidden features
        p_h_given_v = torch.sigmoid(torch.mm(v, self.W) + self.h_bias)
        return p_h_given_v

    def sample_h(self, v):
        p_h_given_v = torch.sigmoid(torch.mm(v, self.W) + self.h_bias)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        p_v_given_h = torch.sigmoid(torch.mm(h, self.W.t()) + self.v_bias)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def contrastive_divergence(self, input_data, lr=0.01):
        # CD-1 training step
        v0 = input_data
        p_h0, h0 = self.sample_h(v0)
        
        p_v1, v1 = self.sample_v(h0)
        p_h1, h1 = self.sample_h(v1)

        # Update weights and biases
        positive_grad = torch.mm(v0.t(), p_h0)
        negative_grad = torch.mm(v1.t(), p_h1)

        self.W.data += lr * (positive_grad - negative_grad) / input_data.size(0)
        self.v_bias.data += lr * torch.mean(v0 - v1, dim=0)
        self.h_bias.data += lr * torch.mean(p_h0 - p_h1, dim=0)
        
        error = torch.mean((v0 - v1) ** 2)
        return error

# ==========================================
# 3. DBN Model Definition (Pre-train + Fine-tune)
# ==========================================
class FallRiskDBN(nn.Module):
    def __init__(self):
        super(FallRiskDBN, self).__init__()
        
        # Stacked RBM Layers [cite: 80]
        self.rbm1 = RBM(INPUT_SIZE, RBM1_UNITS)
        self.rbm2 = RBM(RBM1_UNITS, RBM2_UNITS)
        self.rbm3 = RBM(RBM2_UNITS, RBM3_UNITS)
        
        # Classifier Layer (Fine-tuning phase)
        # รับ input จาก RBM สุดท้าย (150 units) -> Output 3 classes
        self.classifier = nn.Linear(RBM3_UNITS, OUTPUT_CLASSES)
        
    def pretrain(self, dataloader):
        """
        Greedy Layer-by-Layer Pre-training [cite: 207]
        """
        print("--- Starting DBN Pre-training (Unsupervised) ---")
        
        # Train RBM 1
        print(f"Pre-training RBM 1 ({INPUT_SIZE} -> {RBM1_UNITS})")
        for epoch in range(PRETRAIN_EPOCHS):
            epoch_loss = 0
            for batch_idx, (data, _) in enumerate(dataloader):
                loss = self.rbm1.contrastive_divergence(data, lr=LEARNING_RATE_PRETRAIN)
                epoch_loss += loss.item()
            print(f"\tEpoch {epoch+1}/{PRETRAIN_EPOCHS} Loss: {epoch_loss/len(dataloader):.4f}")

        # Train RBM 2 (ใช้ Output จาก RBM 1 เป็น Input)
        print(f"Pre-training RBM 2 ({RBM1_UNITS} -> {RBM2_UNITS})")
        for epoch in range(PRETRAIN_EPOCHS):
            epoch_loss = 0
            for batch_idx, (data, _) in enumerate(dataloader):
                v_rbm2 = self.rbm1.forward(data) # Feed forward ผ่าน RBM 1
                loss = self.rbm2.contrastive_divergence(v_rbm2, lr=LEARNING_RATE_PRETRAIN)
                epoch_loss += loss.item()
            print(f"\tEpoch {epoch+1}/{PRETRAIN_EPOCHS} Loss: {epoch_loss/len(dataloader):.4f}")

        # Train RBM 3 (ใช้ Output จาก RBM 2 เป็น Input)
        print(f"Pre-training RBM 3 ({RBM2_UNITS} -> {RBM3_UNITS})")
        for epoch in range(PRETRAIN_EPOCHS):
            epoch_loss = 0
            for batch_idx, (data, _) in enumerate(dataloader):
                v_rbm2 = self.rbm1.forward(data)
                v_rbm3 = self.rbm2.forward(v_rbm2) # Feed forward ผ่าน RBM 2
                loss = self.rbm3.contrastive_divergence(v_rbm3, lr=LEARNING_RATE_PRETRAIN)
                epoch_loss += loss.item()
            print(f"\tEpoch {epoch+1}/{PRETRAIN_EPOCHS} Loss: {epoch_loss/len(dataloader):.4f}")

    def forward(self, x):
        # Forward pass สำหรับ Fine-tuning [cite: 81]
        x = self.rbm1.forward(x)
        x = self.rbm2.forward(x)
        x = self.rbm3.forward(x)
        x = self.classifier(x) # Linear layer ก่อนเข้า Softmax (CrossEntropyLoss จะทำ Softmax ให้เอง)
        return x

# ==========================================
# 4. Main Execution Pipeline
# ==========================================

# --- A. Mock Dataset (แทนที่ด้วยการโหลดไฟล์จริง) ---
# สร้างข้อมูลจำลองตาม Feature ใน PDF: Sitting, Standing, Walking, Running, Jumping, History
# และกำหนด Label แบบสุ่ม (0=Low, 1=Moderate, 2=High)
print("Generating Mock Data...")
data_size = 100
# Features: [Sit, Stand, Walk, Run, Jump, History_Flag]
X_raw = np.random.rand(data_size, 6) 
# Labels: 0, 1, 2
y_raw = np.random.randint(0, 3, size=(data_size))

# --- B. Data Preprocessing [cite: 157, 158] ---
# ใช้ Min-Max Normalization ตามสมการ (159) ใน PDF
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X_raw)

# แปลงเป็น PyTorch Tensor
X_tensor = torch.FloatTensor(X_normalized)
y_tensor = torch.LongTensor(y_raw)

# สร้าง DataLoader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# --- C. Initialize & Pre-train Model ---
model = FallRiskDBN()
model.pretrain(dataloader)

# --- D. Fine-Tuning (Supervised) [cite: 81, 150] ---
# ใช้ Adam Optimizer และ Cross Entropy Loss (สำหรับ Multi-class)
print("\n--- Starting Fine-tuning (Supervised) ---")
optimizer = optim.Adam(model.parameters(), lr=FINETUNE_LR)
criterion = nn.CrossEntropyLoss()

model.train()
for epoch in range(FINETUNE_EPOCHS):
    total_loss = 0
    correct = 0
    for data, target in dataloader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        
    if (epoch+1) % 10 == 0:
        acc = 100. * correct / data_size
        print(f"Fine-tune Epoch {epoch+1}/{FINETUNE_EPOCHS} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# ==========================================
# 5. Prediction & Risk Level Indicator [cite: 74, 246]
# ==========================================
def predict_fall_risk(input_features, model, scaler):
    """
    ฟังก์ชันสำหรับทำนายผลและแปลงเป็นข้อความ
    input_features: numpy array ขนาด (1, 6)
    """
    model.eval()
    # Normalize input ด้วย scaler ตัวเดิม
    input_norm = scaler.transform(input_features)
    input_tensor = torch.FloatTensor(input_norm)
    
    with torch.no_grad():
        logits = model(input_tensor)
        probs = F.softmax(logits, dim=1) # แปลงเป็นค่าความน่าจะเป็น
        predicted_class = torch.argmax(probs, dim=1).item()
    
    # Mapping ผลลัพธ์ตาม PDF [cite: 74, 91-94]
    risk_levels = {0: "Low Risk", 1: "Moderate Risk", 2: "High Risk"}
    
    return risk_levels[predicted_class], probs.numpy()

# --- ทดสอบการใช้งาน ---
print("\n--- Testing Prediction ---")
# ตัวอย่างข้อมูล: [Sitting, Standing, Walking, Running, Jumping, History=1(Yes)]
sample_input = np.array([[0.5, 0.2, 0.8, 0.1, 0.0, 1.0]]) 
result, probabilities = predict_fall_risk(sample_input, model, scaler)

print(f"Input Features: {sample_input}")
print(f"Predicted Risk Level: {result}")
print(f"Probabilities (Low, Mod, High): {probabilities}")