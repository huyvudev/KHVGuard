import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel
import gradio as gr
import os

# --- CẤU HÌNH ---
MODEL_NAME = 'vinai/phobert-base-v2'
MAX_LEN = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'best_model.pth' 

# --- ĐỊNH NGHĨA MODEL ---
class PIGuardPT(nn.Module):
    def __init__(self, model_name, num_labels=2):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_attentions=True)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled)
        return logits

# --- LOAD MODEL & TOKENIZER ---
print(f"Đang chạy trên: {DEVICE}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
model = PIGuardPT(MODEL_NAME, num_labels=2)

if os.path.exists(MODEL_PATH):
    print(f"Đang load model từ {MODEL_PATH}...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    print("-> Đã load model thành công!")
else:
    print(f"!!! CẢNH BÁO: Không tìm thấy file {MODEL_PATH}. Hãy copy file .pth vào thư mục này.")

model.to(DEVICE)
model.eval()

# --- HÀM DỰ ĐOÁN ---
def predict_text(text):
    if not text: return ""
    enc = tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
    input_ids = enc['input_ids'].to(DEVICE)
    attention_mask = enc['attention_mask'].to(DEVICE)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.nn.functional.softmax(logits, dim=1)
        pred_label = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_label].item()

    if pred_label == 1:
        return f"⚠️ NGUY HIỂM (ATTACK) - {confidence:.1%}"
    else:
        return f"✅ AN TOÀN (BENIGN) - {confidence:.1%}"

# --- GIAO DIỆN ---
if __name__ == "__main__":
    gr.Interface(
        fn=predict_text,
        inputs=gr.Textbox(lines=3, placeholder="Nhập câu prompt..."),
        outputs="text",
        title="🛡️ KHVGuard Check",
        theme="soft"
    ).launch(inbrowser=True)