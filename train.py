import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import json
from transformers import BlipProcessor, BlipForConditionalGeneration
from torch.optim import AdamW

device = "cuda" if torch.cuda.is_available() else "cpu"

# Dataset đọc từ JSON
class CaptionDataset(Dataset):
    def __init__(self, json_file, processor):
        with open(json_file, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.processor = processor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        image = Image.open(item["image"]).convert("RGB")
        caption = item["caption"]
        inputs = self.processor(images=image, text=caption, return_tensors="pt",
                                padding="max_length", truncation=True)
        # thêm labels để mô hình tính loss
        inputs["labels"] = inputs["input_ids"]
        return {k: v.squeeze() for k, v in inputs.items()}

# Load mô hình gốc BLIP
model_name = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(model_name)
model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)

# Dataset và DataLoader
dataset = CaptionDataset("trainDB.json", processor)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

# Optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Huấn luyện
model.train()
for epoch in range(2):  # chỉ 2 epoch cho tập nhỏ
    for batch in dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        if loss is not None:
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    print(f"Epoch {epoch+1} hoàn thành, loss={loss.item():.4f}")

# Lưu mô hình mới
model.save_pretrained("blip_vi_model")
processor.save_pretrained("blip_vi_model")
