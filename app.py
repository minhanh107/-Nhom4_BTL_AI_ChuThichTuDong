import torch
from PIL import Image
import gradio as gr
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import MarianMTModel, MarianTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load mô hình BLIP đã fine-tune
model = BlipForConditionalGeneration.from_pretrained("blip_vi_model").to(device)
processor = BlipProcessor.from_pretrained("blip_vi_model")

# Load mô hình dịch Anh → Việt
mt_model_name = "Helsinki-NLP/opus-mt-en-vi"
mt_tokenizer = MarianTokenizer.from_pretrained(mt_model_name)
mt_model = MarianMTModel.from_pretrained(mt_model_name).to(device)

def translate_en_to_vi(text):
    inputs = mt_tokenizer(text, return_tensors="pt", padding=True).to(device)
    translated = mt_model.generate(
        **inputs,
        num_beams=8,              # tăng số beam để dịch sát nghĩa
        max_length=128,           # giới hạn độ dài hợp lý
        repetition_penalty=1.3,   # tránh lặp từ
        length_penalty=1.0,       # cân bằng câu ngắn/dài
        early_stopping=True,
        no_repeat_ngram_size=2    # tránh lặp cụm từ
    )
    return mt_tokenizer.decode(translated[0], skip_special_tokens=True)

def caption_image(image):
    inputs = processor(images=image, return_tensors="pt").to(device)
    output_ids = model.generate(
        **inputs,
        num_beams=5,
        max_new_tokens=30,
        repetition_penalty=1.25,
        early_stopping=True
    )
    caption_en = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
    caption_vi = translate_en_to_vi(caption_en)
    return caption_en, caption_vi

with gr.Blocks() as demo:
    gr.Markdown("## 🖼️ ỨNG DỤNG TẠO TỰ ĐỘNG CHÚ THÍCH CHO HÌNH ẢNH")

    # Hướng dẫn thao tác
    gr.Markdown("""
### 📌 Hướng dẫn sử dụng:
1. Nhấn **Upload ảnh** để chọn ảnh từ máy tính.
2. Bấm **Submit** để hệ thống sinh chú thích tự động cho ảnh.
3. Xem kết quả ở ô bên phải: caption tiếng Anh và bản dịch tiếng Việt.
4. Bấm **Clear** để xóa kết quả, bấm x để xóa ảnh hiện tại và thử lại với ảnh khác.
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload ảnh")
            submit_btn = gr.Button("Submit", variant="primary")
            clear_btn = gr.Button("Clear", variant="stop")
        with gr.Column():
            output_en = gr.Textbox(label="Caption tiếng Anh", lines=4, interactive=False)
            output_vi = gr.Textbox(label="Caption tiếng Việt", lines=6, interactive=False)

    submit_btn.click(fn=caption_image, inputs=image_input, outputs=[output_en, output_vi])
    clear_btn.click(fn=lambda: ("", ""), inputs=None, outputs=[output_en, output_vi])

if __name__ == "__main__":
    demo.launch()
