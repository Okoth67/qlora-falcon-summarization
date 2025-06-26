# QLoRA Fine-Tuning: Falcon 1B for Text Summarization 🦅📄

This project fine-tunes the 1.3B parameter [`tiiuae/falcon-rw-1b`](https://huggingface.co/tiiuae/falcon-rw-1b) language model using **QLoRA (Quantized Low-Rank Adaptation)** to perform **abstractive summarization** on the CNN/DailyMail dataset — all in **4-bit precision** on a single GPU using Google Colab.

---

## 📌 Project Highlights

- ✅ Uses **QLoRA** (8→4-bit quantization with LoRA adapters)
- ✅ Fine-tuned on **CNN/DailyMail summarization dataset**
- ✅ Efficient: ~1% trainable parameters only
- ✅ Lightweight: runs on free Colab GPU (T4 ~15GB) and Laptop, 16GB RAM
- ✅ Reproducible training pipeline with 🤗 Transformers Trainer

---

## 🧠 What is QLoRA?

QLoRA = **Quantization (4-bit)** + **LoRA adapters**

- Reduces memory usage by quantizing pretrained model weights to 4-bit using [`bitsandbytes`](https://github.com/TimDettmers/bitsandbytes)
- Adds **trainable LoRA adapters** (tiny weight matrices) while keeping the base model frozen
- Perfect for fine-tuning large models with limited compute (like on Laptop/Colab)

---

## 📊 Dataset

- **Name**: `cnn_dailymail` (via 🤗 Datasets)
- **Task**: Summarize news articles
- **Split used**: `train[:1000]` (for quick experimentation)

---

