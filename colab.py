# -*- coding: utf-8 -*-
"""
🚀 Vintern Embedding API Server - Chạy trên Colab GPU

Copy toàn bộ file này vào Google Colab và chạy!

Hướng dẫn:
1. Mở Google Colab: https://colab.research.google.com/
2. Runtime → Change runtime type → GPU
3. Copy toàn bộ code này vào 1 cell
4. Chạy cell
5. Copy URL từ output và dán vào config.py (LOCAL)
"""

# ============================================================================
# BƯỚC 0: XÓA CACHE CŨ (NẾU CÓ)
# ============================================================================
import shutil
import os

cache_dir = "/root/.cache/huggingface/modules/transformers_modules/5CD-AI/Vintern-Embedding-1B"
if os.path.exists(cache_dir):
    print(f"🗑️ Xóa cache cũ...")
    shutil.rmtree(cache_dir, ignore_errors=True)
    print("✅ Đã xóa cache")

# ============================================================================
# BƯỚC 1: CÀI ĐẶT DEPENDENCIES
# ============================================================================
print("📦 Đang cài đặt packages...")
import subprocess
import sys

packages = [
    "transformers==4.48.0",
    "torch",
    "torchvision",
    "Pillow",
    "flask",
    "flask-cors",
    "pyngrok",
    "timm",
    "einops",
    "decord",
    "ninja",  # Required for flash_attn
    "packaging"  # Required for flash_attn
]

# Cài flash_attn riêng (cần compile)
print("⚡ Đang cài flash-attn (mất ~2-3 phút)...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"])
    print("✅ flash-attn đã được cài đặt!")
except Exception as e:
    print(f"⚠️ Không thể cài flash-attn: {e}")
    print("Model vẫn có thể chạy nhưng chậm hơn một chút")

for package in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

print("✅ Packages đã được cài đặt!")

# ============================================================================
# BƯỚC 2: IMPORT LIBRARIES
# ============================================================================
print("📚 Đang import libraries...")

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor
import numpy as np
import base64
import io
from flask import Flask, request, jsonify
from flask_cors import CORS
from pyngrok import ngrok
import threading

print("✅ Import thành công!")

# ============================================================================
# BƯỚC 3: LOAD VINTERN MODEL
# ============================================================================
model_name = "5CD-AI/Vintern-Embedding-1B"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"\n{'='*80}")
print(f"🔥 Device: {device.upper()}")
print(f"📥 Đang load model {model_name}...")
print(f"⏳ Quá trình này mất ~2-5 phút lần đầu...")
print(f"{'='*80}\n")

processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

model = AutoModel.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
).eval()

if device == "cuda":
    model = model.cuda()

print(f"\n✅ Model loaded successfully!\n")

# ============================================================================
# BƯỚC 4: HELPER FUNCTIONS
# ============================================================================
def base64_to_image(base64_str):
    """Convert base64 to PIL Image"""
    img_bytes = base64.b64decode(base64_str)
    return Image.open(io.BytesIO(img_bytes))

def tensor_to_base64(tensor):
    """Convert tensor to base64"""
    buffer = io.BytesIO()
    # Convert to float32 để tương thích với CPU
    numpy_array = tensor.cpu().float().numpy()
    np.save(buffer, numpy_array)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

def base64_to_tensor(base64_str):
    """Convert base64 to tensor"""
    tensor_bytes = base64.b64decode(base64_str)
    buffer = io.BytesIO(tensor_bytes)
    numpy_array = np.load(buffer, allow_pickle=False)
    tensor = torch.from_numpy(numpy_array)
    if device == "cuda":
        tensor = tensor.cuda()
    return tensor

# ============================================================================
# BƯỚC 5: SETUP FLASK API
# ============================================================================
app = Flask(__name__)
CORS(app)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "device": device})

@app.route('/encode_images', methods=['POST'])
def encode_images():
    try:
        data = request.json
        images_b64 = data['images']

        print(f"📸 Encoding {len(images_b64)} images...")

        # Convert base64 to PIL Images
        images = [base64_to_image(img_b64) for img_b64 in images_b64]

        # Process images
        batch_images = processor.process_images(images)

        # Move to device
        if device == "cuda":
            batch_images["pixel_values"] = batch_images["pixel_values"].cuda().bfloat16()
            batch_images["input_ids"] = batch_images["input_ids"].cuda()
            batch_images["attention_mask"] = batch_images["attention_mask"].cuda().bfloat16()
        else:
            batch_images["pixel_values"] = batch_images["pixel_values"].float()
            batch_images["input_ids"] = batch_images["input_ids"]
            batch_images["attention_mask"] = batch_images["attention_mask"].float()

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(**batch_images)

        # Convert to base64
        embeddings_b64 = [tensor_to_base64(embeddings[i]) for i in range(len(images))]

        print(f"✅ Done encoding {len(images)} images")

        return jsonify({"embeddings": embeddings_b64})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/encode_texts', methods=['POST'])
def encode_texts():
    try:
        data = request.json
        texts = data['texts']

        print(f"📝 Encoding {len(texts)} texts...")

        # Process texts
        batch_texts = processor.process_docs(texts)

        # Move to device
        if device == "cuda":
            batch_texts["input_ids"] = batch_texts["input_ids"].cuda()
            batch_texts["attention_mask"] = batch_texts["attention_mask"].cuda().bfloat16()
        else:
            batch_texts["input_ids"] = batch_texts["input_ids"]
            batch_texts["attention_mask"] = batch_texts["attention_mask"].float()

        # Generate embeddings
        with torch.no_grad():
            embeddings = model(**batch_texts)

        # Convert to base64
        embeddings_b64 = [tensor_to_base64(embeddings[i]) for i in range(len(texts))]

        print(f"✅ Done encoding {len(texts)} texts")

        return jsonify({"embeddings": embeddings_b64})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/encode_query', methods=['POST'])
def encode_query():
    try:
        data = request.json
        query = data['query']

        print(f"🔍 Encoding query: {query[:50]}...")

        # Process query
        batch_query = processor.process_queries([query])

        # Move to device
        if device == "cuda":
            batch_query["input_ids"] = batch_query["input_ids"].cuda()
            batch_query["attention_mask"] = batch_query["attention_mask"].cuda().bfloat16()
        else:
            batch_query["input_ids"] = batch_query["input_ids"]
            batch_query["attention_mask"] = batch_query["attention_mask"].float()

        # Generate embedding
        with torch.no_grad():
            embedding = model(**batch_query)

        # Convert to base64
        embedding_b64 = tensor_to_base64(embedding)

        print(f"✅ Done encoding query")

        return jsonify({"embedding": embedding_b64})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/compute_similarity', methods=['POST'])
def compute_similarity():
    try:
        data = request.json
        query_emb_b64 = data['query_embedding']
        docs_emb_b64 = data['doc_embeddings']

        print(f"🔢 Computing similarity for {len(docs_emb_b64)} documents...")

        # Convert to tensors
        query_embedding = base64_to_tensor(query_emb_b64)
        doc_embeddings = [base64_to_tensor(emb_b64) for emb_b64 in docs_emb_b64]

        # Compute similarity
        scores = processor.score_multi_vector(query_embedding, doc_embeddings)

        # Convert to base64
        scores_b64 = tensor_to_base64(scores[0])

        print(f"✅ Done computing similarity")

        return jsonify({"scores": scores_b64})

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

# ============================================================================
# BƯỚC 6: SETUP NGROK VÀ CHẠY SERVER
# ============================================================================
print("\n" + "="*80)
print("🌐 Đang setup ngrok tunnel...")
print("="*80 + "\n")

# Set ngrok auth token
import os
NGROK_AUTH_TOKEN = os.environ.get("NGROK_AUTH_TOKEN")
if not NGROK_AUTH_TOKEN:
    raise RuntimeError("NGROK_AUTH_TOKEN chưa được set trong environment")
ngrok.set_auth_token(NGROK_AUTH_TOKEN)

# Terminate any existing tunnels
ngrok.kill()

# Start ngrok tunnel
public_url = ngrok.connect(5000)

print("\n" + "="*80)
print("🎉 VINTERN API SERVER ĐANG CHẠY!")
print("="*80)
print(f"\n🌐 Public URL: {public_url}")
print(f"\n⚠️ QUAN TRỌNG:")
print(f"   1. Copy URL bên trên")
print(f"   2. Mở file config.py trên LOCAL")
print(f"   3. Set: VINTERN_API_URL = '{public_url}'")
print(f"   4. Restart local app: python app.py")
print(f"\n🔥 Server sẽ chạy cho đến khi bạn stop cell này...")
print(f"💡 Giữ tab Colab mở để tránh bị disconnect!")
print("="*80 + "\n")

# Run Flask server
app.run(port=5000)