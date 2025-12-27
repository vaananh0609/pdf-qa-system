# -*- coding: utf-8 -*-
"""
Vintern Embedding API Client
Client để gọi API Vintern từ Colab server
"""
import requests
import base64
import io
import logging
from typing import List, Optional
from PIL import Image
import torch
import numpy as np
from config import Config

class VinternClient:
    def __init__(self, api_url: Optional[str] = None):
        """
        Khởi tạo Vintern client
        
        Args:
            api_url: URL của Vintern API server (từ Colab)
        """
        self.api_url = api_url or getattr(Config, 'VINTERN_API_URL', None)
        self._initialized = False

        # Nếu chưa cấu hình URL thì coi như service không khả dụng (không raise)
        if not self.api_url:
            logging.info("ℹ️ VINTERN_API_URL chưa được cấu hình; Vintern sẽ ở trạng thái unavailable")
            return

        # Đảm bảo URL không có trailing slash
        self.api_url = self.api_url.rstrip('/')
        self._test_connection()

    def set_api_url(self, api_url: str):
        """
        Cập nhật lại URL Vintern API và test kết nối.
        Dùng cho trường hợp muốn override URL runtime.
        """
        if not api_url:
            raise ValueError("api_url không được rỗng")
        self.api_url = api_url.rstrip('/')
        self._initialized = False
        self._test_connection()
    
    def _test_connection(self):
        """Test connection đến API"""
        try:
            response = requests.get(f"{self.api_url}/health", timeout=5)
            if response.status_code == 200:
                self._initialized = True
                logging.info(f"✅ Kết nối Vintern API thành công: {self.api_url}")
                return True
            else:
                self._initialized = False
                logging.error(f"❌ Vintern API không phản hồi đúng")
                return False
        except Exception as e:
            self._initialized = False
            logging.warning(f"⚠️ Không thể kết nối Vintern API: {e}")
            return False
    
    def is_available(self) -> bool:
        """Kiểm tra API có sẵn không"""
        return self._initialized and self.api_url is not None
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        img_bytes = buffered.getvalue()
        return base64.b64encode(img_bytes).decode('utf-8')
    
    def encode_images(self, images: List[Image.Image]) -> List[torch.Tensor]:
        """
        Encode images thông qua API
        Gọi API /encode_images để tạo embeddings cho ảnh
        
        Args:
            images: List các PIL Image
            
        Returns:
            List các torch.Tensor embeddings (as CPU tensors)
        """
        if not self.is_available():
            raise Exception("Vintern API không khả dụng")
        
        try:
            # Convert images to base64
            images_b64 = [self._image_to_base64(img) for img in images]
            
            # Call API
            response = requests.post(
                f"{self.api_url}/encode_images",
                json={"images": images_b64},
                timeout=300  # 5 minutes timeout
            )
            
            if response.status_code != 200:
                error_msg = response.text
                logging.error(f"❌ Lỗi encode images: {error_msg}")
                raise Exception(f"API error: {error_msg}")
            
            result = response.json()
            embeddings_data = result.get("embeddings", [])
            
            # Convert back to tensors
            embeddings = []
            for emb_data in embeddings_data:
                # Decode base64 numpy array
                emb_bytes = base64.b64decode(emb_data)
                buffer = io.BytesIO(emb_bytes)
                numpy_array = np.load(buffer, allow_pickle=False)
                tensor = torch.from_numpy(numpy_array)
                embeddings.append(tensor)
            
            logging.info(f"✅ Đã encode {len(images)} ảnh thành công")
            return embeddings
            
        except Exception as e:
            logging.error(f"❌ Lỗi encode images: {e}")
            raise
    
    def encode_query(self, query: str) -> torch.Tensor:
        """
        Encode query text thông qua API
        
        Args:
            query: Query text string
            
        Returns:
            torch.Tensor embedding
        """
        if not self.is_available():
            raise Exception("Vintern API không khả dụng")
        
        try:
            response = requests.post(
                f"{self.api_url}/encode_query",
                json={"query": query},
                timeout=30
            )
            
            if response.status_code != 200:
                error_msg = response.text
                logging.error(f"❌ Lỗi encode query: {error_msg}")
                raise Exception(f"API error: {error_msg}")
            
            result = response.json()
            emb_data = result.get("embedding", "")
            
            # Convert back to tensor
            emb_bytes = base64.b64decode(emb_data)
            buffer = io.BytesIO(emb_bytes)
            numpy_array = np.load(buffer, allow_pickle=False)
            tensor = torch.from_numpy(numpy_array)
            
            logging.info(f"✅ Đã encode query thành công")
            return tensor
            
        except Exception as e:
            logging.error(f"❌ Lỗi encode query: {e}")
            raise
    
    def compute_similarity(self, query_embedding: torch.Tensor, 
                          doc_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        Tính similarity qua API
        
        Args:
            query_embedding: Query embedding tensor
            doc_embeddings: List các document embedding tensors
            
        Returns:
            torch.Tensor chứa similarity scores
        """
        if not self.is_available():
            raise Exception("Vintern API không khả dụng")
        
        try:
            # Convert to numpy arrays and then to base64
            def tensor_to_b64(tensor):
                buffer = io.BytesIO()
                np.save(buffer, tensor.cpu().numpy())
                return base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            query_b64 = tensor_to_b64(query_embedding)
            docs_b64 = [tensor_to_b64(emb) for emb in doc_embeddings]
            
            response = requests.post(
                f"{self.api_url}/compute_similarity",
                json={
                    "query_embedding": query_b64,
                    "doc_embeddings": docs_b64
                },
                timeout=60
            )
            
            if response.status_code != 200:
                error_msg = response.text
                logging.error(f"❌ Lỗi compute similarity: {error_msg}")
                raise Exception(f"API error: {error_msg}")
            
            result = response.json()
            scores_b64 = result.get("scores", "")
            
            # Convert back to tensor
            scores_bytes = base64.b64decode(scores_b64)
            buffer = io.BytesIO(scores_bytes)
            numpy_array = np.load(buffer, allow_pickle=False)
            scores = torch.from_numpy(numpy_array)
            
            logging.info(f"✅ Đã tính similarity cho {len(doc_embeddings)} documents")
            return scores
            
        except Exception as e:
            logging.error(f"❌ Lỗi compute similarity: {e}")
            raise
    
    def embedding_to_bytes(self, embedding: torch.Tensor) -> bytes:
        """Convert embedding tensor to bytes"""
        buffer = io.BytesIO()
        np.save(buffer, embedding.cpu().numpy())
        return buffer.getvalue()
    
    def bytes_to_embedding(self, data: bytes) -> torch.Tensor:
        """Convert bytes to embedding tensor"""
        buffer = io.BytesIO(data)
        numpy_array = np.load(buffer, allow_pickle=False)
        return torch.from_numpy(numpy_array)
    
    def health_check(self) -> bool:
        return self.is_available()


def get_vintern_client(api_url: Optional[str] = None) -> VinternClient:
    """
    Helper để tạo VinternClient, fallback sang Config nếu không truyền api_url.
    Được dùng bởi các service khác (vd: PDFService).
    """
    # Nếu không truyền api_url thì VinternClient sẽ tự lấy từ Config.VINTERN_API_URL
    client = VinternClient(api_url=api_url)
    return client
