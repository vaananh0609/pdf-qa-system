import google.generativeai as genai
from groq import Groq
from config import Config
from typing import List, Dict, Optional, Set
import logging
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
try:
    from vintern_client import VinternClient
except ImportError:
    VinternClient = None
try:
    from PIL import Image
    from pdf_processor import PDFProcessor
    from database import db
except ImportError:
    Image = None
    PDFProcessor = None

class GeminiService:
    def __init__(self):
        # Delay configuring Gemini until first use to speed up app start
        # và hỗ trợ nhiều API key để fallback
        self._gemini_keys = getattr(Config, "GEMINI_API_KEYS", [Config.GEMINI_API_KEY])
        self._current_key_index = 0
        self.model = None
        self._gemini_configured = False

        # Mô hình embedding text (fallback khi AI method trả về quá ít chunks)
        self._text_embedding_model: Optional[SentenceTransformer] = None

        # Groq client (fallback khi toàn bộ Gemini lỗi / hết quota)
        self._groq_client: Optional[Groq] = None
        
        # Khởi tạo Vintern client (nếu có)
        self.vintern = None
        if VinternClient:
            try:
                self.vintern = VinternClient()
            except Exception as e:
                logging.warning(f"⚠️ Không thể khởi tạo Vintern client: {e}")
        
        # Khởi tạo PDF processor để xử lý ảnh
        self.processor = None
        if PDFProcessor:
            try:
                self.processor = PDFProcessor()
            except Exception as e:
                logging.warning(f"⚠️ Không thể khởi tạo PDF processor: {e}")

    def _ensure_gemini(self):
        """
        Cấu hình Gemini với cơ chế fallback nhiều API key.
        Thử lần lượt các key trong Config.GEMINI_API_KEYS cho tới khi khởi tạo thành công.
        """
        if self._gemini_configured and self.model is not None:
            return

        last_error = None
        for idx, key in enumerate(self._gemini_keys):
            if not key:
                continue
            try:
                genai.configure(api_key=key)
                self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
                self._current_key_index = idx
                self._gemini_configured = True
                logging.info(f"✅ Khởi tạo Gemini thành công với key index {idx}")
                return
            except Exception as e:
                last_error = e
                logging.warning(f"⚠️ Lỗi khởi tạo Gemini với key index {idx}: {e}")

        logging.error(f"❌ Không thể khởi tạo Gemini với bất kỳ key nào: {last_error}")
        self.model = None
        self._gemini_configured = False

    def _rotate_gemini_key_and_reinit(self) -> bool:
        """
        Khi gặp lỗi quota / 429 / auth, xoay sang key tiếp theo và khởi tạo lại model.
        """
        if not self._gemini_keys:
            return False
        start_index = self._current_key_index
        n = len(self._gemini_keys)
        last_error = None

        for step in range(1, n + 1):
            idx = (start_index + step) % n
            key = self._gemini_keys[idx]
            if not key:
                continue
            try:
                genai.configure(api_key=key)
                self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
                self._current_key_index = idx
                self._gemini_configured = True
                logging.info(f"🔁 Đổi sang Gemini key index {idx} thành công")
                return True
            except Exception as e:
                last_error = e
                logging.warning(f"⚠️ Lỗi khi đổi sang Gemini key index {idx}: {e}")

        logging.error(f"❌ Không thể xoay sang bất kỳ Gemini key nào khác: {last_error}")
        self.model = None
        self._gemini_configured = False
        return False

    def _ensure_groq(self):
        """
        Khởi tạo Groq client nếu có key. Dùng SDK chính thức, không cần base_url.
        """
        if self._groq_client is not None:
            return
        api_key = getattr(Config, "GROQ_API_KEY", None)
        if not api_key:
            logging.warning("GROQ_API_KEY chưa được cấu hình, bỏ qua fallback Groq")
            return
        try:
            self._groq_client = Groq(api_key=api_key)
            logging.info("✅ Khởi tạo Groq client thành công")
        except Exception as e:
            logging.error(f"❌ Không thể khởi tạo Groq client: {e}")
            self._groq_client = None

    def _ensure_text_embedding_model(self):
        """
        Khởi tạo SentenceTransformer cho text embedding (fallback retrieval).
        """
        if self._text_embedding_model is not None:
            return
        try:
            self._text_embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("✅ Khởi tạo text embedding model (SentenceTransformer) thành công")
        except Exception as e:
            logging.error(f"❌ Không thể khởi tạo text embedding model: {e}")
            self._text_embedding_model = None
    
    def generate_answer(self, question: str, context_chunks: List[Dict]) -> Dict:
        """Tạo câu trả lời từ context chunks - hỗ trợ cả text và image chunks"""
        try:
            # Tách chunks thành text và image
            text_chunks = [c for c in context_chunks if c.get('chunk_type') != 'image']
            image_chunks = [c for c in context_chunks if c.get('chunk_type') == 'image']
            
            # Nếu có image chunks, dùng Gemini Vision
            if image_chunks:
                return self._generate_answer_with_images(question, text_chunks, image_chunks)
            
            # Nếu chỉ có text chunks, dùng phương thức cũ
            return self._generate_answer_text_only(question, text_chunks)
            
        except Exception as e:
            logging.error(f"Lỗi tạo câu trả lời: {e}")
            return {
                'answer': "Xin lỗi, có lỗi xảy ra khi tạo câu trả lời.",
                'sources': [],
                'success': False,
                'error': str(e)
            }
    
    def _generate_answer_text_only(self, question: str, context_chunks: List[Dict]) -> Dict:
        """Tạo câu trả lời chỉ từ text chunks"""
        try:
            # Lấy thời gian upload của các file để xác định file mới nhất
            file_upload_times = {}
            if db:
                unique_filenames = list(set(chunk.get('filename') for chunk in context_chunks if chunk.get('filename')))
                for filename in unique_filenames:
                    pdf_files = db.find_documents('pdf_files', {'filename': filename}, limit=1)
                    if pdf_files:
                        file_upload_times[filename] = pdf_files[0].get('created_at')
            
            # Xác định file mới nhất (upload gần nhất)
            latest_file = None
            if file_upload_times:
                latest_file = max(file_upload_times.items(), key=lambda x: x[1] if x[1] else datetime.min)[0]
            
            # Sắp xếp chunks: file mới nhất lên trước
            chunks_with_info = []
            for i, chunk in enumerate(context_chunks):
                chunk_filename = chunk.get('filename', 'N/A')
                is_latest = (latest_file and chunk_filename == latest_file)
                chunks_with_info.append((i, chunk, is_latest))
            
            # Sắp xếp: file mới nhất trước, sau đó mới đến file cũ
            chunks_with_info.sort(key=lambda x: (not x[2], x[0]))
            
            # Build context text và sources với thứ tự mới
            context_text_sorted = ""
            sources = []
            for new_idx, (old_idx, chunk, is_latest) in enumerate(chunks_with_info):
                chunk_text = chunk.get('text', '') or chunk.get('content', '')
                chunk_filename = chunk.get('filename', 'N/A')
                
                file_info = chunk_filename
                if is_latest:
                    file_info += " [FILE MỚI NHẤT - Upload gần nhất - BẢN CẬP NHẬT]"
                elif chunk_filename in file_upload_times:
                    upload_time = file_upload_times[chunk_filename]
                    if upload_time:
                        if isinstance(upload_time, datetime):
                            time_str = upload_time.strftime("%d/%m/%Y %H:%M")
                        else:
                            time_str = str(upload_time)
                        file_info += f" [Upload: {time_str} - FILE CŨ]"
                
                context_text_sorted += f"Chunk {new_idx + 1} (File: {file_info}, Trang: {chunk.get('page_number', 'N/A')}):\n"
                context_text_sorted += chunk_text + "\n\n"
                
                # Build sources theo thứ tự mới
                sources.append({
                    'filename': chunk_filename,
                    'page_number': chunk.get('page_number', 0),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'char_start': chunk.get('char_start', 0),
                    'char_end': chunk.get('char_end', 0),
                    'chunk_id': chunk.get('chunk_id', chunk.get('_id', ''))
                })
            
            # Tạo prompt sau khi đã build xong context và sources
            prompt = f"""
Bạn là một trợ lý AI chuyên về việc trả lời câu hỏi dựa trên tài liệu PDF. 
Hãy trả lời câu hỏi dựa trên thông tin trong các đoạn văn bản được cung cấp.

Thông tin tài liệu (đã sắp xếp: file mới nhất ở trên):
{context_text_sorted}

Câu hỏi: {question}

Hướng dẫn:
1. Trả lời câu hỏi một cách chính xác dựa trên thông tin trong tài liệu
2. Nếu không tìm thấy thông tin, hãy nói rõ "Không tìm thấy thông tin trong tài liệu"
3. Trích dẫn chính xác từ tài liệu khi có thể
4. Trả lời bằng tiếng Việt có dấu, không in đậm
5. QUY TẮC QUAN TRỌNG NHẤT - Ưu tiên file mới nhất: 
   - Nếu có nhiều file chứa thông tin về CÙNG MỘT CHỦ ĐỀ (ví dụ: cùng một sự kiện, cùng một quy định, cùng một lịch trình), BẮT BUỘC chỉ sử dụng thông tin từ file được đánh dấu [FILE MỚI NHẤT - Upload gần nhất - BẢN CẬP NHẬT].
   - KHÔNG ĐƯỢC kết hợp hoặc đề cập đến thông tin từ file cũ nếu file mới đã có thông tin đó.
   - Chỉ sử dụng thông tin từ file cũ (đánh dấu [FILE CŨ]) khi file mới KHÔNG chứa thông tin đó.
   - File mới nhất là bản cập nhật, nên thông tin trong đó luôn chính xác và đầy đủ hơn file cũ.
6. RẤT QUAN TRỌNG: Sau phần trả lời, hãy thêm một dòng duy nhất chứa MỘT đối tượng JSON hợp lệ với hai khóa: "text" và "images". Giá trị của mỗi khóa là mảng các chỉ số (1-based) của các chunks bạn thực sự đã SỬ DỤNG để tạo câu trả lời. Ví dụ:
    {{"text": [1, 3], "images": []}}
    Nếu bạn không sử dụng chunk nào, trả về {{"text": [], "images": []}}.
7. Tuyệt đối không in thêm bất kỳ danh sách nguồn nào khác ngoài dòng JSON đó (không thêm chữ "CHUNKS_USED" hay giải thích).

Trả lời:
"""
            
            # Gọi API Gemini (ensure configured) với fallback nhiều key + Groq
            response = None
            # 1) Thử Gemini với key hiện tại, nếu lỗi quota/auth thì rotate key
            for attempt in range(len(self._gemini_keys)):
                if not self.model:
                    self._ensure_gemini()
                if not self.model:
                    break
                try:
                    response = self.model.generate_content(prompt)
                    break
                except Exception as e:
                    msg = str(e).lower()
                    logging.warning(f"⚠️ Lỗi khi gọi Gemini (attempt {attempt}): {e}")
                    # Một số lỗi quota/429 hoặc auth → thử đổi key
                    if any(x in msg for x in ["429", "quota", "rate limit", "permission", "unauthorized", "invalid api key"]):
                        rotated = self._rotate_gemini_key_and_reinit()
                        if not rotated:
                            logging.error("❌ Hết key Gemini khả dụng, sẽ fallback sang Groq")
                            self.model = None
                            break
                        continue
                    else:
                        # Lỗi khác không chắc do key → thoát để fallback Groq
                        self.model = None
                        break

            # 2) Nếu không có response từ Gemini → fallback Groq nếu có
            if response is None:
                self._ensure_groq()
                if self._groq_client is None:
                    raise Exception("Không có Gemini cũng như Groq khả dụng")
                try:
                    groq_resp = self._groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "Bạn là trợ lý AI tiếng Việt, trả lời ngắn gọn, chính xác, dựa trên context PDF được cung cấp."},
                            {"role": "user", "content": prompt},
                        ],
                        temperature=0.2,
                    )
                    answer = groq_resp.choices[0].message["content"]
                    # Không có CHUNKS_USED chuẩn từ Groq, nên trả về toàn bộ sources
                    return {
                        'answer': answer,
                        'sources': sources,
                        'success': True
                    }
                except Exception as ge:
                    logging.error(f"❌ Lỗi khi fallback Groq: {ge}")
                    raise Exception(f"Không thể tạo câu trả lời bằng Gemini/Groq: {ge}")

            try:
                answer = response.text
            except AttributeError:
                # Fallback nếu response không có .text
                try:
                    answer = str(response.candidates[0].content.parts[0].text)
                except (AttributeError, IndexError, KeyError):
                    raise Exception("Không thể lấy text từ response Gemini")
            
            # Tách answer và chunks được dùng bằng cách tìm JSON footer {"text": [...], "images": [...]}
            used_chunks_indices = []
            actual_sources = []
            try:
                # Try to extract a JSON object at the end of the response
                jstart = answer.rfind('{')
                jend = answer.rfind('}')
                json_obj = None
                if jstart != -1 and jend != -1 and jend > jstart:
                    json_str = answer[jstart:jend+1]
                    import json as _json
                    try:
                        parsed = _json.loads(json_str)
                        # Remove JSON footer from answer
                        answer = answer[:jstart].strip()
                        text_indices = parsed.get('text', []) if isinstance(parsed.get('text', []), list) else []
                        # Convert to 0-based indices (sau khi sắp xếp, sources đã được build theo thứ tự mới)
                        used_chunks_indices = [int(n) - 1 for n in text_indices if isinstance(n, int) or (isinstance(n, str) and n.isdigit())]
                    except Exception:
                        used_chunks_indices = []
                else:
                    used_chunks_indices = []
            except Exception as e:
                logging.warning(f"Không thể parse JSON footer từ Gemini response: {e}")
                used_chunks_indices = []

            # Chỉ lấy sources của chunks được sử dụng; nếu model không trả về JSON hoặc mảng rỗng => trả về sources rỗng để tránh gán nhầm nguồn
            if used_chunks_indices:
                actual_sources = [sources[i] for i in used_chunks_indices if 0 <= i < len(sources)]
            else:
                actual_sources = []
            
            return {
                'answer': answer,
                'sources': actual_sources,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Lỗi tạo câu trả lời text only: {e}")
            return {
                'answer': "Xin lỗi, có lỗi xảy ra khi tạo câu trả lời.",
                'sources': [],
                'success': False,
                'error': str(e)
            }
    
    def _generate_answer_with_images(self, question: str, text_chunks: List[Dict], 
                                    image_chunks: List[Dict]) -> Dict:
        """Tạo câu trả lời với Gemini Vision đọc ảnh"""
        try:
            if not self.model:
                self._ensure_gemini()
            if not self.model:
                raise Exception('Gemini không khả dụng')
            
            # Load ảnh từ database
            images = []
            image_sources = []
            
            for chunk in image_chunks:
                chunk_id = chunk.get('chunk_id') or chunk.get('_id')
                if not chunk_id:
                    continue
                
                # Lấy image_data từ database
                image_data = db.get_binary_field('pdf_chunks', {'chunk_id': chunk_id}, 'image_data')
                if image_data is not None and self.processor:
                    try:
                        # get_binary_field đã trả về bytes rồi, không cần convert lại
                        img = self.processor.bytes_to_image(image_data)
                        # Resize ảnh nếu quá lớn (Gemini có giới hạn)
                        max_size = 2048
                        if img.width > max_size or img.height > max_size:
                            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                        images.append(img)
                        # Thông tin file sẽ được thêm sau khi có file_upload_times
                        image_sources.append({
                            'filename': chunk.get('filename', 'N/A'),
                            'page_number': chunk.get('page_number', 0),
                            'chunk_index': chunk.get('chunk_index', 0),
                            'chunk_id': chunk_id,
                            'chunk_type': 'image'
                        })
                    except Exception as e:
                        filename = chunk.get('filename', 'unknown')
                        page_num = chunk.get('page_number', chunk.get('chunk_index', 0) + 1 if chunk.get('chunk_index') is not None else '?')
                        logging.warning(f"⚠️ Không thể load ảnh từ {filename} (trang {page_num}): {e}")
                else:
                    # Log khi không có image_data
                    filename = chunk.get('filename', 'unknown')
                    page_num = chunk.get('page_number', chunk.get('chunk_index', 0) + 1 if chunk.get('chunk_index') is not None else '?')
                    logging.debug(f"⚠️ Chunk {filename} (trang {page_num}) không có image_data")
            
            # Lấy thời gian upload của các file để xác định file mới nhất
            file_upload_times = {}
            if db:
                all_filenames = list(set(chunk.get('filename') for chunk in (text_chunks + image_chunks) if chunk.get('filename')))
                for filename in all_filenames:
                    pdf_files = db.find_documents('pdf_files', {'filename': filename}, limit=1)
                    if pdf_files:
                        file_upload_times[filename] = pdf_files[0].get('created_at')
            
            # Xác định file mới nhất
            latest_file = None
            if file_upload_times:
                latest_file = max(file_upload_times.items(), key=lambda x: x[1] if x[1] else datetime.min)[0]
            
            # Chuẩn bị text context
            context_text = ""
            text_sources = []
            for i, chunk in enumerate(text_chunks):
                chunk_text = chunk.get('text', '') or chunk.get('content', '')
                chunk_filename = chunk.get('filename', 'N/A')
                
                # Đánh dấu file mới nhất
                file_info = chunk_filename
                if latest_file and chunk_filename == latest_file:
                    file_info += " [FILE MỚI NHẤT - Upload gần nhất]"
                elif chunk_filename in file_upload_times:
                    upload_time = file_upload_times[chunk_filename]
                    if upload_time:
                        if isinstance(upload_time, datetime):
                            time_str = upload_time.strftime("%d/%m/%Y %H:%M")
                        else:
                            time_str = str(upload_time)
                        file_info += f" [Upload: {time_str}]"
                
                context_text += f"Chunk {i+1} (File: {file_info}, Trang: {chunk.get('page_number', 'N/A')}):\n"
                context_text += chunk_text + "\n\n"
                
                text_sources.append({
                    'filename': chunk_filename,
                    'page_number': chunk.get('page_number', 0),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'char_start': chunk.get('char_start', 0),
                    'char_end': chunk.get('char_end', 0),
                    'chunk_id': chunk.get('chunk_id', chunk.get('_id', '')),
                    'chunk_type': 'text'
                })
            
            # Tạo prompt với ảnh
            prompt_parts = []
            
            if context_text:
                prompt_parts.append(f"""
Bạn là một trợ lý AI chuyên về việc trả lời câu hỏi dựa trên tài liệu PDF. 
Hãy trả lời câu hỏi dựa trên thông tin trong các đoạn văn bản và hình ảnh được cung cấp.

Thông tin văn bản:
{context_text}

Câu hỏi: {question}

Hướng dẫn:
1. Trả lời câu hỏi một cách chính xác dựa trên thông tin trong tài liệu và hình ảnh
2. Nếu không tìm thấy thông tin, hãy nói rõ "Không tìm thấy thông tin trong tài liệu"
3. Trích dẫn chính xác từ tài liệu khi có thể
4. Trả lời bằng tiếng Việt có dấu, không in đậm
5. RẤT QUAN TRỌNG - Ưu tiên file mới nhất: Nếu có nhiều file chứa thông tin tương tự hoặc trùng lặp về cùng một chủ đề, HÃY ƯU TIÊN SỬ DỤNG THÔNG TIN TỪ FILE ĐƯỢC UPLOAD GẦN NHẤT (file mới nhất). File mới nhất thường là bản cập nhật của file cũ, nên thông tin trong file mới nhất chính xác và đáng tin cậy hơn. Chỉ sử dụng thông tin từ file cũ nếu file mới không chứa thông tin đó.
6. Cuối câu trả lời, hãy thêm dòng "CHUNKS_USED:" và liệt kê:
   - Số thứ tự các text chunks bạn đã sử dụng (ví dụ: TEXT: 1, 3)
   - Số thứ tự các hình ảnh bạn đã sử dụng (ví dụ: IMAGES: 1, 2)
   (Ví dụ: CHUNKS_USED: TEXT: 1, 3 | IMAGES: 1, 2)

Trả lời:
""")
            else:
                prompt_parts.append(f"""
Bạn là một trợ lý AI chuyên về việc trả lời câu hỏi dựa trên hình ảnh từ tài liệu PDF.
Hãy trả lời câu hỏi dựa trên thông tin trong các hình ảnh được cung cấp.

Câu hỏi: {question}

Hướng dẫn:
1. Trả lời câu hỏi một cách chính xác dựa trên thông tin trong hình ảnh
2. Nếu không tìm thấy thông tin, hãy nói rõ "Không tìm thấy thông tin trong tài liệu"
3. Trả lời bằng tiếng Việt có dấu, không in đậm
4. RẤT QUAN TRỌNG - Ưu tiên file mới nhất: Nếu có nhiều file chứa thông tin tương tự hoặc trùng lặp về cùng một chủ đề, HÃY ƯU TIÊN SỬ DỤNG THÔNG TIN TỪ FILE ĐƯỢC UPLOAD GẦN NHẤT (file mới nhất). File mới nhất thường là bản cập nhật của file cũ, nên thông tin trong file mới nhất chính xác và đáng tin cậy hơn.
5. Cuối câu trả lời, hãy thêm dòng "CHUNKS_USED:" và liệt kê số thứ tự các hình ảnh bạn đã sử dụng (ví dụ: CHUNKS_USED: IMAGES: 1, 2, 4)

Trả lời:
""")
            
            # Thêm ảnh vào prompt
            for img in images:
                prompt_parts.append(img)
            
            # Gọi Gemini Vision (tạm thời chưa fallback Groq vì cần hỗ trợ hình ảnh)
            response = self.model.generate_content(prompt_parts)
            try:
                answer = response.text
            except AttributeError:
                try:
                    answer = str(response.candidates[0].content.parts[0].text)
                except (AttributeError, IndexError, KeyError):
                    raise Exception("Không thể lấy text từ response Gemini")
            
            # Tách answer và chunks được dùng từ format: CHUNKS_USED: TEXT: 1, 3 | IMAGES: 1, 2
            # Hoặc: CHUNKS_USED: IMAGES: 1, 2
            used_text_indices = []
            used_image_indices = []
            actual_sources = []
            
            try:
                # Tìm dòng CHUNKS_USED trong answer
                chunks_used_pattern = re.search(r'CHUNKS_USED:\s*(.+)', answer, re.IGNORECASE | re.MULTILINE)
                if chunks_used_pattern:
                    chunks_used_line = chunks_used_pattern.group(1).strip()
                    # Remove dòng CHUNKS_USED khỏi answer
                    answer = answer[:chunks_used_pattern.start()].strip()
                    
                    # Parse TEXT: 1, 3
                    text_match = re.search(r'TEXT:\s*([0-9,\s]+)', chunks_used_line, re.IGNORECASE)
                    if text_match:
                        text_nums = re.findall(r'\d+', text_match.group(1))
                        used_text_indices = [int(n) - 1 for n in text_nums if 0 <= int(n) - 1 < len(text_sources)]
                    
                    # Parse IMAGES: 1, 2
                    image_match = re.search(r'IMAGES?:\s*([0-9,\s]+)', chunks_used_line, re.IGNORECASE)
                    if image_match:
                        image_nums = re.findall(r'\d+', image_match.group(1))
                        used_image_indices = [int(n) - 1 for n in image_nums if 0 <= int(n) - 1 < len(image_sources)]
                    
                    logging.info(f"✅ Parse CHUNKS_USED: TEXT={used_text_indices}, IMAGES={used_image_indices}")
                else:
                    # Fallback: thử parse JSON format cũ
                    jstart = answer.rfind('{')
                    jend = answer.rfind('}')
                    if jstart != -1 and jend != -1 and jend > jstart:
                        json_str = answer[jstart:jend+1]
                        import json as _json
                        try:
                            parsed = _json.loads(json_str)
                            answer = answer[:jstart].strip()
                            text_list = parsed.get('text', []) if isinstance(parsed.get('text', []), list) else []
                            image_list = parsed.get('images', []) if isinstance(parsed.get('images', []), list) else []
                            used_text_indices = [int(n) - 1 for n in text_list if isinstance(n, int) or (isinstance(n, str) and n.isdigit())]
                            used_image_indices = [int(n) - 1 for n in image_list if isinstance(n, int) or (isinstance(n, str) and n.isdigit())]
                            logging.info(f"✅ Parse JSON footer: TEXT={used_text_indices}, IMAGES={used_image_indices}")
                        except Exception as e:
                            logging.warning(f"⚠️ Không thể parse JSON footer: {e}")
            except Exception as e:
                logging.warning(f"⚠️ Lỗi khi parse CHUNKS_USED: {e}")
            
            # Nếu không parse được indices nhưng có image chunks được truyền vào,
            # thì coi như tất cả image chunks đã được sử dụng (fallback)
            if not used_image_indices and image_sources:
                logging.info(f"⚠️ Không parse được image indices, fallback: dùng tất cả {len(image_sources)} image chunks")
                used_image_indices = list(range(len(image_sources)))
            
            # Lấy sources thực tế
            if used_text_indices:
                actual_sources.extend([text_sources[i] for i in used_text_indices if 0 <= i < len(text_sources)])
            if used_image_indices:
                actual_sources.extend([image_sources[i] for i in used_image_indices if 0 <= i < len(image_sources)])
            
            logging.info(f"📌 Tổng số sources: {len(actual_sources)} (text: {len([s for s in actual_sources if s.get('chunk_type') != 'image'])}, image: {len([s for s in actual_sources if s.get('chunk_type') == 'image'])})")
            
            return {
                'answer': answer,
                'sources': actual_sources,
                'success': True
            }
            
        except Exception as e:
            logging.error(f"Lỗi tạo câu trả lời với ảnh: {e}")
            return {
                'answer': "Xin lỗi, có lỗi xảy ra khi tạo câu trả lời.",
                'sources': [],
                'success': False,
                'error': str(e)
            }
    
    def find_relevant_chunks(self, question: str, all_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Tìm chunks liên quan với chiến lược hai tầng:
        1) Dùng AI method (Gemini) để chọn các đoạn quan trọng nhất (cả text và image).
        2) Nếu AI method trả về ít hơn 3 chunks, bổ sung thêm bằng:
           - Embedding-based retrieval cho text chunks
           - Vintern similarity cho image chunks
           để đảm bảo luôn có tối đa top_k (mặc định 5) chunks đưa vào Gemini sinh câu trả lời.
        """
        try:
            if not all_chunks:
                return []

            # Tách text chunks và image chunks
            text_chunks = [c for c in all_chunks if c.get('chunk_type') != 'image']
            image_chunks = [c for c in all_chunks if c.get('chunk_type') == 'image']
            
            logging.info(f"📊 Tổng số chunks: {len(all_chunks)} (text: {len(text_chunks)}, image: {len(image_chunks)})")

            # Bước 1: AI search (Gemini) trên toàn bộ chunks (cả text và image)
            logging.info("🤖 Sử dụng AI search (Gemini) để tìm chunks liên quan")
            ai_chunks = self._find_relevant_chunks_ai(question, all_chunks, top_k)

            # Nếu AI đã tìm được đủ tốt (>=3 hoặc >=top_k) thì dùng luôn
            if len(ai_chunks) >= min(3, top_k):
                logging.info(f"✅ AI method tìm đủ chunks: {len(ai_chunks)}")
                return ai_chunks[:top_k]

            # Bước 2: Fallback bằng embedding để bổ sung cho đủ top_k
            logging.info(f"⚙️ AI method chỉ tìm được {len(ai_chunks)} chunks, fallback thêm bằng embedding để đủ {top_k}")
            remaining = max(0, top_k - len(ai_chunks))
            if remaining == 0:
                return ai_chunks[:top_k]

            # Đánh dấu các chunks đã được chọn
            selected_ids: Set[str] = set()
            for c in ai_chunks:
                cid = str(c.get('chunk_id') or c.get('_id') or f"{c.get('filename','')}#{c.get('chunk_index')}")
                selected_ids.add(cid)

            # Bổ sung text chunks bằng embedding-based retrieval
            embed_text_chunks = self._find_relevant_chunks_embedding(question, text_chunks, remaining, selected_ids)
            
            # Cập nhật selected_ids sau khi thêm text chunks
            for c in embed_text_chunks:
                cid = str(c.get('chunk_id') or c.get('_id') or f"{c.get('filename','')}#{c.get('chunk_index')}")
                selected_ids.add(cid)
            
            # Bổ sung image chunks bằng Vintern similarity (nếu còn slot và có image chunks)
            remaining_after_text = max(0, top_k - len(ai_chunks) - len(embed_text_chunks))
            embed_image_chunks = []
            if remaining_after_text > 0 and image_chunks and self.vintern:
                try:
                    embed_image_chunks = self._find_relevant_image_chunks_vintern(
                        question, image_chunks, remaining_after_text, selected_ids
                    )
                except Exception as e:
                    logging.warning(f"⚠️ Lỗi khi tìm image chunks bằng Vintern: {e}")

            # Kết hợp tất cả chunks
            combined = ai_chunks + embed_text_chunks + embed_image_chunks
            
            # Loại trùng theo chunk_id
            seen: Set[str] = set()
            unique: List[Dict] = []
            for c in combined:
                cid = str(c.get('chunk_id') or c.get('_id') or f"{c.get('filename','')}#{c.get('chunk_index')}")
                if cid in seen:
                    continue
                seen.add(cid)
                unique.append(c)

            logging.info(f"📌 Tổng số chunks sau fallback: {len(unique)} (text: {sum(1 for c in unique if c.get('chunk_type') != 'image')}, image: {sum(1 for c in unique if c.get('chunk_type') == 'image')})")
            return unique[:top_k]
                
        except Exception as e:
            logging.error(f"Lỗi tìm kiếm chunks: {e}")
            # Fallback cuối cùng về chunks đầu tiên
            return all_chunks[:top_k]
    
    def _find_relevant_chunks_ai(self, question: str, all_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """AI method - tìm kiếm semantic"""
        try:
            # Lấy thời gian upload của các file để xác định file mới nhất
            file_upload_times = {}
            if db:
                unique_filenames = list(set(chunk.get('filename') for chunk in all_chunks if chunk.get('filename')))
                for filename in unique_filenames:
                    pdf_files = db.find_documents('pdf_files', {'filename': filename}, limit=1)
                    if pdf_files:
                        file_upload_times[filename] = pdf_files[0].get('created_at')
            
            # Xác định file mới nhất
            latest_file = None
            if file_upload_times:
                latest_file = max(file_upload_times.items(), key=lambda x: x[1] if x[1] else datetime.min)[0]
            
            # Tạo prompt để tìm kiếm semantic
            search_prompt = f"""
            Tìm kiếm thông tin liên quan đến câu hỏi: "{question}"

            Các đoạn văn bản và hình ảnh:
            """
            
            for i, chunk in enumerate(all_chunks):
                chunk_text = chunk.get('text', '') or chunk.get('content', '')
                chunk_type = chunk.get('chunk_type', 'text')
                page_num = chunk.get('page_number', '?')
                filename = chunk.get('filename', 'unknown')
                
                # Đánh dấu file mới nhất
                file_marker = ""
                if latest_file and filename == latest_file:
                    file_marker = " [FILE MỚI NHẤT - Upload gần nhất]"
                elif filename in file_upload_times:
                    upload_time = file_upload_times[filename]
                    if upload_time:
                        if isinstance(upload_time, datetime):
                            time_str = upload_time.strftime("%d/%m/%Y")
                        else:
                            time_str = str(upload_time)
                        file_marker = f" [Upload: {time_str}]"
                
                if chunk_text:
                    # Text chunk: hiển thị nội dung
                    search_prompt += f"\nĐoạn {i+1} (Văn bản, file {filename}{file_marker}, trang {page_num}): {chunk_text[:200]}..."
                elif chunk_type == 'image':
                    # Image chunk: mô tả ngắn để AI biết có ảnh
                    search_prompt += f"\nĐoạn {i+1} (Hình ảnh, file {filename}{file_marker}, trang {page_num}): [Trang {page_num} của file {filename} - chứa hình ảnh/scanned PDF, có thể liên quan đến câu hỏi]"
            
            search_prompt += f"""
            \nHãy xác định các đoạn (văn bản hoặc hình ảnh) nào liên quan nhất đến câu hỏi "{question}".
            LƯU Ý: Nếu có nhiều file chứa thông tin tương tự, hãy ưu tiên chọn các đoạn từ file được đánh dấu [FILE MỚI NHẤT - Upload gần nhất] vì đó là bản cập nhật mới nhất.
            Trả lời chỉ bằng số thứ tự các đoạn (ví dụ: 1, 3, 5) hoặc "không có" nếu không tìm thấy.
            """
            
            if not self.model:
                self._ensure_gemini()
            if not self.model:
                raise Exception('Gemini không khả dụng')
            response = self.model.generate_content(search_prompt)
            relevant_indices = []
            
            # Parse response để lấy các index
            try:
                response_text = response.text.strip()
            except AttributeError:
                # Fallback nếu response không có .text
                try:
                    response_text = str(response.candidates[0].content.parts[0].text).strip()
                except (AttributeError, IndexError, KeyError):
                    logging.error("Không thể lấy text từ response")
                    return []
            if "không có" not in response_text.lower():
                try:
                    # Tìm các số trong response
                    import re
                    numbers = re.findall(r'\d+', response_text)
                    relevant_indices = [int(num) - 1 for num in numbers if int(num) - 1 < len(all_chunks)]
                except:
                    pass
            
            # Trả về top_k chunks liên quan nhất
            relevant_chunks = [all_chunks[i] for i in relevant_indices if i < len(all_chunks)]
            logging.info(f"AI method tìm thấy {len(relevant_chunks)} chunks")

            # Log chi tiết các chunk được chọn (không có score vì là AI chọn theo thứ tự)
            for rank, chunk in enumerate(relevant_chunks[:top_k], start=1):
                filename = chunk.get('filename', 'unknown')
                page_num = chunk.get('page_number', chunk.get('chunk_index', 0) + 1 if chunk.get('chunk_index') is not None else '?')
                chunk_idx = chunk.get('chunk_index', '?')
                logging.info(f"🔎 Text chunk được chọn #{rank}: {filename} (chunk_index={chunk_idx}, trang {page_num})")
            
            return relevant_chunks[:top_k]
            
        except Exception as e:
            logging.error(f"Lỗi AI method: {e}")
            return []

    def _find_relevant_chunks_embedding(
        self,
        question: str,
        all_chunks: List[Dict],
        top_k: int,
        exclude_ids: Set[str]
    ) -> List[Dict]:
        """
        Fallback retrieval dựa trên embedding text khi AI method trả về quá ít chunks.
        Chỉ áp dụng cho text chunks có sẵn trường 'embedding' (list float).
        """
        try:
            if top_k <= 0:
                return []

            # Khởi tạo embedding model
            self._ensure_text_embedding_model()
            if self._text_embedding_model is None:
                logging.warning("⚠️ Không có text embedding model, bỏ qua fallback embedding")
                return []

            # Lọc các text chunks có embedding và chưa bị loại trừ
            candidates = []
            embeddings = []
            for chunk in all_chunks:
                if chunk.get('chunk_type') == 'image':
                    continue
                emb = chunk.get('embedding')
                if not emb:
                    continue
                cid = str(chunk.get('chunk_id') or chunk.get('_id') or f"{chunk.get('filename','')}#{chunk.get('chunk_index')}")
                if cid in exclude_ids:
                    continue
                try:
                    vec = np.array(emb, dtype=np.float32)
                except Exception:
                    continue
                candidates.append(chunk)
                embeddings.append(vec)

            if not candidates:
                logging.info("⚠️ Không có text chunk nào có embedding để fallback")
                return []

            # Tính embedding cho câu hỏi
            q_vec = self._text_embedding_model.encode([question], convert_to_numpy=True)[0].astype(np.float32)

            # Chuẩn hoá vector
            q_norm = np.linalg.norm(q_vec) + 1e-10
            q_vec = q_vec / q_norm
            E = np.stack(embeddings, axis=0)
            norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-10
            E_norm = E / norms

            scores = E_norm @ q_vec  # cosine similarity

            # Lấy top_k theo điểm similarity
            top_k = min(top_k, len(candidates))
            top_indices = np.argsort(-scores)[:top_k]

            results: List[Dict] = []
            for rank, idx in enumerate(top_indices, start=1):
                chunk = candidates[int(idx)]
                filename = chunk.get('filename', 'unknown')
                page_num = chunk.get('page_number', chunk.get('chunk_index', 0) + 1 if chunk.get('chunk_index') is not None else '?')
                chunk_idx = chunk.get('chunk_index', '?')
                logging.info(f"📐 Embedding fallback chọn chunk #{rank}: {filename} (chunk_index={chunk_idx}, trang {page_num}) - score: {scores[int(idx)]:.4f}")
                results.append(chunk)

            return results

        except Exception as e:
            logging.error(f"❌ Lỗi embedding fallback retrieval: {e}")
            return []
    
    def _find_relevant_image_chunks_vintern(
        self,
        question: str,
        image_chunks: List[Dict],
        top_k: int,
        exclude_ids: Set[str]
    ) -> List[Dict]:
        """
        Tìm image chunks liên quan bằng Vintern similarity.
        """
        try:
            if top_k <= 0 or not image_chunks:
                return []
            
            if not self.vintern or not self.vintern.is_available():
                logging.warning("⚠️ Vintern không khả dụng, bỏ qua tìm image chunks")
                return []
            
            # Lọc các image chunks có embedding và chưa bị loại trừ
            candidates = []
            embeddings = []
            for chunk in image_chunks:
                cid = str(chunk.get('chunk_id') or chunk.get('_id') or f"{chunk.get('filename','')}#{chunk.get('chunk_index')}")
                if cid in exclude_ids:
                    continue
                
                embedding_data = chunk.get('embedding_data')
                if not embedding_data:
                    continue
                
                try:
                    # Convert bytes to tensor
                    import torch
                    emb_tensor = self.vintern.bytes_to_embedding(embedding_data)
                    candidates.append(chunk)
                    embeddings.append(emb_tensor)
                except Exception as e:
                    logging.warning(f"⚠️ Không thể load embedding cho image chunk {cid}: {e}")
                    continue
            
            if not candidates:
                logging.info("⚠️ Không có image chunk nào có embedding để tìm kiếm")
                return []
            
            # Encode câu hỏi thành embedding
            query_embedding = self.vintern.encode_query(question)
            
            # Tính similarity
            scores = self.vintern.compute_similarity(query_embedding, embeddings)
            
            # Lấy top_k theo điểm similarity
            top_k = min(top_k, len(candidates))
            top_indices = torch.argsort(scores, descending=True)[:top_k]
            
            results: List[Dict] = []
            for rank, idx in enumerate(top_indices, start=1):
                chunk = candidates[int(idx)]
                filename = chunk.get('filename', 'unknown')
                page_num = chunk.get('page_number', '?')
                chunk_idx = chunk.get('chunk_index', '?')
                score = float(scores[int(idx)])
                logging.info(f"🖼️ Vintern chọn image chunk #{rank}: {filename} (chunk_index={chunk_idx}, trang {page_num}) - score: {score:.4f}")
                results.append(chunk)
            
            return results
            
        except Exception as e:
            logging.error(f"❌ Lỗi tìm image chunks bằng Vintern: {e}")
            return []
    
    def _find_relevant_chunks_hybrid(self, question: str, text_chunks: List[Dict], 
                                    image_chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """Tìm kiếm hybrid sử dụng Vintern cho cả text và image"""
        try:
            relevant_chunks = []
            
            # 1. Tìm text chunks liên quan bằng AI method
            if text_chunks:
                text_relevant = self._find_relevant_chunks_ai(question, text_chunks, top_k=top_k//2 + 1)
                relevant_chunks.extend(text_relevant)
            
            # 2. Tìm image chunks liên quan bằng Vintern similarity
            if image_chunks and self.vintern:
                try:
                    # Encode query
                    query_embedding = self.vintern.encode_query(question)
                    
                    # Load embeddings từ database
                    doc_embeddings = []
                    valid_image_chunks = []
                    
                    for chunk in image_chunks:
                        chunk_id = chunk.get('chunk_id') or chunk.get('_id')
                        if not chunk_id:
                            continue
                        
                        # Lấy embedding_data từ database
                        embedding_data = db.get_binary_field('pdf_chunks', {'chunk_id': chunk_id}, 'embedding_data')
                        if embedding_data is not None:
                            try:
                                # get_binary_field đã trả về bytes rồi, không cần convert lại
                                embedding = self.vintern.bytes_to_embedding(embedding_data)
                                doc_embeddings.append(embedding)
                                valid_image_chunks.append(chunk)
                            except Exception as e:
                                filename = chunk.get('filename', 'unknown')
                                page_num = chunk.get('page_number', chunk.get('chunk_index', 0) + 1 if chunk.get('chunk_index') is not None else '?')
                                logging.warning(f"⚠️ Không thể load embedding từ {filename} (trang {page_num}): {e}")
                        else:
                            # Log khi không có embedding_data
                            filename = chunk.get('filename', 'unknown')
                            page_num = chunk.get('page_number', chunk.get('chunk_index', 0) + 1 if chunk.get('chunk_index') is not None else '?')
                            logging.debug(f"⚠️ Chunk {filename} (trang {page_num}) không có embedding_data")
                    
                    if doc_embeddings:
                        # Tính similarity
                        scores = self.vintern.compute_similarity(query_embedding, doc_embeddings)
                        
                        # Lấy top image chunks
                        if len(scores.shape) > 0:
                            scores_list = scores.cpu().numpy().tolist()
                            if isinstance(scores_list[0], list):
                                scores_list = scores_list[0]
                            
                            # Sắp xếp theo score
                            scored_chunks = list(zip(valid_image_chunks, scores_list))
                            scored_chunks.sort(key=lambda x: x[1], reverse=True)

                            # Log top-n image chunks theo score để thấy toàn bộ quá trình
                            max_log = min(5, len(scored_chunks))
                            logging.info(f"📊 Top {max_log} image chunks theo similarity cho câu hỏi: \"{question}\"")
                            for rank, (chunk, score) in enumerate(scored_chunks[:max_log], start=1):
                                filename = chunk.get('filename', 'unknown')
                                page_num = chunk.get('page_number', chunk.get('chunk_index', 0) + 1 if chunk.get('chunk_index') is not None else '?')
                                chunk_idx = chunk.get('chunk_index', '?')
                                logging.info(f"  #{rank}: {filename} (chunk_index={chunk_idx}, trang {page_num}) - score: {score:.4f}")
                            
                            # Lấy top image chunks thực sự dùng cho trả lời
                            top_image_count = min(top_k - len(relevant_chunks), len(scored_chunks))
                            for chunk, score in scored_chunks[:top_image_count]:
                                relevant_chunks.append(chunk)
                                # Vẫn giữ log ngắn gọn cho các chunk cuối cùng được chọn
                                filename = chunk.get('filename', 'unknown')
                                page_num = chunk.get('page_number', chunk.get('chunk_index', 0) + 1 if chunk.get('chunk_index') is not None else '?')
                                logging.info(f"📸 Image chunk được chọn: {filename} (trang {page_num}) - score: {score:.4f}")
                except Exception as e:
                    logging.warning(f"⚠️ Lỗi tìm kiếm image chunks: {e}")
                    # Fallback: thêm một vài image chunks đầu tiên
                    relevant_chunks.extend(image_chunks[:top_k - len(relevant_chunks)])
            
            # Giới hạn số lượng chunks
            return relevant_chunks[:top_k]
            
        except Exception as e:
            logging.error(f"Lỗi hybrid search: {e}")
            # Fallback: kết hợp text và image chunks
            combined = text_chunks[:top_k//2] + image_chunks[:top_k//2]
            return combined[:top_k]