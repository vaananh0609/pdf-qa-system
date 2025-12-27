from __future__ import annotations

import PyPDF2
import os
import hashlib
from datetime import datetime
import re
from typing import List, Dict
import google.generativeai as genai
from config import Config
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import logging
try:
    import underthesea
except Exception:
    underthesea = None
import json
try:
    import fitz  # PyMuPDF để chuyển PDF thành ảnh
except ImportError:
    fitz = None
try:
    from PIL import Image
except ImportError:
    Image = None

class PDFProcessor:
    def __init__(self):
        self.chunk_size = 1000  # Số ký tự mỗi chunk (fallback)
        self.chunk_overlap = 200  # Số ký tự overlap giữa các chunk
        # Khởi tạo Gemini cho semantic chunking (nếu có key từ environment)
        try:
            primary_key = None
            keys = getattr(Config, "GEMINI_API_KEYS", None)
            if isinstance(keys, list) and keys:
                primary_key = keys[0]
            else:
                primary_key = getattr(Config, "GEMINI_API_KEY", None)

            if primary_key:
                genai.configure(api_key=primary_key)
                self.model = genai.GenerativeModel('gemini-2.5-pro')
            else:
                self.model = None
        except Exception as e:
            logging.warning(f"Lỗi khởi tạo Gemini cho PDFProcessor: {e}")
            self.model = None
        # Delay embedding model initialization to avoid slow startup
        self.embedding_model = None
        # Vietnamese sentence tokenizer
        self.use_underthesea = underthesea is not None
        # Whether to use AI to return offsets (fallback khi embedding lỗi)
        # self.use_ai_offsets = True  # Commented out, no AI fallback
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Trích xuất text từ file PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text += page.extract_text() + "\n"
                
                return text.strip()
        except Exception as e:
            logging.error(f"Lỗi đọc PDF: {e}")
            return ""

    def _repair_extraction_artifacts(self, text: str) -> str:
        """Sửa artifact từ trình trích xuất PDF"""
        if not text:
            return text

        # 1) Remove spaces between digits
        text = re.sub(r'(?<=\d) (?=\d)', '', text)

        # 2) Remove spaces after dot when between digits
        text = re.sub(r'(?<=\d)\. (?=\d)', '.', text)

        # 3) Join single-letter token with following token when safe: pattern (word1) (single_letter) (word2)
        # We only join single_letter with word2 (keep space before single_letter) to fix cases like 'Hà N ội' -> 'Hà Nội'
        def _join_single_letter(match):
            g1 = match.group(1)
            single = match.group(2)
            g3 = match.group(3)
            return f"{g1} {single}{g3}"

        text = re.sub(r"(\b\w+)\s+(\w)\s+(\w{2,})", _join_single_letter, text)

        return text
    
    def create_semantic_chunks(self, text: str, filename: str) -> List[Dict]:
        """Tạo semantic chunks từ text sử dụng embedding để giữ nguyên nội dung gốc.

        Ý tưởng:
        - Không thay đổi (clean) văn bản gốc khi tạo chunk; chỉ dùng văn bản gốc để tạo các substring
        - Tách văn bản thành câu kèm span (start/end)
        - Tính embedding cho mỗi câu và gom các câu liên tiếp có ý nghĩa thành chunk dựa trên similarity
        - Mỗi chunk là substring nguyên vẹn của văn bản gốc (char_start/char_end chính xác)
        """
        chunks: List[Dict] = []

        original_text = text or ""
        if not original_text:
            return chunks

        # Nếu quá ngắn (< 500 ký tự), tạo 1 chunk duy nhất
        if len(original_text) < 500:
            chunk_data = {
                'filename': filename,
                'chunk_index': 0,
                'text': original_text,
                'page_number': 1,
                'char_start': 0,
                'char_end': len(original_text),
                'created_at': datetime.now(),
                'chunk_id': hashlib.md5(f"{filename}_0".encode()).hexdigest()
            }
            chunks.append(chunk_data)
            logging.info(f"✅ Văn bản quá ngắn, tạo 1 chunk duy nhất cho file {filename}")
            return chunks

        estimated_pages = max(1, len(original_text) // 2000)

        # Thử size-based chunking trước (chính)
        try:
            logging.info(f"🔍 Bắt đầu chunking bằng kích thước cho file {filename}")
            # Tách thành câu kèm vị trí (span)
            sentences = self._split_into_sentences_with_spans(original_text)
            if not sentences:
                raise Exception("Không thể tách thành câu")

            logging.info(f"📊 Tách được {len(sentences)} câu cho file {filename}")

            # Ensure embedding model is loaded (lazy) - cần cho việc embed chunks
            if self.embedding_model is None:
                try:
                    self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                    logging.info("✅ Embedding model loaded thành công")
                except Exception as e:
                    logging.warning(f"Không thể khởi tạo embedding model: {e}")
                    raise Exception("Embedding model không khả dụng")

            # Gom các câu liên tiếp thành chunk dựa trên kích thước ký tự
            min_chars = 500
            max_chars = 1000

            chunk_index = 0
            i = 0
            n = len(sentences)

            while i < n:
                # Start a new chunk at sentence i
                start_span = sentences[i]['start']
                end_span = sentences[i]['end']
                curr_chars = len(sentences[i]['text'])

                j = i + 1
                while j < n:
                    next_len = len(sentences[j]['text'])
                    if curr_chars + next_len > max_chars:
                        if curr_chars >= min_chars:
                            break  # Dừng, tạo chunk hiện tại
                        else:
                            # Vẫn thêm để đạt min_chars, dù vượt max_chars một chút
                            logging.info(f"  ➕ Thêm câu {j} để đạt min_chars, chars={curr_chars}+{next_len}={curr_chars+next_len} (> {max_chars})")
                            end_span = sentences[j]['end']
                            curr_chars += next_len
                            j += 1
                    else:
                        # Thêm câu bình thường
                        logging.info(f"  ➕ Thêm câu {j}, chars={curr_chars}+{next_len}={curr_chars+next_len}")
                        end_span = sentences[j]['end']
                        curr_chars += next_len
                        j += 1

                # Điều chỉnh span để không cắt token
                if 'prev_end' not in locals():
                    prev_end = 0
                adj_start, adj_end = self._adjust_span_to_token_boundary(original_text, start_span, end_span)
                if adj_start < prev_end:
                    adj_start = prev_end
                if adj_end <= adj_start:
                    adj_start, adj_end = start_span, end_span

                chunk_text = original_text[adj_start:adj_end]

                # Embed chunk text để có vector cho retrieval
                try:
                    chunk_embedding = self.embedding_model.encode([chunk_text], convert_to_numpy=True)[0]
                    chunk_embedding_list = chunk_embedding.tolist()
                    logging.info(f"  📥 Embedded chunk {chunk_index}: {len(chunk_embedding_list)} dims")
                except Exception as e:
                    logging.warning(f"Lỗi embed chunk {chunk_index}: {e}, bỏ qua embedding")
                    chunk_embedding_list = []

                chunk_data = {
                    'filename': filename,
                    'chunk_index': chunk_index,
                    'text': chunk_text,
                    'embedding': chunk_embedding_list,  # Lưu vector embedding
                    'page_number': min(chunk_index + 1, estimated_pages),
                    'char_start': adj_start,
                    'char_end': adj_end,
                    'created_at': datetime.now(),
                    'chunk_id': hashlib.md5(f"{filename}_{chunk_index}".encode()).hexdigest()
                }
                chunks.append(chunk_data)
                logging.info(f"  ✅ Tạo chunk {chunk_index}: {len(chunk_text)} ký tự (câu {i} đến {j-1})")
                chunk_index += 1
                prev_end = adj_end
                i = j

            # Sanity check: đảm bảo không mất ký tự
            total_chunk_chars = sum(len(c['text']) for c in chunks)
            if total_chunk_chars != len(original_text):
                logging.warning(f"Tổng ký tự chunks ({total_chunk_chars}) != ký tự gốc ({len(original_text)})")

            logging.info(f"✅ Chunking bằng kích thước thành công: {len(chunks)} chunks cho file {filename}")
            return chunks

        except Exception as e:
            logging.error(f"❌ Lỗi size-based chunking cho file {filename}: {e}")
            raise  # Không fallback, raise exception

        # # Fallback 1: AI-based offsets nếu embedding lỗi
        # if self.model and self.use_ai_offsets:
        #     try:
        #         logging.info(f"🤖 Bắt đầu chunking bằng AI fallback cho file {filename}")
        #         offsets = self._get_semantic_offsets_from_ai(original_text)
        #         if offsets and isinstance(offsets, list):
        #             # Validate and build chunks from offsets
        #             chunks = []
        #             prev_end = 0
        #             chunk_index = 0
        #             for off in offsets:
        #                 if not isinstance(off, dict):
        #                     continue
        #                 s = int(off.get('start', 0))
        #                 e = int(off.get('end', 0))
        #                 # clamp
        #                 s = max(0, min(s, len(original_text)))
        #                 e = max(0, min(e, len(original_text)))
        #                 if e <= s:
        #                     continue
        #                 # If gap exists before this offset, fill gap with a chunk (non-overlap)
        #                 if s > prev_end:
        #                     gap_text = original_text[prev_end:s]
        #                     chunks.append({
        #                         'filename': filename,
        #                         'chunk_index': chunk_index,
        #                         'text': gap_text,
        #                         'page_number': min(chunk_index + 1, estimated_pages),
        #                         'char_start': prev_end,
        #                         'char_end': s,
        #                         'created_at': datetime.now(),
        #                         'chunk_id': hashlib.md5(f"{filename}_{chunk_index}".encode()).hexdigest()
        #                     })
        #                     chunk_index += 1

        #                 chunk_text = original_text[s:e]
        #                 chunks.append({
        #                     'filename': filename,
        #                     'chunk_index': chunk_index,
        #                     'text': chunk_text,
        #                     'page_number': min(chunk_index + 1, estimated_pages),
        #                         'char_start': s,
        #                         'char_end': e,
        #                         'created_at': datetime.now(),
        #                         'chunk_id': hashlib.md5(f"{filename}_{chunk_index}".encode()).hexdigest()
        #                     })
        #                     chunk_index += 1
        #                     prev_end = e

        #                 # If there's trailing text after last offset, include it
        #                 if prev_end < len(original_text):
        #                     tail = original_text[prev_end:]
        #                     chunks.append({
        #                         'filename': filename,
        #                         'chunk_index': chunk_index,
        #                         'text': tail,
        #                         'page_number': min(chunk_index + 1, estimated_pages),
        #                         'char_start': prev_end,
        #                         'char_end': len(original_text),
        #                         'created_at': datetime.now(),
        #                         'chunk_id': hashlib.md5(f"{filename}_{chunk_index}".encode()).hexdigest()
        #                     })

        #                 # Validate total coverage (no loss). If mismatch, fallback to fixed-size
        #                 total_chunk_chars = sum(len(c['text']) for c in chunks)
        #                 if total_chunk_chars == len(original_text):
        #                     logging.info(f"✅ Chunking bằng AI thành công: {len(chunks)} chunks cho file {filename}")
        #                     return chunks
        #                 else:
        #                     logging.warning(f"AI offsets coverage mismatch ({total_chunk_chars} != {len(original_text)}), falling back to fixed-size")
        #     except Exception as e:
        #         logging.warning(f"❌ Lỗi AI chunking cho file {filename}: {e}, dùng fixed-size")

        # # Fallback cuối cùng: fixed-size non-overlap
        # logging.info(f"📏 Bắt đầu chunking bằng fixed-size cho file {filename}")
        # chunks = self.create_fixed_size_chunks_nonoverlap(original_text, filename)
        # logging.info(f"✅ Chunking bằng fixed-size thành công: {len(chunks)} chunks cho file {filename}")
        # return chunks

    def _split_into_sentences_with_spans(self, text: str) -> List[Dict]:
        """Tách văn bản thành các câu với span (start/end)"""
        sentences: List[Dict] = []
        if not text:
            return sentences

        # If underthesea is available, use it for Vietnamese sentence tokenization
        if self.use_underthesea:
            try:
                v_sents = underthesea.sent_tokenize(text)
                # find spans by searching from last end to avoid mismatches
                cursor = 0
                for s in v_sents:
                    s_stripped = s.strip()
                    if not s_stripped:
                        continue
                    idx = text.find(s_stripped, cursor)
                    if idx == -1:
                        # fallback to regex method for remaining text
                        break
                    start = idx
                    end = idx + len(s_stripped)
                    sentences.append({'text': s_stripped, 'start': start, 'end': end})
                    cursor = end
                # If couldn't tokenize fully, fall back to regex for remaining parts
                if not sentences:
                    raise Exception('underthesea returned no sentences')
                return sentences
            except Exception:
                # fallback to regex below
                pass

        # Regex fallback: match up to a sentence end (., !, ?) or end of text. Use DOTALL to allow newlines.
        pattern = re.compile(r'.+?(?:[.!?]+|$)', re.S)
        for m in pattern.finditer(text):
            s = m.group(0)
            if s is None:
                continue
            if s.strip() == '':
                continue
            start, end = m.span()
            sentences.append({'text': s, 'start': start, 'end': end})

        return sentences

    def _adjust_span_to_token_boundary(self, text: str, start: int, end: int, max_extend: int = 50) -> tuple:
        """Điều chỉnh span để không cắt giữa token"""
        n = len(text)

        def is_boundary_char(ch):
            if ch.isspace():
                return True
            if ch in '.,;:!?()[]{}"\'"-–—/\\':
                return True
            return False

        new_start = start
        new_end = end

        # Adjust start backward if it is in the middle of an alnum token
        if 0 < new_start < n and text[new_start].isalnum() and text[new_start-1].isalnum():
            # try move backward to previous boundary within max_extend
            limit = max(0, new_start - max_extend)
            found = False
            i = new_start
            while i > limit:
                if is_boundary_char(text[i-1]):
                    new_start = i
                    found = True
                    break
                i -= 1
            if not found:
                # fallback: move to the leftmost within limit
                new_start = max(limit, 0)

        # Adjust end forward if it is in the middle of an alnum token
        if 0 <= new_end < n and text[new_end-1].isalnum() and text[new_end].isalnum():
            # try move forward to next boundary within max_extend
            limit = min(n, new_end + max_extend)
            found = False
            i = new_end
            while i < limit:
                if is_boundary_char(text[i]):
                    new_end = i
                    found = True
                    break
                i += 1
            if not found:
                # fallback: move backward to previous boundary
                j = new_end
                while j > max(0, new_end - max_extend):
                    if is_boundary_char(text[j-1]):
                        new_end = j
                        found = True
                        break
                    j -= 1
                if not found:
                    new_end = new_end  # give up; leave as is

        # clamp
        new_start = max(0, min(new_start, n))
        new_end = max(new_start, min(new_end, n))

        return new_start, new_end

    def create_fixed_size_chunks_nonoverlap(self, text: str, filename: str) -> List[Dict]:
        """Tạo chunks không chồng (non-overlap) từ văn bản gốc, không gọi clean_text.

        Dùng khi không muốn mất/đổi nội dung và cần fallback an toàn.
        """
        chunks = []
        original_text = text or ""
        if not original_text:
            return chunks

        estimated_pages = max(1, len(original_text) // 2000)
        start = 0
        chunk_index = 0

        while start < len(original_text):
            end = min(start + self.chunk_size, len(original_text))

            # Adjust only end to avoid splitting tokens for non-overlap fixed chunks
            _, new_end = self._adjust_span_to_token_boundary(original_text, start, end)
            if new_end <= start:
                new_end = end

            chunk_text = original_text[start:new_end]

            chunk_data = {
                'filename': filename,
                'chunk_index': chunk_index,
                'text': chunk_text,
                'page_number': min(chunk_index + 1, estimated_pages),
                'char_start': start,
                'char_end': new_end,
                'created_at': datetime.now(),
                'chunk_id': hashlib.md5(f"{filename}_{chunk_index}".encode()).hexdigest()
            }
            chunks.append(chunk_data)
            chunk_index += 1

            start = new_end

        return chunks

    def create_chunks(self, text: str, filename: str) -> List[Dict]:
        """API đơn giản để tạo chunks cho các đoạn text nhỏ (ví dụ theo trang)."""
        if not text:
            return []
        cleaned = text.strip()
        if not cleaned:
            return []
        return self.create_fixed_size_chunks_nonoverlap(cleaned, filename)
    
    def _get_semantic_offsets_from_ai(self, text: str) -> List[Dict]:
        """Yêu cầu AI trả về offsets (start/end) cho semantic chunks"""
        if not self.model:
            return []

        try:
            # Build a strict prompt asking for JSON output only
            prompt = f"""
You are an expert text segmenter. I will provide a text. Do NOT change the text.
Return a JSON array of objects where each object has two integer fields: start and end.
Each start/end must be a character offset (0-based) in the exact text I provide. Offsets should partition the text into meaningful semantic chunks (each chunk a contiguous substring). Do NOT output any explanatory text, only the JSON array.

Text:
{text}

Requirements:
- Output must be valid JSON array like [{"start":0,"end":123},{"start":123,"end":456},...]
- Offsets must be in increasing order, non-overlapping. It's OK to omit very small fragments but prefer full coverage.
"""

            response = self.model.generate_content(prompt)
            resp_text = response.text.strip()

            # Try to load JSON directly
            try:
                data = json.loads(resp_text)
                if isinstance(data, list):
                    # sanitize items
                    out = []
                    for item in data:
                        if isinstance(item, dict) and 'start' in item and 'end' in item:
                            out.append({'start': int(item['start']), 'end': int(item['end'])})
                    return out
            except Exception:
                pass

            # Try to extract a JSON substring from the response
            jstart = resp_text.find('[')
            jend = resp_text.rfind(']')
            if jstart != -1 and jend != -1 and jend > jstart:
                sub = resp_text[jstart:jend+1]
                try:
                    data = json.loads(sub)
                    out = []
                    for item in data:
                        if isinstance(item, dict) and 'start' in item and 'end' in item:
                            out.append({'start': int(item['start']), 'end': int(item['end'])})
                    return out
                except Exception:
                    pass

            # Fallback: try to parse numbers in the response as pairs
            import re
            nums = re.findall(r'\d+', resp_text)
            if nums and len(nums) >= 2:
                pairs = []
                try:
                    it = iter(nums)
                    while True:
                        s = int(next(it))
                        e = int(next(it))
                        pairs.append({'start': s, 'end': e})
                except StopIteration:
                    pass
                return pairs

            return []

        except Exception as e:
            logging.warning(f"Lỗi khi gọi AI trả offsets: {e}")
            return []
    
    def process_pdf_file(self, pdf_path: str, caption: str = '') -> Dict:
        """Xử lý file PDF và tạo metadata"""
        try:
            filename = os.path.basename(pdf_path)
            
            # Tính hash của file (nếu chưa có trong metadata)
            file_hash = ""
            try:
                hash_md5 = hashlib.md5()
                with open(pdf_path, "rb") as f:
                    for chunk in iter(lambda: f.read(4096), b""):
                        hash_md5.update(chunk)
                file_hash = hash_md5.hexdigest()
            except Exception as e:
                logging.warning(f"Không thể tính hash file {pdf_path}: {e}")
            
            # Trích xuất text
            text = self.extract_text_from_pdf(pdf_path)
            # Repair common extraction artifacts before chunking
            text = self._repair_extraction_artifacts(text)
            
            if not text:
                return None
            
            # Tạo chunks
            chunks = self.create_semantic_chunks(text, filename)
            
            # Tạo metadata
            total_chunk_chars = sum(len(c.get('text', '')) for c in chunks)
            char_mismatch = len(text) - total_chunk_chars

            metadata = {
                'filename': filename,
                'file_path': pdf_path,
                'file_size': os.path.getsize(pdf_path),
                'file_hash': file_hash,  # Thêm file_hash vào metadata
                'caption': caption,
                'total_chunks': len(chunks),
                'total_text_length': len(text),
                'total_chunk_chars': total_chunk_chars,
                'char_mismatch': char_mismatch,
                'created_at': datetime.now(),
                'processed': True,
                'chunking_strategy': 'size-based-with-embedding',  # Dựa trên kích thước, embed từng chunk
                'file_id': hashlib.md5(filename.encode()).hexdigest()
            }
            
            return {
                'metadata': metadata,
                'chunks': chunks
            }
            
        except Exception as e:
            logging.error(f"Lỗi xử lý PDF {pdf_path}: {e}")
            return None
    
    def analyze_pdf_pages(self, pdf_path: str, threshold_chars: int = 50) -> List[Dict]:
        """
        Phân tích từng trang: trả về list các dict {page_num, text, is_text}.
        
        Args:
            pdf_path: Đường dẫn file PDF
            threshold_chars: Ngưỡng số ký tự để phân loại trang là text hay image
            
        Returns:
            List[Dict] với mỗi dict có:
                - page_num: số trang (0-based)
                - text: text trích xuất được
                - is_text: True nếu là trang text, False nếu là trang image
        """
        page_infos = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    text_length = len(text.strip())
                    
                    is_text = text_length >= threshold_chars
                    
                    page_infos.append({
                        'page_num': page_num,
                        'text': text,
                        'is_text': is_text,
                        'text_length': text_length
                    })
        except Exception as e:
            logging.error(f"Lỗi phân tích PDF pages: {e}")
        
        return page_infos
    
    def convert_pdf_page_to_image(self, pdf_path: str, page_num: int, zoom: float = 1.5):
        """Chuyển 1 trang PDF thành PIL Image (page_num là 0-based, tối ưu tốc độ)."""
        if not fitz or not Image:
            raise ImportError("PyMuPDF (fitz) và Pillow (PIL) cần được cài đặt để chuyển PDF thành ảnh")
        
        try:
            doc = fitz.open(pdf_path)
            page = doc.load_page(page_num)
            
            # Render page thành pixmap với độ phân giải tối ưu (giảm zoom để tăng tốc)
            matrix = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=matrix)
            
            # Chuyển pixmap thành PIL Image
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            doc.close()
            return img
        except Exception as e:
            logging.error(f"Lỗi chuyển PDF page {page_num} thành ảnh: {e}")
            raise
    
    def convert_pdf_pages_to_images(self, pdf_path: str, max_pages: int = 200, max_size: int = 1600) -> List:
        """
        Chuyển đổi từng trang PDF thành ảnh PIL (tối ưu tốc độ)
        
        Args:
            pdf_path: Đường dẫn file PDF
            max_pages: Số trang tối đa để xử lý
            max_size: Kích thước tối đa của ảnh (width hoặc height) để tối ưu tốc độ
            
        Returns:
            List các PIL Image (đã được resize nếu cần)
        """
        if not fitz or not Image:
            raise ImportError("PyMuPDF (fitz) và Pillow (PIL) cần được cài đặt")
        
        images = []
        try:
            # Mở PDF bằng PyMuPDF (fitz)
            doc = fitz.open(pdf_path)
            total_pages = min(doc.page_count, max_pages)
            logging.info(f"📄 Đang chuyển đổi {total_pages} trang PDF thành ảnh...")
            
            # Giảm zoom từ 2x xuống 1.5x để tăng tốc độ
            zoom = 1.5
            matrix = fitz.Matrix(zoom, zoom)
            
            for page_num in range(total_pages):
                page = doc.load_page(page_num)
                
                # Render page thành pixmap với độ phân giải tối ưu
                pix = page.get_pixmap(matrix=matrix)
                
                # Chuyển pixmap thành PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Resize ảnh nếu quá lớn để tối ưu tốc độ encode và lưu trữ
                if img.width > max_size or img.height > max_size:
                    img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                
                images.append(img)
                
                # Progress logging mỗi 5 trang
                if (page_num + 1) % 5 == 0 or (page_num + 1) == total_pages:
                    logging.info(f"  ⏳ Đã xử lý {page_num + 1}/{total_pages} trang...")
            
            doc.close()
            logging.info(f"✅ Đã chuyển đổi {len(images)} trang thành ảnh (kích thước tối đa: {max_size}px)")
            return images
        except Exception as e:
            logging.error(f"❌ Lỗi chuyển PDF thành ảnh: {e}")
            raise
    
    def image_to_bytes(self, image, format: str = 'JPEG', quality: int = 85) -> bytes:
        """
        Chuyển PIL Image thành bytes để lưu vào MongoDB (tối ưu kích thước)
        
        Args:
            image: PIL Image
            format: Format ảnh (JPEG để tiết kiệm dung lượng, PNG nếu cần chất lượng cao)
            quality: Chất lượng JPEG (1-100, mặc định 85)
            
        Returns:
            bytes
        """
        if not Image:
            raise ImportError("Pillow (PIL) cần được cài đặt")
        
        buffer = __import__('io').BytesIO()
        if format.upper() == 'JPEG':
            # Convert RGBA to RGB nếu cần (JPEG không hỗ trợ alpha)
            if image.mode == 'RGBA':
                rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                rgb_image.paste(image, mask=image.split()[3])  # Use alpha channel as mask
                image = rgb_image
            image.save(buffer, format=format, quality=quality, optimize=True)
        else:
            image.save(buffer, format=format, optimize=True)
        return buffer.getvalue()
    
    def bytes_to_image(self, data: bytes):
        """
        Chuyển bytes thành PIL Image
        
        Args:
            data: bytes data
            
        Returns:
            PIL Image
        """
        if not Image:
            raise ImportError("Pillow (PIL) cần được cài đặt")
        
        buffer = __import__('io').BytesIO(data)
        return Image.open(buffer)
