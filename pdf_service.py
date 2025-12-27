import os
import shutil
import time
from datetime import datetime
from typing import List, Dict
from pdf_processor import PDFProcessor
from database import db
from gemini_service import GeminiService
from vintern_client import get_vintern_client
from config import Config
import logging
import hashlib

class PDFService:
    def __init__(self):
        self.processor = PDFProcessor()
        self.gemini = GeminiService()
        self.upload_folder = 'uploads'
        
        # Tạo thư mục uploads nếu chưa có
        if not os.path.exists(self.upload_folder):
            os.makedirs(self.upload_folder)

    def _log_event(self, event_type: str, details: Dict):
        """Lưu log hệ thống vào collection system_logs."""
        try:
            log_entry = {
                'event_type': event_type,
                'timestamp': datetime.now(),
                'details': details
            }
            db.insert_document('system_logs', log_entry)
        except Exception as log_error:
            logging.warning(f"Không thể ghi log {event_type}: {log_error}")
    
    def upload_pdf(self, file, filename: str, caption: str = '') -> Dict:
        """Upload và xử lý file PDF - hỗ trợ cả PDF text và PDF scanned"""
        operation_start = time.time()
        try:
            print(f"Bắt đầu upload_pdf: {filename}")
            
            # Lưu file vào thư mục uploads
            file_path = os.path.join(self.upload_folder, filename)
            print(f"Lưu file vào: {file_path}")
            file.save(file_path)
            
            # Phân tích từng trang để hỗ trợ file mixed (text + scanned)
            page_infos = self.processor.analyze_pdf_pages(file_path)
            total_pages = len(page_infos)
            text_pages = sum(1 for p in page_infos if p.get('is_text'))
            image_pages = total_pages - text_pages

            if total_pages == 0:
                return {'success': False, 'message': 'Không thể đọc file PDF hoặc file rỗng'}

            if text_pages == 0:
                print(f"📷 PDF Scanned detected (toàn bộ) - Sử dụng Vintern")
                pdf_mode = 'scanned'
                result = self._upload_scanned_pdf(file_path, filename, caption)
            elif image_pages == 0:
                print(f"📝 PDF Text detected (toàn bộ) - Sử dụng semantic chunking")
                pdf_mode = 'text'
                result = self._upload_text_pdf(file_path, filename, caption)
            else:
                print(f"🔀 PDF Mixed detected - xử lý từng trang (text: {text_pages}, image: {image_pages})")
                pdf_mode = 'mixed'
                result = self._upload_mixed_pdf(file_path, filename, caption, page_infos)

            duration_ms = int((time.time() - operation_start) * 1000)
            self._log_event('upload', {
                'filename': filename,
                'caption': caption,
                'pdf_mode_detected': pdf_mode,
                'total_pages': total_pages,
                'text_pages': text_pages,
                'image_pages': image_pages,
                'duration_ms': duration_ms,
                'success': result.get('success'),
                'message': result.get('message'),
                'metadata_id': result.get('metadata_id')
            })
            return result
            
        except Exception as e:
            # Xóa file nếu có lỗi
            file_path = os.path.join(self.upload_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Lỗi upload PDF: {e}")
            duration_ms = int((time.time() - operation_start) * 1000)
            self._log_event('upload', {
                'filename': filename,
                'caption': caption,
                'pdf_mode_detected': 'unknown',
                'duration_ms': duration_ms,
                'success': False,
                'error': str(e)
            })
            return {'success': False, 'message': f'Lỗi upload: {str(e)}'}
    
    def _upload_text_pdf(self, file_path: str, filename: str, caption: str) -> Dict:
        """Xử lý PDF text thường - giữ nguyên logic cũ"""
        try:
            # Xử lý PDF
            print(f"Bắt đầu xử lý PDF text: {filename}")
            result = self.processor.process_pdf_file(file_path, caption)
            
            if not result:
                print(f"Lỗi xử lý PDF: {filename}")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return {'success': False, 'message': 'Không thể xử lý file PDF'}
            
            metadata = result['metadata']
            metadata['pdf_type'] = 'text'  # Đánh dấu loại PDF
            chunks = result['chunks']
            
            # Lưu metadata vào database
            metadata_id = db.insert_document('pdf_files', metadata)
            
            # Lưu chunks vào database
            chunk_ids = []
            for chunk in chunks:
                chunk['metadata_id'] = metadata_id
                chunk['chunk_type'] = 'text'  # Đánh dấu loại chunk
                chunk_id = db.insert_document('pdf_chunks', chunk)
                chunk_ids.append(chunk_id)
            
            return {
                'success': True,
                'message': 'Upload PDF text thành công',
                'metadata_id': str(metadata_id),
                'chunk_ids': [str(chunk_id) for chunk_id in chunk_ids],
                'total_chunks': len(chunks),
                'pdf_type': 'text'
            }
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Lỗi xử lý PDF text: {e}")
            return {'success': False, 'message': f'Lỗi upload PDF text: {str(e)}'}
    
    def _upload_scanned_pdf(self, file_path: str, filename: str, caption: str) -> Dict:
        """Xử lý PDF scanned - chuyển thành ảnh và dùng Vintern embedding"""
        try:
            print(f"Bắt đầu xử lý PDF scanned: {filename}")
            
            # Chuyển đổi PDF pages thành images
            images = self.processor.convert_pdf_pages_to_images(file_path, max_pages=200)
            
            if not images:
                if os.path.exists(file_path):
                    os.remove(file_path)
                return {'success': False, 'message': 'Không thể chuyển đổi PDF thành ảnh'}
            
            # Khởi tạo Vintern client (gọi API Colab)
            vintern = get_vintern_client()
            
            # Set API URL nếu có trong config
            if Config.VINTERN_API_URL:
                vintern.set_api_url(Config.VINTERN_API_URL)
            
            if not vintern.is_available():
                logging.warning("Vintern API không khả dụng, fallback về xử lý text")
                return self._upload_text_pdf(file_path, filename, caption)
            
            # Tạo embeddings cho các ảnh (batch processing)
            print(f"Đang tạo embeddings cho {len(images)} trang...")
            
            # CPU: giảm batch size để tránh out of memory
            # GPU: có thể xử lý batch lớn hơn
            import torch
            batch_size = 2 if not torch.cuda.is_available() else 8
            
            if not torch.cuda.is_available():
                print(f"⚠️ Đang chạy trên CPU - Quá trình này có thể mất 5-10 phút cho {len(images)} trang")
            
            all_embeddings = []
            
            for i in range(0, len(images), batch_size):
                batch_images = images[i:i + batch_size]
                progress = f"[{i+len(batch_images)}/{len(images)}]"
                print(f"  {progress} Đang xử lý batch...")
                
                batch_embeddings = vintern.encode_images(batch_images)
                
                # Chuyển từng embedding thành list
                for j in range(len(batch_images)):
                    all_embeddings.append(batch_embeddings[j])
            
            # Tạo metadata
            metadata = {
                'filename': filename,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'caption': caption,
                'total_chunks': len(images),
                'total_pages': len(images),
                'pdf_type': 'scanned',  # Đánh dấu loại PDF
                'created_at': datetime.now(),
                'processed': True,
                'chunking_strategy': 'image-embedding',
                'file_id': hashlib.md5(filename.encode()).hexdigest()
            }
            
            # Lưu metadata vào database
            metadata_id = db.insert_document('pdf_files', metadata)
            
            # Lưu từng trang (ảnh + embedding) vào database
            chunk_ids = []
            for page_num, (image, embedding) in enumerate(zip(images, all_embeddings)):
                # Chuyển image thành bytes
                image_bytes = self.processor.image_to_bytes(image, format='JPEG')
                
                # Chuyển embedding thành bytes
                embedding_bytes = vintern.embedding_to_bytes(embedding)
                
                chunk_data = {
                    'metadata_id': metadata_id,
                    'filename': filename,
                    'chunk_index': page_num,
                    'page_number': page_num + 1,
                    'chunk_type': 'image',  # Đánh dấu loại chunk
                    'image_data': image_bytes,  # Lưu ảnh
                    'embedding_data': embedding_bytes,  # Lưu embedding
                    'created_at': datetime.now(),
                    'chunk_id': hashlib.md5(f"{filename}_{page_num}".encode()).hexdigest()
                }
                
                chunk_id = db.insert_document('pdf_chunks', chunk_data)
                chunk_ids.append(chunk_id)
            
            print(f"✅ Đã lưu {len(chunk_ids)} trang vào database")
            
            return {
                'success': True,
                'message': 'Upload PDF scanned thành công',
                'metadata_id': str(metadata_id),
                'chunk_ids': [str(chunk_id) for chunk_id in chunk_ids],
                'total_chunks': len(chunk_ids),
                'pdf_type': 'scanned'
            }
            
        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Lỗi xử lý PDF scanned: {e}")
            return {'success': False, 'message': f'Lỗi upload PDF scanned: {str(e)}'}
    
    def _upload_mixed_pdf(self, file_path: str, filename: str, caption: str, page_infos: List[Dict]) -> Dict:
        """Xử lý PDF mixed: trang text sẽ dùng semantic chunking, trang image sẽ dùng Vintern embedding"""
        try:
            print(f"Bắt đầu xử lý PDF mixed: {filename}")

            # Khởi tạo Vintern client nếu có trang image
            vintern = None
            if any(not p.get('is_text') for p in page_infos):
                vintern = get_vintern_client()
                if Config.VINTERN_API_URL:
                    vintern.set_api_url(Config.VINTERN_API_URL)
                if not vintern.is_available():
                    logging.warning("Vintern API không khả dụng, các trang ảnh sẽ bị bỏ qua hoặc fallback về OCR nếu implement")

            # Prepare containers
            text_chunks_all = []
            image_pages = []

            # Process text pages: create chunks per page and record
            global_chunk_index = 0
            for p in page_infos:
                if p.get('is_text'):
                    page_num = p.get('page_num')
                    page_text = p.get('text', '') or ''
                    page_text = self.processor._repair_extraction_artifacts(page_text)
                    if not page_text:
                        continue
                    page_chunks = self.processor.create_chunks(page_text, filename)
                    # Normalize and annotate chunks
                    for c in page_chunks:
                        c['filename'] = filename
                        c['page_number'] = page_num + 1
                        c['chunk_index'] = global_chunk_index
                        c['chunk_type'] = 'text'
                        c['created_at'] = c.get('created_at') or datetime.now()
                        c['chunk_id'] = c.get('chunk_id') or hashlib.md5(f"{filename}_{global_chunk_index}".encode()).hexdigest()
                        text_chunks_all.append(c)
                        global_chunk_index += 1
                else:
                    image_pages.append(p.get('page_num'))

            # Process image pages: convert to images and create embeddings via Vintern
            image_chunks = []
            all_embeddings = []
            images_for_encoding = []
            image_page_order = []

            for page_num in image_pages:
                img = self.processor.convert_pdf_page_to_image(file_path, page_num)
                if img is not None:
                    images_for_encoding.append(img)
                    image_page_order.append(page_num)

            if images_for_encoding and vintern and vintern.is_available():
                import torch
                batch_size = 2 if not torch.cuda.is_available() else 8
                for i in range(0, len(images_for_encoding), batch_size):
                    batch = images_for_encoding[i:i+batch_size]
                    batch_embeddings = vintern.encode_images(batch)
                    for emb in batch_embeddings:
                        all_embeddings.append(emb)

            # Build image chunk records (embedding may be missing if vintern unavailable)
            for idx, page_num in enumerate(image_page_order):
                img = images_for_encoding[idx]
                embedding = all_embeddings[idx] if idx < len(all_embeddings) else None

                image_bytes = self.processor.image_to_bytes(img, format='JPEG') if img is not None else None
                embedding_bytes = vintern.embedding_to_bytes(embedding) if (embedding is not None and vintern) else None

                chunk_data = {
                    'filename': filename,
                    'chunk_index': global_chunk_index,
                    'page_number': page_num + 1,
                    'chunk_type': 'image',
                    'image_data': image_bytes,
                    'embedding_data': embedding_bytes,
                    'created_at': datetime.now(),
                    'chunk_id': hashlib.md5(f"{filename}_{global_chunk_index}".encode()).hexdigest()
                }

                image_chunks.append(chunk_data)
                global_chunk_index += 1

            # Combine all chunks (text first then image) and save
            total_chunks = len(text_chunks_all) + len(image_chunks)
            metadata = {
                'filename': filename,
                'file_path': file_path,
                'file_size': os.path.getsize(file_path),
                'caption': caption,
                'total_chunks': total_chunks,
                'total_pages': len(page_infos),
                'total_text_pages': sum(1 for p in page_infos if p.get('is_text')),
                'total_image_pages': sum(1 for p in page_infos if not p.get('is_text')),
                'pdf_type': 'mixed',
                'created_at': datetime.now(),
                'processed': True,
                'chunking_strategy': 'mixed',
                'file_id': hashlib.md5(filename.encode()).hexdigest()
            }

            metadata_id = db.insert_document('pdf_files', metadata)

            chunk_ids = []
            # Insert text chunks
            for c in text_chunks_all:
                c['metadata_id'] = metadata_id
                cid = db.insert_document('pdf_chunks', c)
                chunk_ids.append(cid)

            # Insert image chunks
            for c in image_chunks:
                c['metadata_id'] = metadata_id
                cid = db.insert_document('pdf_chunks', c)
                chunk_ids.append(cid)

            print(f"✅ Đã lưu mixed PDF: {len(chunk_ids)} chunks (text {len(text_chunks_all)}, image {len(image_chunks)})")

            return {
                'success': True,
                'message': 'Upload PDF mixed thành công',
                'metadata_id': str(metadata_id),
                'chunk_ids': [str(chunk_id) for chunk_id in chunk_ids],
                'total_chunks': len(chunk_ids),
                'pdf_type': 'mixed'
            }

        except Exception as e:
            if os.path.exists(file_path):
                os.remove(file_path)
            logging.error(f"Lỗi xử lý PDF mixed: {e}")
            return {'success': False, 'message': f'Lỗi upload PDF mixed: {str(e)}'}
    
    
    def get_all_pdfs(self) -> List[Dict]:
        """Lấy danh sách tất cả PDF files (sắp xếp theo thời gian mới nhất)"""
        try:
            pdfs = db.find_documents('pdf_files', {'processed': True}, sort=[('created_at', -1)])
            return pdfs
        except Exception as e:
            print(f"Lỗi lấy danh sách PDF: {e}")
            return []
    
    def delete_pdf(self, filename: str) -> Dict:
        """Xóa PDF file và các chunks liên quan"""
        try:
            # Tìm metadata của file
            pdf_file = db.find_documents('pdf_files', {'filename': filename})
            
            if not pdf_file:
                return {'success': False, 'message': 'File không tồn tại'}
            
            metadata_id = pdf_file[0]['_id']
            
            # Xóa tất cả chunks liên quan (delete_many)
            db.delete_documents('pdf_chunks', {'metadata_id': metadata_id})
            
            # Xóa metadata
            db.delete_document('pdf_files', {'_id': metadata_id})
            
            # Xóa file vật lý (ưu tiên đường dẫn lưu trong metadata)
            file_path = pdf_file[0].get('file_path') or os.path.join(self.upload_folder, filename)
            try:
                if file_path and os.path.exists(file_path):
                    os.remove(file_path)
            except Exception:
                # Bỏ qua lỗi xóa file vật lý để không chặn việc xóa DB
                pass
            
            return {'success': True, 'message': 'Đã xóa file thành công'}
            
        except Exception as e:
            return {'success': False, 'message': f'Lỗi xóa file: {str(e)}'}
    
    def search_and_answer(self, question: str, filename: str = None) -> Dict:
        """Tìm kiếm và trả lời câu hỏi"""
        query_start = time.time()
        try:
            # Tìm tất cả chunks (hoặc chunks của file cụ thể)
            query = {}
            if filename:
                query['filename'] = filename
            
            all_chunks = db.find_documents('pdf_chunks', query)
            
            if not all_chunks:
                duration_ms = int((time.time() - query_start) * 1000)
                self._log_event('query', {
                    'question': question,
                    'filename_filter': filename,
                    'total_chunks_scanned': 0,
                    'relevant_chunks': 0,
                    'duration_ms': duration_ms,
                    'success': False,
                    'reason': 'no_chunks'
                })
                return {
                    'success': False,
                    'message': 'Không tìm thấy tài liệu để tìm kiếm'
                }
            
            # Tìm chunks liên quan
            relevant_chunks = self.gemini.find_relevant_chunks(question, all_chunks)
            
            if not relevant_chunks:
                duration_ms = int((time.time() - query_start) * 1000)
                self._log_event('query', {
                    'question': question,
                    'filename_filter': filename,
                    'total_chunks_scanned': len(all_chunks),
                    'relevant_chunks': 0,
                    'duration_ms': duration_ms,
                    'success': False,
                    'reason': 'no_relevant_chunks'
                })
                return {
                    'success': False,
                    'message': 'Không tìm thấy thông tin liên quan đến câu hỏi'
                }
            
            # Tạo câu trả lời
            answer_result = self.gemini.generate_answer(question, relevant_chunks)
            duration_ms = int((time.time() - query_start) * 1000)
            filenames_used = list({chunk.get('filename') for chunk in relevant_chunks if chunk.get('filename')})
            self._log_event('query', {
                'question': question,
                'filename_filter': filename,
                'total_chunks_scanned': len(all_chunks),
                'relevant_chunks': len(relevant_chunks),
                'duration_ms': duration_ms,
                'success': True,
                'filenames_used': filenames_used
            })
            return {
                'success': True,
                'question': question,
                'answer': answer_result['answer'],
                'sources': answer_result['sources'],
                'relevant_chunks': len(relevant_chunks),
                'total_chunks_searched': len(all_chunks)
            }
            
        except Exception as e:
            duration_ms = int((time.time() - query_start) * 1000)
            self._log_event('query', {
                'question': question,
                'filename_filter': filename,
                'duration_ms': duration_ms,
                'success': False,
                'error': str(e)
            })
            return {
                'success': False,
                'message': f'Lỗi tìm kiếm: {str(e)}'
            }
    
    def get_pdf_content(self, filename: str) -> Dict:
        """Lấy nội dung PDF để hiển thị"""
        try:
            # Tìm metadata
            pdf_file = db.find_documents('pdf_files', {'filename': filename})
            
            if not pdf_file:
                return {'success': False, 'message': 'File không tồn tại'}
            
            # Lấy tất cả chunks của file (KHÔNG lấy binary data)
            chunks_raw = db.find_documents('pdf_chunks', {'filename': filename})
            
            # Xử lý chunks: loại bỏ binary data cho image chunks
            chunks = []
            for chunk in chunks_raw:
                chunk_type = chunk.get('chunk_type', 'text')
                
                if chunk_type == 'image':
                    # Image chunk: không trả về image_data và embedding_data nhưng có URL
                    chunks.append({
                        '_id': chunk.get('_id'),
                        'filename': chunk.get('filename'),
                        'chunk_index': chunk.get('chunk_index'),
                        'page_number': chunk.get('page_number'),
                        'chunk_type': 'image',
                        'text': f'[Trang {chunk.get("page_number")} - Ảnh từ PDF scanned]',
                        'image_url': f'/pdf_image/{chunk.get("chunk_id")}',  # URL để lấy ảnh
                        'created_at': chunk.get('created_at'),
                        'chunk_id': chunk.get('chunk_id')
                    })
                else:
                    # Text chunk: giữ nguyên
                    chunks.append(chunk)
            
            # Sắp xếp theo chunk_index
            chunks.sort(key=lambda x: x['chunk_index'])
            
            return {
                'success': True,
                'metadata': pdf_file[0],
                'chunks': chunks
            }
            
        except Exception as e:
            logging.error(f"Lỗi get_pdf_content: {e}")
            return {'success': False, 'message': f'Lỗi lấy nội dung: {str(e)}'}
