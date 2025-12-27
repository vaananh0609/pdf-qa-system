from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
import os
from werkzeug.utils import secure_filename
from pdf_service import PDFService
from database import db
import logging
from bson import ObjectId
from config import Config

# Cấu hình logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config.from_object('config')

# Khởi tạo services
pdf_service = PDFService()

# Cấu hình admin (lấy từ env/config; KHÔNG hardcode trong repo)
ADMIN_USERNAME = getattr(Config, 'ADMIN_USERNAME', 'admin')
ADMIN_PASSWORD = getattr(Config, 'ADMIN_PASSWORD', None)

def convert_objectid_to_str(obj):
    """Chuyển đổi ObjectId thành string cho JSON serialization"""
    if isinstance(obj, list):
        return [convert_objectid_to_str(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_objectid_to_str(value) for key, value in obj.items()}
    elif isinstance(obj, ObjectId):
        return str(obj)
    else:
        return obj

@app.route('/')
def index():
    """Trang chủ - hiển thị danh sách PDF và form hỏi đáp"""
    try:
        pdfs = pdf_service.get_all_pdfs()
        return render_template('index.html', pdfs=pdfs)
    except Exception as e:
        logging.error(f"Lỗi trang chủ: {e}")
        return render_template('index.html', pdfs=[], error="Có lỗi xảy ra")

@app.route('/ask', methods=['POST'])
def ask_question():
    """Xử lý câu hỏi từ user"""
    try:
        import time
        data = request.get_json()
        question = data.get('question', '').strip()
        filename = data.get('filename', '')
        
        if not question:
            return jsonify({'success': False, 'message': 'Vui lòng nhập câu hỏi'})
        
        # Đo thời gian bắt đầu
        start_time = time.time()
        
        # Tìm kiếm và trả lời
        result = pdf_service.search_and_answer(question, filename if filename else None)
        
        # Tính thời gian xử lý
        response_time_ms = int((time.time() - start_time) * 1000)

        # Lưu lịch sử chat khi thành công
        if result.get('success'):
            sources = result.get('sources', []) or []
            actual_files = sorted(list({s.get('filename') for s in sources if s.get('filename')}))
            
            # Lấy similarity stats nếu có
            similarity_stats = result.get('similarity_scores')
            history_item = {
                'question': question,
                'answer': result.get('answer', ''),
                'filename': filename if filename else 'Tất cả file',
                'actual_files_used': actual_files,
                'sources': sources,
                'total_chunks_scanned': result.get('total_chunks_searched', 0),
                'chunks_used_count': result.get('relevant_chunks', 0),
                'response_time_ms': response_time_ms,  # Tổng thời gian
                'generation_time_ms': result.get('generation_time_ms', 0),  # Thời gian sinh câu trả lời
                'timestamp': __import__('datetime').datetime.now()
            }
            
            # Thêm similarity scores nếu có
            if similarity_stats:
                history_item['similarity_scores'] = similarity_stats.get('scores', [])
                history_item['avg_similarity_score'] = similarity_stats.get('avg_score')
                history_item['max_similarity_score'] = similarity_stats.get('max_score')
                history_item['min_similarity_score'] = similarity_stats.get('min_score')
            
            db.insert_document('search_history', history_item)

        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Lỗi xử lý câu hỏi: {e}")
        return jsonify({'success': False, 'message': 'Có lỗi xảy ra khi xử lý câu hỏi'})

@app.route('/pdf/<filename>')
def view_pdf(filename):
    """Hiển thị nội dung PDF"""
    try:
        result = pdf_service.get_pdf_content(filename)
        
        if not result['success']:
            flash(result['message'], 'error')
            # Quay lại trang phù hợp
            if 'admin_logged_in' in session:
                return redirect(url_for('admin_dashboard'))
            else:
                return redirect(url_for('index'))
        
        # Xác định nguồn gốc để quay lại đúng trang
        from_admin = request.args.get('from') == 'admin'
        
        return render_template('pdf_viewer.html', 
                             metadata=result['metadata'],
                             chunks=result['chunks'],
                             from_admin=from_admin)
        
    except Exception as e:
        logging.error(f"Lỗi hiển thị PDF: {e}")
        flash('Có lỗi xảy ra khi hiển thị PDF', 'error')
        # Quay lại trang phù hợp
        if 'admin_logged_in' in session:
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('index'))

@app.route('/view_pdf_file/<filename>')
def view_pdf_file(filename):
    """Xem PDF trực tiếp trên trình duyệt (không tải xuống)"""
    try:
        file_path = os.path.join('uploads', filename)
        
        if not os.path.exists(file_path):
            flash('File không tồn tại', 'error')
            return redirect(url_for('index'))
        
        # Trả về PDF với content-type đúng để trình duyệt hiển thị
        return send_file(file_path, mimetype='application/pdf', as_attachment=False)
        
    except Exception as e:
        logging.error(f"Lỗi xem PDF: {e}")
        flash('Có lỗi xảy ra khi xem PDF', 'error')
        return redirect(url_for('index'))

@app.route('/download/<filename>')
def download_pdf(filename):
    """Tải xuống file PDF"""
    try:
        file_path = os.path.join('uploads', filename)
        
        if not os.path.exists(file_path):
            flash('File không tồn tại', 'error')
            return redirect(url_for('index'))
        
        return send_file(file_path, as_attachment=True)
        
    except Exception as e:
        logging.error(f"Lỗi tải xuống PDF: {e}")
        flash('Có lỗi xảy ra khi tải xuống file', 'error')
        return redirect(url_for('index'))

@app.route('/pdf_image/<chunk_id>')
def get_pdf_image(chunk_id):
    """Lấy ảnh của chunk PDF scanned"""
    try:
        from flask import Response
        from database import db
        from bson import Binary
        
        # Tìm chunk trong database
        chunks = db.find_documents_with_binary('pdf_chunks', {'chunk_id': chunk_id})
        
        if not chunks:
            return "Chunk không tồn tại", 404
        
        chunk = chunks[0]
        
        # Lấy image_data
        image_data = chunk.get('image_data')
        if not image_data:
            return "Không tìm thấy ảnh", 404
        
        # Convert BSON Binary to bytes
        if isinstance(image_data, Binary):
            # BSON Binary object - convert sang bytes
            # BSON Binary có thể dùng như bytes, nhưng để chắc chắn ta convert rõ ràng
            image_bytes = bytes(image_data)
        elif isinstance(image_data, bytes):
            # Đã là bytes
            image_bytes = image_data
        else:
            # Thử convert sang bytes
            image_bytes = bytes(image_data)
        
        # Trả về ảnh
        return Response(image_bytes, mimetype='image/png')
        
    except Exception as e:
        logging.error(f"Lỗi lấy ảnh PDF: {e}", exc_info=True)
        return f"Lỗi: {str(e)}", 500

# Admin routes
@app.route('/admin')
def admin_login():
    """Trang đăng nhập admin"""
    if 'admin_logged_in' in session:
        return redirect(url_for('admin_dashboard'))
    return render_template('admin_login.html')

@app.route('/admin/login', methods=['POST'])
def admin_login_post():
    """Xử lý đăng nhập admin"""
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '').strip()

    if not ADMIN_PASSWORD:
        flash('ADMIN_PASSWORD chưa được cấu hình trên server', 'error')
        return redirect(url_for('admin_login'))
    
    if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
        session['admin_logged_in'] = True
        flash('Đăng nhập thành công', 'success')
        return redirect(url_for('admin_dashboard'))
    else:
        flash('Tên đăng nhập hoặc mật khẩu sai', 'error')
        return redirect(url_for('admin_login'))

@app.route('/admin/logout')
def admin_logout():
    """Đăng xuất admin"""
    session.pop('admin_logged_in', None)
    flash('Đã đăng xuất', 'info')
    return redirect(url_for('admin_login'))

@app.route('/admin/dashboard')
def admin_dashboard():
    """Trang quản lý admin"""
    if 'admin_logged_in' not in session:
        flash('Vui lòng đăng nhập', 'error')
        return redirect(url_for('admin_login'))
    
    try:
        pdfs = pdf_service.get_all_pdfs()
        return render_template('admin_dashboard.html', pdfs=pdfs)
    except Exception as e:
        logging.error(f"Lỗi admin dashboard: {e}")
        return render_template('admin_dashboard.html', pdfs=[], error="Có lỗi xảy ra")

@app.route('/admin/upload', methods=['POST'])
def admin_upload():
    """Upload file PDF (admin)"""
    if 'admin_logged_in' not in session:
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'})
    
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'Không có file'})
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Chưa chọn file'})
        
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({'success': False, 'message': 'Chỉ chấp nhận file PDF'})
        
        filename = secure_filename(file.filename)
        caption = request.form.get('caption', '').strip()
        
        logging.info(f"Đang xử lý upload file: {filename}")
        
        # Kiểm tra file đã tồn tại chưa (chỉ kiểm tra database)
        existing_files = pdf_service.get_all_pdfs()
        if any(pdf['filename'] == filename for pdf in existing_files):
            logging.warning(f"File {filename} đã tồn tại trong database")
            return jsonify({'success': False, 'message': f'File "{filename}" đã tồn tại trong hệ thống'})
        
        logging.info(f"Bắt đầu upload file: {filename}")
        result = pdf_service.upload_pdf(file, filename, caption)
        logging.info(f"Kết quả upload {filename}: {result.get('success', False)}")
        return jsonify(convert_objectid_to_str(result))
        
    except Exception as e:
        logging.error(f"Lỗi upload admin: {e}")
        return jsonify({'success': False, 'message': f'Lỗi upload: {str(e)}'})

@app.route('/admin/delete/<filename>', methods=['POST'])
def admin_delete_pdf(filename):
    """Xóa PDF file (admin)"""
    if 'admin_logged_in' not in session:
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'})
    
    try:
        result = pdf_service.delete_pdf(filename)
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Lỗi xóa PDF: {e}")
        return jsonify({'success': False, 'message': f'Lỗi xóa file: {str(e)}'})

@app.route('/admin/stats')
def admin_stats():
    """Thống kê hệ thống (admin)"""
    if 'admin_logged_in' not in session:
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'})
    
    try:
        pdfs = pdf_service.get_all_pdfs()
        total_chunks = db.find_documents('pdf_chunks', {})
        search_history = db.find_documents('search_history', {})

        stats = {
            'total_files': len(pdfs),
            'total_chunks': len(total_chunks),
            'total_size': sum(pdf.get('file_size', 0) for pdf in pdfs),
            'total_searches': len(search_history),
            'files': pdfs
        }
        
        return jsonify({'success': True, 'stats': convert_objectid_to_str(stats)})
    except Exception as e:
        logging.error(f"Lỗi thống kê: {e}")
        return jsonify({'success': False, 'message': f'Lỗi thống kê: {str(e)}'})
        
@app.route('/admin/chat-history')
def admin_chat_history():
    """Lịch sử chat gần đây (admin)"""
    if 'admin_logged_in' not in session:
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'})
    try:
        history = db.find_documents('search_history', {}, sort=[('timestamp', -1)], limit=100)
        return jsonify({'success': True, 'history': convert_objectid_to_str(history)})
    except Exception as e:
        logging.error(f"Lỗi lấy lịch sử chat: {e}")
        return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})

@app.route('/admin/chart-data')
def admin_chart_data():
    """Dữ liệu biểu đồ admin"""
    if 'admin_logged_in' not in session:
        return jsonify({'success': False, 'message': 'Chưa đăng nhập'})
    try:
        from datetime import datetime, timedelta

        # Upload theo ngày (7 ngày gần nhất)
        upload_data, labels = [], []
        for i in range(6, -1, -1):
            d = datetime.now() - timedelta(days=i)
            start = d.replace(hour=0, minute=0, second=0, microsecond=0)
            end = d.replace(hour=23, minute=59, second=59, microsecond=999999)
            count = len(db.find_documents('pdf_files', {'created_at': {'$gte': start, '$lte': end}}))
            upload_data.append(count)
            labels.append(d.strftime('%d/%m'))

        # Quan tâm nhiều nhất theo file từ sources thực tế
        history = db.find_documents('search_history', {})
        file_counts = {}
        for item in history:
            files = item.get('actual_files_used') or []
            if files:
                for f in files:
                    file_counts[f] = file_counts.get(f, 0) + 1
            else:
                # fallback cũ
                fname = item.get('filename', 'Tất cả file')
                file_counts[fname] = file_counts.get(fname, 0) + 1
        top = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        return jsonify({
            'success': True,
            'upload_data': {'labels': labels, 'data': upload_data},
            'file_data': {'labels': [t[0] for t in top], 'data': [t[1] for t in top]}
        })
    except Exception as e:
        logging.error(f"Lỗi chart-data: {e}")
        return jsonify({'success': False, 'message': f'Lỗi: {str(e)}'})

if __name__ == '__main__':
    # Tạo thư mục uploads
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    
    # Tạo thư mục templates
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    app.run(debug=True, host='0.0.0.0', port=5000)
