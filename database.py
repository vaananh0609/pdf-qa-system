from pymongo import MongoClient
from config import Config
import logging

class Database:
    def __init__(self):
        self.client = None
        self.db = None
        # Connect now if configured to do so (fail-fast), otherwise delay connect until first use
        if getattr(Config, 'DB_CONNECT_ON_START', False):
            self.connect()
    
    def connect(self):
        try:
            self.client = MongoClient(Config.MONGODB_URI)
            self.db = self.client.pdf_qa_system
            # Test connection
            self.client.admin.command('ping')
            print("Kết nối MongoDB thành công!")
        except Exception as e:
            print(f"Lỗi kết nối MongoDB: {e}")
            raise
    
    def get_collection(self, collection_name):
        """Lấy collection từ database"""
        if self.db is None:
            self.connect()
        return self.db[collection_name]
    
    def insert_document(self, collection_name, document):
        """Thêm document vào collection"""
        try:
            collection = self.get_collection(collection_name)
            result = collection.insert_one(document)
            return result.inserted_id
        except Exception as e:
            print(f"Lỗi thêm document: {e}")
            return None
    
    def find_documents(self, collection_name, query={}, limit=None, sort=None):
        """Tìm documents trong collection"""
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query)
            
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
                
            return list(cursor)
        except Exception as e:
            print(f"Lỗi tìm documents: {e}")
            return []
    
    def delete_document(self, collection_name, query):
        """Xóa document từ collection"""
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_one(query)
            return result.deleted_count
        except Exception as e:
            print(f"Lỗi xóa document: {e}")
            return 0

    def delete_documents(self, collection_name, query):
        """Xóa NHIỀU documents từ collection (delete_many)"""
        try:
            collection = self.get_collection(collection_name)
            result = collection.delete_many(query)
            return result.deleted_count
        except Exception as e:
            print(f"Lỗi xóa nhiều documents: {e}")
            return 0
    
    def update_document(self, collection_name, query, update_data):
        """Cập nhật document trong collection"""
        try:
            collection = self.get_collection(collection_name)
            result = collection.update_one(query, {"$set": update_data})
            return result.modified_count
        except Exception as e:
            print(f"Lỗi cập nhật document: {e}")
            return 0
    
    def insert_binary_data(self, collection_name, document_with_bytes):
        """
        Thêm document có chứa binary data vào collection
        Tự động convert bytes thành BSON Binary type
        """
        try:
            from bson import Binary
            collection = self.get_collection(collection_name)
            # Convert bytes to Binary recursively
            converted_doc = self._convert_bytes_to_binary(document_with_bytes)
            result = collection.insert_one(converted_doc)
            return result.inserted_id
        except Exception as e:
            logging.error(f"Lỗi thêm document với binary data: {e}")
            return None
    
    def _convert_bytes_to_binary(self, obj):
        """
        Recursively convert bytes to BSON Binary trong dict/list
        """
        from bson import Binary
        if isinstance(obj, dict):
            return {k: self._convert_bytes_to_binary(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_bytes_to_binary(item) for item in obj]
        elif isinstance(obj, bytes):
            return Binary(obj)
        else:
            return obj
    
    def find_documents_with_binary(self, collection_name, query={}, limit=None, sort=None):
        """
        Tìm documents có chứa binary data
        """
        try:
            collection = self.get_collection(collection_name)
            cursor = collection.find(query)
            
            if sort:
                cursor = cursor.sort(sort)
            if limit:
                cursor = cursor.limit(limit)
                
            return list(cursor)
        except Exception as e:
            logging.error(f"Lỗi tìm documents với binary: {e}")
            return []
    
    def get_binary_field(self, collection_name, query, field_name):
        """
        Lấy một binary field cụ thể từ document
        """
        try:
            from bson import Binary
            collection = self.get_collection(collection_name)
            doc = collection.find_one(query)
            if doc and field_name in doc:
                binary_data = doc[field_name]
                
                # Kiểm tra None trước khi convert
                if binary_data is None:
                    return None
                
                # Convert BSON Binary to bytes
                if isinstance(binary_data, Binary):
                    # BSON Binary object - convert sang bytes
                    return bytes(binary_data)
                elif isinstance(binary_data, bytes):
                    return binary_data
                else:
                    # Thử convert sang bytes
                    try:
                        return bytes(binary_data)
                    except (TypeError, ValueError):
                        logging.warning(f"Không thể convert {type(binary_data)} sang bytes cho field {field_name}")
                        return None
            return None
        except Exception as e:
            logging.error(f"Lỗi lấy binary field: {e}")
            return None

# Tạo instance global
db = Database()
