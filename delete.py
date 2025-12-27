from pymongo import MongoClient
from config import Config

client = MongoClient(Config.MONGODB_URI)
db = client.pdf_qa_system

pdf_files = db.pdf_files
search_history = db.search_history

# 1. Lấy danh sách tên file còn tồn tại
existing_files = set(doc["filename"] for doc in pdf_files.find({}, {"filename": 1}))

# 2. Tìm các bản ghi lịch sử có actual_files_used chứa file không tồn tại
docs = list(search_history.find({}, {"actual_files_used": 1}))
ids_to_delete = []

for d in docs:
    files_used = d.get("actual_files_used") or []
    # nếu có ít nhất một file không nằm trong existing_files → xoá
    if any(f not in existing_files for f in files_used):
        ids_to_delete.append(d["_id"])

print(f"Sẽ xoá {len(ids_to_delete)} bản ghi lịch sử…")

# 3. Thực hiện xoá
if ids_to_delete:
    result = search_history.delete_many({"_id": {"$in": ids_to_delete}})
    print("Đã xoá", result.deleted_count, "bản ghi.")
else:
    print("Không có bản ghi nào cần xoá.")