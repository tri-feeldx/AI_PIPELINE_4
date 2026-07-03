# 5 Phút Setup & Chạy Thử

## 1️⃣ Cài đặt (2 phút)

```bash
# Clone repo
git clone https://github.com/tri-feeldx/AI_PIPELINE_5.git
cd AI_PIPELINE_5
git checkout submit-minimal

# Cài Python packages
pip install pymupdf shapely pydantic dataclasses-json Pillow numpy scipy scikit-learn
```

## 2️⃣ Chuẩn bị file PDF (30 giây)

Cần 1 file PDF bản vẽ kỹ thuật Revit (ví dụ: `drawing.pdf`):
- GA plan (mặt bằng — có sàn tô màu)
- Scale: 1:50 hoặc 1:100 (tùy loại bản vẽ)

## 3️⃣ Chạy extraction (1 phút)

Tạo file `test_extract.py`:

```python
from src.slab_v2.config import SlabV2Config
from src.slab_v2.pipeline import extract_slabs_v2

# Config
cfg = SlabV2Config(
    debug_images=False,
    manual_scale=100  # ← thay số tỉ lệ bản vẽ ở đây
)

# Chạy
result = extract_slabs_v2(
    pdf_path="drawing.pdf",
    page_index=10,        # ← trang muốn extract (0-based)
    config=cfg,
    use_ai=False
)

# Kết quả
print(f"Status: {result.status}")
print(f"Slabs: {len(result.slabs)}")
print(f"Columns: {len(result.columns)}")
print(f"Walls: {len(result.walls)}")

if result.status == "OK":
    total_area = sum(s.get('area_m2', 0) for s in result.slabs)
    print(f"Total slab area: {total_area:.1f} m²")
```

Chạy:
```bash
python test_extract.py
```

**Kỳ vọng output:**
```
Status: OK
Slabs: 1
Columns: 15
Walls: 12
Total slab area: 3450.5 m²
```

## 4️⃣ Xuất kết quả (nếu cần)

```python
import json

# Thêm vào test_extract.py
output = {
    "page": result.page_index + 1,
    "status": result.status,
    "slabs_count": len(result.slabs),
    "slabs_area_m2": sum(s.get('area_m2', 0) for s in result.slabs),
    "columns_count": len(result.columns),
    "walls_count": len(result.walls),
}

with open("result.json", "w") as f:
    json.dump(output, f, indent=2)

print("✓ Saved to result.json")
```

Chạy lại:
```bash
python test_extract.py
cat result.json
```

## 5️⃣ Batch processing (nhiều trang)

```python
for page_idx in range(5):  # trang 1-5
    result = extract_slabs_v2("drawing.pdf", page_idx, cfg, use_ai=False)
    print(f"p{page_idx+1}: {result.status} - "
          f"{len(result.slabs)} slab, "
          f"{len(result.columns)} col, "
          f"{len(result.walls)} wall")
```

---

## ❓ FAQ

**Q: Làm sao biết tỉ lệ bản vẽ?**
- Thường viết trên title block: "1:100" hay "1:50"
- Nếu không biết → thử `manual_scale=100` (phổ biến nhất)

**Q: Kết quả sai sao?**
- Check: Trang đó là GA plan (mặt bằng) không?
- Check: Sàn có được tô màu (filled) không? (không phải dashed)
- Check: Scale có đúng không?

**Q: Mất bao lâu?**
- ~10-30 giây per trang
- Phụ thuộc độ phức tạp bản vẽ

---

## 📊 Output chi tiết

```json
{
  "page": 11,
  "status": "OK",
  "slabs_count": 1,
  "slabs_area_m2": 3450.5,
  "columns_count": 15,
  "walls_count": 12
}
```

- `status="OK"` → thành công
- `columns_count` → số cột phát hiện (từ schedule)
- `walls_count` → số tường phát hiện (từ schedule)

---

**Version**: 1.0 (Slab + Columns + Walls extraction)
**Last updated**: 2026-07-03
**Repo**: https://github.com/tri-feeldx/AI_PIPELINE_5

