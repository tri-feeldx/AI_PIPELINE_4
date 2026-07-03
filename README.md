# Slab + Columns + Walls Extraction (Minimal Version)

Trích xuất **sàn, cột, tường** từ PDF bản vẽ kỹ thuật Revit — **100% deterministic, không cần AI hay API**.

## Cài đặt

```bash
pip install pymupdf shapely pydantic dataclasses-json Pillow numpy scipy scikit-learn
```

## Cách dùng (30 giây)

```python
from src.slab_v2.config import SlabV2Config
from src.slab_v2.pipeline import extract_slabs_v2

cfg = SlabV2Config(debug_images=False)
result = extract_slabs_v2(
    pdf_path="drawing.pdf",
    page_index=10,  # trang 11 (0-based)
    config=cfg,
    use_ai=False    # chỉ geometry + text, không AI
)

print(f"✓ Status: {result.status}")
print(f"✓ Slabs: {len(result.slabs)} ({sum(s.get('area_m2',0) for s in result.slabs):.0f} m²)")
print(f"✓ Columns: {len(result.columns)}")
print(f"✓ Walls: {len(result.walls)}")
```

## Các ví dụ

### Trích xuất nhiều trang

```python
for page_idx in range(10, 20):
    r = extract_slabs_v2("drawing.pdf", page_idx, cfg, use_ai=False)
    if r.status == "OK":
        print(f"p{page_idx+1}: {len(r.slabs)} slab, {len(r.columns)} col, {len(r.walls)} wall")
```

### Xuất JSON

```python
import json

output = {
    "page": result.page_index + 1,
    "status": result.status,
    "slabs_count": len(result.slabs),
    "slabs_area_m2": sum(s.get('area_m2', 0) for s in result.slabs),
    "columns": [
        {
            "symbol": c.symbol,
            "center": (c.polygon.centroid.x, c.polygon.centroid.y) if c.polygon else None,
        }
        for c in result.columns
    ],
    "walls_count": len(result.walls),
}

with open(f"page_{result.page_index+1}.json", "w") as f:
    json.dump(output, f, indent=2)
```

## Tham số config

```python
cfg = SlabV2Config(
    debug_images=False,       # True = lưu debug image (chậm)
    manual_scale=100,         # 1:100 — thay theo bản vẽ
    enable_opening_judge=False,  # detect VOID lỗ sàn
)
```

## Troubleshooting

| Vấn đề | Giải pháp |
|---|---|
| No slabs detected | Trang là GA (mặt bằng) không? Sàn có tô màu không? |
| No columns/walls | Trang có COLUMN/WALL SCHEDULE bảng không? |
| ImportError: fitz | `pip install pymupdf` |

## Chi tiết kỹ thuật

- **Use case**: Trích xuất từ Revit PDF exports (các bản vẽ kỹ thuật)
- **Tốc độ**: 10–30 giây/trang
- **Độ chính xác**: ~95% trên bản vẽ tiêu chuẩn
- **Dependencies**: Pure Python (numpy, shapely, pydantic)
- **Scale**: ~8,000 dòng code, 24 commit core

## Licence

See LICENSE file.

