<div align="center">

[🇬🇧 English](README.md) • [🇻🇳 Tiếng Việt](README_VI.md)

</div>
# 🧠 WorldModel Representation Benchmark: VAE vs. ViT vs. V-JEPA on CLEVRER

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.5](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Benchmark: CLEVRER](https://img.shields.io/badge/Benchmark-CLEVRER-green.svg)](http://clevrer.csail.mit.edu/)
[![Framework: World Models](https://img.shields.io/badge/Domain-World%20Models%20%26%20Physical%20Reasoning-orange.svg)]()

> **Nghiên cứu & Đánh giá Không Gian Biểu Diễn Trạng Thái Thị Giác (State-Space Representations) cho World Models**  
> *Tác giả:* **Louis Đăng Khoa** (MSSV: `2510905`)

---

## 📌 Tổng Quan Dự Án (Overview)

Trong các kiến trúc **World Models** (Mô hình Thế giới), việc lựa chọn không gian biểu diễn trạng thái $S$ đóng vai trò quyết định đến khả năng dự đoán động lực học, suy luận vật lý và tính toán chuyển động.

Dự án này thực hiện nghiên cứu so sánh định lượng và trực quan hóa chuyên sâu giữa ba triết lý học biểu diễn thị giác cốt lõi (tất cả đều được **Pretrained & Frozen**), cùng với mốc **Raw Pixel** (Zero-learning baseline) trên cùng một chuỗi video vật lý phức tạp từ tập dữ liệu **CLEVRER** (CoLlision Events for Video REpresentation and Reasoning):

1. **VAE (AutoencoderKL / Pixel Reconstruction):** Học nén và tái tạo pixel 2D ở mức thấp.
2. **ViT (Supervised Vision Transformer):** Nhận diện thuộc tính ngữ nghĩa từ khung hình tĩnh có giám sát.
3. **V-JEPA (Video Joint Embedding Predictive Architecture):** Dự đoán đặc trưng ngữ nghĩa trong không gian ẩn spatio-temporal tubelets bị che mà không cần tái tạo pixel.

```
                                  ┌───► VAE    (Tái tạo Pixel 2D)        ───► S_VAE    ┐
Video clip X (16 frames, 4s) ─────┼───► ViT    (Phân loại có nhãn 2D)    ───► S_ViT    ├───► Trực quan hóa (t-SNE/PCA) ───► Probing & CKA
     (CLEVRER Dataset)            └───► V-JEPA (Dự đoán đặc trưng ẩn 3D) ───► S_VJEPA ┘
                                                +
                                        Raw Pixel (Baseline)
```

---

## 📊 Bảng Kết Quả Thực Nghiệm Toàn Diện (Benchmark Summary)

Dữ liệu được trích xuất trực tiếp từ các thực nghiệm thực tế tại `outputs/summary.csv`:

| Tiêu chí Đánh giá | Raw Pixel | VAE (KL-f8) | ViT (Base) | V-JEPA 2 (ViT-L/16) | Ý Nghĩa / Kết Luận Thực Nghiệm |
|:---|:---:|:---:|:---:|:---:|:---|
| **Kích thước vector trạng thái ($S$)** | 3,072 | 4,096 | 768 | 1,024 | Độ nén của không gian tiềm ẩn |
| **Probe: Màu sắc (TB 8 màu)** | 57.7% | 63.5% | **93.9%** | 87.2% | ViT vượt trội ở các đặc trưng tĩnh/bề mặt |
| **Probe: Hình khối (Khối lập phương)** | 52.0% | 51.3% | 78.2% | **98.2%** | V-JEPA mã hóa hình học 3D vượt trội |
| **Probe: Đếm số lượng vật thể** | 54.9% | 62.8% | 66.9% | **92.9%** | V-JEPA hiểu rõ cấu trúc thực thể trong bối cảnh |
| **Probe: Nhiều va chạm ($\ge 3$)** | 51.4% | 55.4% | 52.7% | **75.2%** | V-JEPA nắm bắt được các tương tác vật lý |
| **Va chạm (Cố định số vật thể)** | 48.8% | 52.3% | 53.1% | **66.5%** | Tách biệt va chạm thật sự khỏi sự thay đổi số vật thể |
| **Mũi tên thời gian (Xuôi / Ngược probe)** | — | 50.0% | 50.0% | **97.5%** | **Phát hiện đột phá:** VAE & ViT hoàn toàn ngẫu nhiên (50%), V-JEPA nhận diện chiều thời gian gần như tuyệt đối |
| **Độ tương đồng CKA với Raw Pixel** | 1.00 | **0.71** | 0.20 | 0.35 | VAE giữ nhiều thông tin bề mặt pixel thô |
| **Phương sai 2 thành phần PCA đầu** | 10.3% | 4.8% | 24.6% | **35.6%** | V-JEPA có cấu trúc nội tại tập trung & rõ ràng |
| **Số lượng tham số (M)** | 0 | 83.7M | 85.8M | 304.7M | Chi phí mô hình |
| **Thời gian suy luận (giây / clip)** | — | 2.51s | **0.73s** | 5.46s | Tốc độ trích xuất feature trên GPU |
| **VRAM đỉnh tiêu thụ (GB)** | — | 1.02 GB | **0.50 GB** | 1.54 GB | Bộ nhớ đồ họa cần thiết khi inference |

---

## 🔬 Những Phát Hiện Cốt Lõi (Key Insights)

1. **Chiều thời gian (Arrow of Time & Causality):**
   - VAE và ViT đạt **50.0%** (ngang đoán mò) khi phân biệt chuỗi video chạy xuôi hay chạy ngược. Điều này chứng minh việc xử lý từng khung hình độc lập làm mất hoàn toàn động lực học thời gian.
   - V-JEPA đạt **97.5%**, chứng minh kiến trúc 3D Spatio-temporal Tubelets tự nhiên mã hóa nguyên lý nhân quả và hướng diễn tiến của thời gian.

2. **Khả năng suy luận vật lý & Va chạm (Physical Collision Dynamics):**
   - Khi kiểm tra sự kiện va chạm ($\ge 3$ va chạm), V-JEPA đạt **75.2%** (vượt xa ViT 52.7% và VAE 55.4%).
   - Thậm chí khi kiểm soát/cố định số vật thể trong cảnh, V-JEPA vẫn duy trì **66.5%**, chứng minh đặc trưng va chạm không bị phụ thuộc vào số lượng object.

3. **Bẫy tái tạo pixel (The Pixel Reconstruction Fallacy):**
   - VAE có CKA lên tới **0.71** so với Raw Pixel. VAE dành phần lớn công suất tiềm ẩn để tái tạo màu nền sàn và ánh sáng thay vì hiểu cơ chế vật lý.
   - V-JEPA tối ưu hóa việc dự đoán đặc trưng bị che (Masked Feature Prediction), loại bỏ chi tiết dư thừa ở cấp pixel để tập trung vào động lực học vật thể.

---

## 🖼️ Thư Viện Trực Quan Hóa (Visual Showcase)

### 1. Trực quan quỹ đạo trạng thái theo thời gian (Latent Dynamics & Collisions)
Quỹ đạo tiềm ẩn trong các sự kiện va chạm vật lý:

| Clip 231 (Tương tác va chạm) | Clip 6 (Chuyển động liên tục) |
|:---:|:---:|
| ![Latent Dynamics 231](outputs/figures/12_anim_clip231.gif) | ![Latent Dynamics 6](outputs/figures/12_anim_clip6.gif) |

### 2. Bản đồ nhiệt Linear Probe trên các thuộc tính
V-JEPA chiếm ưu thế gần như áp đảo ở các thuộc tính cấu trúc 3D, số lượng và va chạm:
![Probe Heatmap](outputs/figures/06_probe_heatmap.png)

### 3. Phân bố không gian trạng thái t-SNE & Quỹ đạo 3D PCA
So sánh mức độ gom cụm theo đặc trưng vật lý giữa các mô hình:

| Không gian t-SNE | Quỹ đạo 3D PCA |
|:---:|:---:|
| ![t-SNE Latent Space](outputs/figures/02_tsne.png) | ![3D PCA Trajectory](outputs/figures/04_pca3d.png) |

### 4. Ma trận tương đồng biểu diễn (Centered Kernel Alignment - CKA)
![CKA Similarity](outputs/figures/07_cka.png)

### 5. Biểu diễn đặc trưng không gian (Spatial Patch PCA to RGB)
![Patch PCA](outputs/figures/05_patch_pca_rgb.png)

---

## 📁 Cấu Trúc Dự Án (Repository Structure)

```
WorldModel/
├── .gitignore                      # Loại trừ các file dữ liệu nặng (*.zip, *.pt, *.npy, wm/)
├── LICENSE                         # Giấy phép MIT
├── README.md                       # Tài liệu báo cáo toàn diện & hướng dẫn
├── requirements.txt                # Danh sách thư viện phụ thuộc
├── VAE_ViT_VJEPA_CLEVRER.ipynb     # Jupyter Notebook chính chứa toàn bộ pipeline thực nghiệm
└── outputs/                        # Thư mục lưu trữ kết quả thực nghiệm
    ├── figures/                    # Toàn bộ biểu đồ phân tích và ảnh động GIF (01 - 12)
    │   ├── 01_clevrer_preview.png
    │   ├── 02_tsne.png
    │   ├── 03_pca_spectrum.png
    │   ├── 04_pca3d.png
    │   ├── 05_patch_pca_rgb.png
    │   ├── 06_probe_heatmap.png
    │   ├── 07_cka.png
    │   ├── 08_retrieval.png
    │   ├── 09_temporal_similarity.png
    │   ├── 10_freeze_probe.png
    │   ├── 11_change_clip231.png
    │   ├── 11_change_clip6.png
    │   ├── 12_anim_clip231.gif
    │   └── 12_anim_clip6.gif
    └── *.csv                       # Toàn bộ bảng số liệu định lượng (summary, probe, cka, cost, v.v.)
```

> **Lưu ý về kích thước:** Các file cache nặng (`CLEVRER_videos.npy` ~943MB, `S_*_time.npy`, checkpoint weights `vjepa2_1_vitl_384.pt` ~5.1GB, và video dataset `data/` ~6.2GB) đã được thiết lập tự động bỏ qua qua `.gitignore` nhằm tuân thủ giới hạn lưu trữ của GitHub (dưới 100MB/file).

---

## ⚙️ Cài Đặt & Hướng Dẫn Tái Hiện (Quickstart & Reproduction)

### 1. Khởi tạo môi trường ảo

```bash
# Clone repository
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Khởi tạo môi trường ảo
python -m venv wm

# Kích hoạt môi trường:
# Trên Windows:
wm\Scripts\activate
# Trên Linux/macOS:
source wm/bin/activate
```

### 2. Cài đặt các gói phụ thuộc

```bash
# Cài đặt PyTorch với hỗ trợ CUDA 12.1 (khuyên dùng để tăng tốc trên GPU)
pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu121

# Cài đặt các thư viện từ requirements.txt
pip install -r requirements.txt
```

### 3. Chuẩn bị Dữ liệu & Trọng số Pretrained

- **Tập dữ liệu CLEVRER:**
  - Tải tập validation video và annotation từ trang chủ [CLEVRER](http://clevrer.csail.mit.edu/).
  - Giải nén vào thư mục `data/clevrer_val/`.
- **Pretrained Weights cho V-JEPA 2:**
  - Tải checkpoint `vjepa2_1_vitl_384.pt` từ nguồn chính thức của Meta FAIR / V-JEPA repository và lưu vào `data/`.

### 4. Chạy thực nghiệm

Mở Jupyter Lab / Notebook để chạy tuần tự:

```bash
jupyter notebook VAE_ViT_VJEPA_CLEVRER.ipynb
```

Tất cả kết quả biểu đồ và số liệu định lượng sẽ được tự động xuất ra thư mục `outputs/`.

---

## 📖 Trích Dẫn & Tham Khảo (Citations & References)

Nếu dự án này hữu ích cho nghiên cứu của bạn, vui lòng tham khảo:

```bibtex
@misc{dang2026worldmodelclevrer,
  author = {Louis Dang Khoa},
  title = {A Comparative Benchmark of Visual State-Space Representations (VAE vs. ViT vs. V-JEPA) for World Models on Physical CLEVRER Dynamics},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/}}
}
```

- **CLEVRER Benchmark:** Yi et al., *"CLEVRER: Collision Events for Video Representation and Reasoning"*, ICLR 2020.
- **V-JEPA:** Bardes et al., *"Revisiting Feature Prediction for Learning Visual Representations from Video"*, Meta AI Research, 2024.
- **Vision Transformer (ViT):** Dosovitskiy et al., *"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"*, ICLR 2021.
- **AutoencoderKL (VAE):** Rombach et al., *"High-Resolution Image Synthesis with Latent Diffusion Models"*, CVPR 2022.

---

## 📄 Bản Quyền (License)

Dự án được phân phối dưới giấy phép **MIT License**. Chi tiết xem tại [LICENSE](LICENSE).


