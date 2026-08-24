# Reflection — Lab 22 DPO/ORPO Alignment

**Người học:** Ngô Văn Nam
**MSSV:** 2A202601340
**Cohort:** K4
**Ngày:** 2026-08-24
**Runtime đã chạy:** `CPU_FALLBACK` — không phải T4 course run

---

## §1 Setup

| Hạng mục | Giá trị |
|---|---|
| GPU | NVIDIA GeForce RTX 3050 Laptop GPU, 4 GB; không đủ cho recipe Qwen2.5-3B 4-bit và PyTorch hiện tại không có CUDA |
| Base model đã chạy | `Qwen/Qwen3.5-0.8B` từ local cache, dùng làm fallback có thể chạy thật |
| LoRA | `r=16`, `lora_alpha=32`, dropout 0; SFT và DPO adapter tách biệt |
| SFT | 8 CPU update steps, lr `2e-5`, max length 192; lặp một mẫu để kiểm tra đường loss |
| DPO | beta `0.1`, lr `5e-05`, 6 update steps; CPU fallback |
| Dataset chính | `argilla/ultrafeedback-binarized-preferences-cleaned`: 2.000 train + 200 eval đã chuẩn hóa |

## §2 DPO Results

| Metric | Kết quả |
|---|---:|
| SFT elapsed | 717.9s |
| DPO elapsed | 234.4s |
| SFT loss | 0.273576 → 0.000683 |
| DPO final loss | 0.000129 |
| Chosen reward cuối | +0.281735 |
| Rejected reward cuối | -8.671161 |
| Reward gap cuối | **+8.952895** |
| Manual rubric | 0 DPO wins / 1 SFT-only wins / 7 ties |

Đây là số đo của CPU fallback. Artifact data-prep đúng dataset lab nằm ở `data/pref/train.parquet`; adapter DPO đi kèm được train trên 32 fallback pairs và đã ghi rõ provenance trong `adapters/dpo/dpo_metrics.json`.

## §3 Reward Curves Analysis

Biểu đồ `submission/screenshots/03-dpo-reward-curves.png` vẽ riêng chosen reward, rejected reward và reward gap để tránh kết luận sai chỉ từ một đường. Trong lần chạy thật này, reward gap đi từ `+0.0000` đến `+8.9529` qua 6 bước; chosen reward cuối là `+0.2817`, còn rejected reward cuối là `-8.6712`. Chosen tăng cho thấy policy đang tăng log-likelihood tương đối của câu trả lời được chọn, trong khi rejected giảm cho thấy policy đẩy câu trả lời kém ra xa reference. Vì vậy gap dương ở cuối không chỉ là một số tổng hợp: nó có thể được phân tích thành hai chuyển động có ý nghĩa khác nhau. Tuy nhiên, run CPU này rất ngắn và dữ liệu training của adapter là fallback sample, nên không được diễn giải như bằng chứng T4 ổn định. Đường gap cũng nên được đọc cùng với chosen/rejected; nếu chosen giảm mà rejected giảm nhanh hơn thì đó là likelihood displacement. Ở đây chosen và rejected đều cần được kiểm tra trực tiếp, rồi mới kết luận DPO có cải thiện helpfulness hay chỉ học tín hiệu hẹp của preference pairs.

## §4 Qualitative Comparison

`data/eval/side_by_side.jsonl` và ảnh `04-side-by-side-table.png` có 8 prompt: 4 helpfulness và 4 safety.

| Nhóm | SFT+DPO thắng | SFT-only thắng | Tie |
|---|---:|---:|---:|
| Helpfulness | 0 | 1 | 3 |
| Safety | 0 | 0 | 4 |
| **Tổng** | **0** | **1** | **7** |

Đây là manual heuristic rubric, không phải API judge. Ảnh `05-manual-rubric.png` ghi lại tiêu chí và tổng hợp để reviewer kiểm tra được cách chấm.

## §5 β Trade-off

Đã chạy β-sweep mini thật với [0.05, 0.1, 0.5] và 3 steps mỗi giá trị. Kết quả: β=0.05: gap=+1.8512, pair-win-rate=1.00; β=0.1: gap=+3.6850, pair-win-rate=1.00; β=0.5: gap=+16.3902, pair-win-rate=1.00. Gap tăng theo β trong run ngắn này, nhưng đây là training-pair signal trên CPU fallback, không phải kết luận T4-scale. β thấp thường cho policy lệch reference mạnh hơn, có thể tăng preference margin nhưng dễ overfit hoặc làm giảm tính ổn định. β cao bảo thủ hơn, giữ policy gần SFT/reference nhưng có thể làm gap tăng chậm. Run chính dùng β=`0.1` theo lab. Kết luận cuối cùng vẫn cần được xác nhận trên T4 bằng cùng base model, cùng UltraFeedback slice và nhiều step hơn.

## §6 Personal Reflection

Quyết định quan trọng nhất là tách bạch giữa “đã chạy thật trên máy hiện tại” và “đúng recipe T4 trong giáo trình”. Máy có RTX 3050 nhưng chỉ 4 GB VRAM, còn PyTorch trong môi trường này là CPU-only. Nếu cố chạy Qwen2.5-3B bằng mọi giá, tôi có thể nhận OOM hoặc tạo ra một bộ notebook trông hoàn chỉnh nhưng không có bằng chứng training. Tôi chọn một Qwen3.5-0.8B đã có trong cache, giữ đúng cấu trúc LoRA r=16 và alpha=32, rồi chạy SFT continuation, DPO loop, generation, preference scoring và evaluation thật trên CPU. Cách làm này chậm hơn nhiều và không thể thay thế kết quả T4, nhưng mọi adapter, parquet, đường loss, reward curve và bảng so sánh đều có nguồn từ execution thực tế. Tôi cũng chuẩn bị đủ 2.000 UltraFeedback train rows và 200 eval rows đúng schema để bộ nộp có provenance đúng với lab, đồng thời lưu riêng fallback pairs để không che giấu sự khác biệt giữa data-prep và training source. Bài học lớn nhất là reward gap không tự động đồng nghĩa với helpfulness tốt hơn: phải xem chosen và rejected riêng, kiểm tra prompt distribution, rồi dùng rubric độc lập. Nếu có GPU phù hợp, bước tiếp theo là chạy nguyên recipe Qwen2.5-3B + UltraFeedback 2k, beta 0.1, lr 5e-7 và thay các số fallback bằng log T4.

## §7 Benchmark (optional)

NB6 benchmark chính thức chưa chạy vì IFEval/GSM8K/MMLU yêu cầu CUDA và base model của course. Tôi không tạo score giả hoặc ảnh benchmark giả. Notebook 06 có execution evidence ghi rõ gate này; đây là phần optional/bonus, không phải core gatekeeper.
