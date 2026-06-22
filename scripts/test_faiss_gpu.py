import time
import numpy as np
import faiss


def test_faiss_gpu():
    print("FAISS version:", getattr(faiss, "__version__", "unknown"))

    # 1. Kiểm tra FAISS build có GPU API không
    has_gpu_api = hasattr(faiss, "StandardGpuResources")
    print("Has GPU API:", has_gpu_api)

    if not has_gpu_api:
        print("\n❌ FAISS hiện tại KHÔNG có GPU support.")
        print("Có thể bạn đang dùng faiss-cpu.")
        print("Gợi ý cài bản GPU:")
        print("  conda install -c pytorch -c nvidia faiss-gpu")
        return

    # 2. Kiểm tra FAISS nhìn thấy bao nhiêu GPU
    num_gpus = faiss.get_num_gpus()
    print("Number of GPUs visible to FAISS:", num_gpus)

    if num_gpus == 0:
        print("\n❌ FAISS có GPU API nhưng không nhìn thấy GPU.")
        print("Kiểm tra NVIDIA driver, CUDA runtime, hoặc CUDA_VISIBLE_DEVICES.")
        return

    # 3. Tạo dữ liệu test
    d = 768
    nb = 200_000
    nq = 1_000
    k = 10

    np.random.seed(42)
    xb = np.random.random((nb, d)).astype("float32")
    xq = np.random.random((nq, d)).astype("float32")

    # Normalize để dùng inner product như cosine similarity
    faiss.normalize_L2(xb)
    faiss.normalize_L2(xq)

    # 4. CPU index
    cpu_index = faiss.IndexFlatIP(d)

    t0 = time.perf_counter()
    cpu_index.add(xb)
    cpu_add_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    D_cpu, I_cpu = cpu_index.search(xq, k)
    cpu_search_time = time.perf_counter() - t0

    print("\nCPU index type:", type(cpu_index))
    print(f"CPU add time:    {cpu_add_time:.4f} s")
    print(f"CPU search time: {cpu_search_time:.4f} s")

    # 5. Chuyển index sang GPU
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, faiss.IndexFlatIP(d))

    t0 = time.perf_counter()
    gpu_index.add(xb)
    gpu_add_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    D_gpu, I_gpu = gpu_index.search(xq, k)
    gpu_search_time = time.perf_counter() - t0

    print("\nGPU index type:", type(gpu_index))
    print(f"GPU add time:    {gpu_add_time:.4f} s")
    print(f"GPU search time: {gpu_search_time:.4f} s")

    # 6. Kiểm tra kết quả CPU/GPU có gần giống nhau không
    same_top1 = np.mean(I_cpu[:, 0] == I_gpu[:, 0])
    print(f"\nTop-1 match CPU vs GPU: {same_top1 * 100:.2f}%")

    if gpu_search_time < cpu_search_time:
        print("\n✅ FAISS đang tận dụng GPU và GPU nhanh hơn CPU trong test này.")
    else:
        print("\n⚠️ FAISS chạy được trên GPU, nhưng chưa nhanh hơn CPU trong test này.")
        print("Điều này có thể xảy ra nếu dữ liệu nhỏ, batch query nhỏ, hoặc overhead copy CPU↔GPU lớn.")


if __name__ == "__main__":
    test_faiss_gpu()