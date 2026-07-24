import time
from src.evaluation.metrics import Metrics
import pandas as pd

class Benchmark:
    def __init__(self, model):
        self.quality = None
        self.throughput = None
        self.latency = None
        self.safety = None
        self.model = model
        self.has_labels = False

    def run(self, dataset):
        X = dataset["prompt"]
        num_samples = len(X)
        
        # Kiểm tra xem có nhãn không (Tập Test thi thật thường không có)
        self.has_labels = "label_unsafe" in dataset.columns
        if self.has_labels:
            y = dataset["label_unsafe"]

        start = time.time()
        prediction = self.model.predict(X)
        end = time.time()

        # Tính tốc độ
        self.latency = Metrics.latency(start, end, num_samples)
        self.throughput = Metrics.throughput(num_samples, start, end)

        # Tính điểm (chỉ làm khi có nhãn)
        if self.has_labels:
            self.quality = Metrics.quality(y, prediction)
            self.safety = Metrics.safety(y, prediction)
        else:
            # Nếu chạy tập Test không nhãn, tự động lưu kết quả ra CSV để đi nộp
            output_df = pd.DataFrame({"prompt": X, "label_unsafe": prediction})
            output_df.to_csv("submission_test.csv", index=False)
            print("Đã lưu kết quả dự đoán ra file submission_test.csv")

        self.report()

    def report(self):
        print("===== Benchmark Result =====")
        print(f"Latency (Độ trễ):    {self.latency:.2f} ms/câu")
        print(f"Throughput (Tốc độ): {self.throughput:.2f} câu/giây")
        
        if self.has_labels:
            print(f"Safety (Accuracy):   {self.safety:.4f}")
            print("Quality Metrics:")
            for k, v in self.quality.items():
                print(f"  - {k.capitalize()}: {v:.4f}")
        else:
            print("Không thể chấm điểm F1/Quality vì tập dữ liệu Test không có nhãn.")