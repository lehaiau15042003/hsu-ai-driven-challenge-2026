import pandas as pd
import os
import argparse

from src.models.baseline_model import TFIDFBaselineModel
from src.models.phobert_model import PhoBERTModel
from src.evaluation.benchmark import Benchmark

def main():
    parser = argparse.ArgumentParser(description="Run Benchmark on AI Prompt Firewall")
    parser.add_argument("--model", type=str, choices=["baseline", "phobert"], default="phobert", help="Model to evaluate")
    parser.add_argument("--dataset", type=str, choices=["val", "test"], default="test", help="Dataset to evaluate on")
    args = parser.parse_args()

    val_path = "data/processed/val_master_vi.csv"
    test_path = "data/processed/AI_Driven_Challenge_VI_Test.csv"
    
    data_path = test_path if args.dataset == "test" else val_path
    
    if not os.path.exists(data_path):
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {data_path}")
        print("Vui lòng đảm bảo bạn đã đặt file .csv đúng vị trí (thư mục data/processed/)")
        return

    print(f"Đang đọc dữ liệu từ: {data_path}")
    df = pd.read_csv(data_path)

    if args.model == "phobert":
        print("Khởi tạo mô hình PhoBERT...")
        model_path = "models/phobert_model.pt"
        if not os.path.exists(model_path):
            print(f"Lỗi: Không tìm thấy file trọng số tại {model_path}")
            print("Vui lòng đặt file phobert_model.pt vào thư mục models/")
            return
            
        model = PhoBERTModel(model_path=model_path)
        model.load()
    else:
        print("Khởi tạo Baseline Model...")
        model = TFIDFBaselineModel()
        print("Lưu ý: Bạn cần train baseline model trước.")
        return

    print("\nBẮT ĐẦU CHẠY BENCHMARK...")
    benchmark = Benchmark(model=model)
    benchmark.run(df)
    
if __name__ == "__main__":
    main()
