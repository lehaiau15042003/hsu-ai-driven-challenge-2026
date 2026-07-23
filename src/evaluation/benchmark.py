import time

from src.evaluation.metrics import Metrics

class Benchmark:

    def __init__(self, model):
        self.quality = None
        self.throughput = None
        self.latency = None
        self.safety = None
        self.model = model

    def run(self, dataset):
        X = dataset["prompt"]
        y = dataset["label_unsafe"]

        start = time.time()
        prediction = self.model.predict(X)
        end = time.time()

        self.latency = self.evaluate_latency(start, end)
        self.throughput = self.evaluate_throughput(len(dataset), self.latency)
        self.quality = self.evaluate_quality(y, prediction)
        self.safety = self.evaluate_safety(y, prediction)

        self.report()

    @staticmethod
    def evaluate_latency(start, end):
        return Metrics.latency(start, end)

    @staticmethod
    def evaluate_throughput(num_samples, latency):
        return Metrics.throughput(num_samples, latency)

    @staticmethod
    def evaluate_quality(y_true, y_pred):
        return Metrics.quality(y_true, y_pred)

    @staticmethod
    def evaluate_safety(y_true, y_pred):
        return Metrics.safety(y_true, y_pred)

    def evaluate_resource(self):
        pass

    def report(self):
        print("===== Benchmark Result =====")
        print(f"Latency: {self.latency:.4f} s")
        print(f"Throughput: {self.throughput:.2f} samples/s")
        print(f"Safety: {self.safety:.2f} samples/s")
        print(self.quality)