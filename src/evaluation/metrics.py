from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metrics:

    @staticmethod
    def latency(start_time, end_time):
        return end_time - start_time

    @staticmethod
    def throughput(num_samples, latency):
        if latency == 0:
            return 0
        return num_samples / latency

    @staticmethod
    def safety(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def quality(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred,zero_division=0),
            "recall": recall_score(y_true, y_pred,zero_division=0),
            "f1": f1_score(y_true, y_pred)
        }

    @staticmethod
    def resource_usage():
        pass