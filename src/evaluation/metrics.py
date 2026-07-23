from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class Metrics:
    @staticmethod
    def latency(start_time, end_time, num_samples):
        total_time_seconds = end_time - start_time
        return (total_time_seconds / num_samples) * 1000

    @staticmethod
    def throughput(num_samples, start_time, end_time):
        total_time_seconds = end_time - start_time
        if total_time_seconds == 0:
            return 0
        return num_samples / total_time_seconds

    @staticmethod
    def safety(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def quality(y_true, y_pred):
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, average="macro")
        }