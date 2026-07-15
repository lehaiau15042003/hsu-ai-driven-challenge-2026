from abc import ABC, abstractmethod

class BaseModel(ABC):

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, prompt: str):
        pass

    @abstractmethod
    def unload(self):
        pass