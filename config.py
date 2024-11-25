from dataclasses import dataclass

@dataclass
class SimpleTrainingConfig:
    batch_size: int = 16
    epochs: int = 10
    learning_rate: float = 0.001
    
    use_cuda: bool = True
    save_model: bool = True
    model_path: str = "simple_model.pt"
    
    train_split: float = 0.8
    shuffle: bool = True
    
    num_classes: int = 5
