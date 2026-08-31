class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, mode='max', verbose=False):
        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.verbose = verbose
        self.best_score = None
        self.no_improvement_count = 0
        self.stop_training = False
    
    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        else:
            if self.mode == 'max' and current_score > self.best_score + self.delta:
                self.best_score = current_score
                self.no_improvement_count = 0
            elif self.mode == 'min' and current_score < self.best_score - self.delta:
                self.best_score = current_score
                self.no_improvement_count = 0
            else:
                self.no_improvement_count += 1
                
        if self.no_improvement_count >= self.patience:
            self.stop_training = True
            if self.verbose:
                print(f"EARLY STOPPING -> No improvement for {self.patience} epochs.")



import os
import random
from monai.transforms import MapTransform

class RandomStainStyleD(MapTransform):
    """
    Claas that draws different cohort at each epoch for more generalization, forces model to ignore colors ->
    better understand the biological structure.
    """
    def __init__(self, keys):
        super().__init__(keys)
        self.styles = ["", "_pink", "_purple", "_median"]

    def __call__(self, data):
        d = dict(data)
        chosen_style = random.choice(self.styles)
        
        test_path = d[self.keys[0]].replace(".png", f"{chosen_style}.png")
        
        if not os.path.exists(test_path):
            chosen_style = ""

        for key in self.keys:
            original_path = d[key]
            new_path = original_path.replace(".png", f"{chosen_style}.png")
            d[key] = new_path
            
        return d