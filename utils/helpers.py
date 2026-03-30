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