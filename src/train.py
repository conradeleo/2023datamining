class TrainEngine:
    def __init__(self, gru, SF_conbine, train_loss_function, test_loss_function, optimizer):
        self.gru = gru
        self.SF_conbine = SF_conbine
        self.train_loss_function = train_loss_function
        self.test_loss_function = test_loss_function
        self.optimizer = optimizer

    def __call__(self):
        ...