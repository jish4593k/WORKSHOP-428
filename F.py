
import torch
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

a

class GameController:
    def __init__(self):  # Initialize your machine learning model here
        self.model = self.build_model()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()

    def build_model(self):
     
        return nn.Sequential(
            nn.Linear(2, 1),
        )

    def train_model(self, features, targets, epochs=10):
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

    def predict_ml_model(self, input_data):
        
        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        prediction = self.model(input_tensor)
        return prediction.item()


game_controller = GameController()


def index():
    return game_controller.render_index()

@
def results():
    stats = request.form['data']
    info = game_controller.get_stats(stats)
    title = game_controller.make_string(stats.split(',')[0], stats.split(',')[1])
    
    ml_input = [float(stats.split(',')[0]), float(stats.split(',')[1])]
    ml_prediction = game_controller.predict_ml_model(ml_input)

    return game_controller.render_results(info, title, ml_prediction)

def internal_error(error):
    return game_controller.handle_error(error)

if __name__ == '__main__':
    app.run()
