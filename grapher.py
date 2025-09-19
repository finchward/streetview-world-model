from config import Config
import matplotlib
matplotlib.use('Agg') # Set the non-interactive 'Agg' backend
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class Grapher:
    def __init__(self):
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(11, 5))
       
        # Data storage for graph 1
        self.train_x_1 = []
        self.train_y_1 = []
        self.val_x_1 = []
        self.val_y_1 = []

        # Data storage for graph 2
        self.train_x_2 = []
        self.train_y_2 = []
        self.val_x_2 = []
        self.val_y_2 = []
       
        # Line objects for graph 1
        self.train_line_1, = self.ax1.plot([], [], 'r-', label='Training', linewidth=1)
        self.val_line_1, = self.ax1.plot([], [], 'b-', label='Validation')
        
        # Line objects for graph 2
        self.train_line_2, = self.ax2.plot([], [], 'r-', label='Training', linewidth=1)
        self.val_line_2, = self.ax2.plot([], [], 'b-', label='Validation')
        
        # self.ax1.set_yscale("log")
        # self.ax2.set_yscale("log")
     
        self.ax1.legend()
        self.ax2.legend()
       
    def update_line(self, graph_index, line_type, x_data, y_data):
        x_data = np.array(x_data)
        y_data = np.array(y_data)

        if graph_index == 0:
            if line_type == "Training":
                self.train_x_1 = x_data
                self.train_y_1 = y_data
                self.train_line_1.set_data(x_data, y_data)
            elif line_type == "Validation":
                self.val_x_1 = x_data
                self.val_y_1 = y_data
                self.val_line_1.set_data(x_data, y_data)
            
            self.ax1.relim()
            self.ax1.autoscale_view()
        else:
            if line_type == "Training":
                self.train_x_2 = x_data
                self.train_y_2 = y_data
                self.train_line_2.set_data(x_data, y_data)
            elif line_type == "Validation":
                self.val_x_2 = x_data
                self.val_y_2 = y_data
                self.val_line_2.set_data(x_data, y_data)
            self.ax2.relim()
            self.ax2.autoscale_view()

        # Logic for saving the graph
        graph_dir = Path.cwd() / Path(Config.graph_dir) / Config.model_name
        graph_dir.mkdir(parents=True, exist_ok=True)

        save_path = graph_dir / "graph.png"
        self.fig.savefig(save_path, bbox_inches='tight')
        print(f'Graph saved to {save_path}')
