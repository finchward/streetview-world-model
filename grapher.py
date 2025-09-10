import matplotlib.pyplot as plt

class Grapher:
    def __init__(self):
        plt.ion()
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 5))
       
        # Data storage for graph 1
        self.train_x_1 = []
        self.train_y_1 = []
        self.val_x_1 = []
        self.val_y_1 = []
        self.recon_val_x_1 = []
        self.recon_val_y_1 = []
        self.div_val_x_1 = []
        self.div_val_y_1 = []
        
        # Data storage for graph 2
        self.train_x_2 = []
        self.train_y_2 = []
        self.val_x_2 = []
        self.val_y_2 = []
        self.recon_val_x_2 = []
        self.recon_val_y_2 = []
        self.div_val_x_2 = []
        self.div_val_y_2 = []
       
        # Line objects for graph 1
        self.train_line_1, = self.ax1.plot([], [], 'r-', label='Training', linewidth=1)
        self.val_line_1, = self.ax1.plot([], [], 'b-', label='Validation')
        self.recon_val_line_1, = self.ax1.plot([], [], 'g-', label='Reconstruction Val', linewidth=1)
        self.div_val_line_1, = self.ax1.plot([], [], 'm-', label='Divergence Val', linewidth=1)
        
        # Line objects for graph 2
        self.train_line_2, = self.ax2.plot([], [], 'r-', label='Training', linewidth=1)
        self.val_line_2, = self.ax2.plot([], [], 'b-', label='Validation')
        self.recon_val_line_2, = self.ax2.plot([], [], 'g-', label='Reconstruction Val', linewidth=1)
        self.div_val_line_2, = self.ax2.plot([], [], 'm-', label='Divergence Val', linewidth=1)
        
        self.ax1.set_yscale("log")
        self.ax2.set_yscale("log")
     
        self.ax1.legend()
        self.ax2.legend()
       
    def update_line(self, graph_index, line_type, x_data, y_data):
        if graph_index == 0:
            if line_type == "Training":
                self.train_x_1 = x_data
                self.train_y_1 = y_data
                self.train_line_1.set_data(x_data, y_data)
            elif line_type == "Validation":
                self.val_x_1 = x_data
                self.val_y_1 = y_data
                self.val_line_1.set_data(x_data, y_data)
            elif line_type == "Reconstruction loss":
                self.recon_val_x_1 = x_data
                self.recon_val_y_1 = y_data
                self.recon_val_line_1.set_data(x_data, y_data)
            elif line_type == "Divergence loss":
                self.div_val_x_1 = x_data
                self.div_val_y_1 = y_data
                self.div_val_line_1.set_data(x_data, y_data)
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
            elif line_type == "Reconstruction Val":
                self.recon_val_x_2 = x_data
                self.recon_val_y_2 = y_data
                self.recon_val_line_2.set_data(x_data, y_data)
            elif line_type == "Divergence Val":
                self.div_val_x_2 = x_data
                self.div_val_y_2 = y_data
                self.div_val_line_2.set_data(x_data, y_data)
            self.ax2.relim()
            self.ax2.autoscale_view()
       
        plt.draw()
        plt.pause(0.001)