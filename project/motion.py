import numpy as np

class SimpleMotionModel:
    """
    A simple motion model to track position and velocity using exponential smoothing.
    predict adds the calculated velocity to the current position to predict the next position
    update updates the position and velocity based on the new measurement

    """
    def __init__(self, state_dim=8):
        self.state = np.zeros(state_dim)  # [x, y, w, h, dx, dy, dw, dh]
        self.velocity_weight = 0.7
        
    def predict(self):
        self.state[0:4] += self.state[4:8]
        return self.state[0:4]
        
    def update(self, measurement):
        """assumes position is x1, y1, x2, y2"""
        if len(measurement) == 4: 
            pos = np.array([
                measurement[0],  # x
                measurement[1],  # y
                measurement[2] - measurement[0],  # width
                measurement[3] - measurement[1]   # height
            ])
        else:
            pos = measurement
            
        # Update position
        old_pos = self.state[0:4].copy()
        self.state[0:4] = pos #update position with current frame
        
        # Update velocity with exponential smoothing
        new_velocity = pos - old_pos
        self.state[4:8] = self.velocity_weight * self.state[4:8] + \
                         (1 - self.velocity_weight) * new_velocity
        
        return self.state[0:4]