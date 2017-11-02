import numpy as np
import os
import time

class Visualizer():
    def __init__(self, checkpoints_path, visdom_port=None):
        if visdom_port is not None:
            import visdom
            self.vis = visdom.Visdom(port=visdom_port)
            self.use_vis = True
        else:
            self.use_vis = False
            
        self.checkpoints_path = checkpoints_path
        self.log_name = os.path.join(checkpoints_path, 'loss_log.txt')
        now = time.strftime('%c')
        self.start_time = time.time()
        
        with open(self.log_name, 'a') as log_file:
            log_file.write('Training {} \n'.format(now))

    def plot_quality(self, name, quality, epoch):
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X':[],'Y':[], 'legend':list(quality.keys())}
            self.plot_data['X'].append(epoch)
            self.plot_data['Y'].append([quality[k] for k in self.plot_data['legend']])
            
        self.plot_data['X'].append(epoch)
        self.plot_data['Y'].append([quality[k] for k in self.plot_data['legend']])
        print(self.plot_data)
        self.vis.line(
            X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1),
            Y=np.array(self.plot_data['Y']),
            opts={
                'title': name,
                'legend': self.plot_data['legend'],
                'xlabel': 'epoch'},
            win=name)

    def print_quality(self, quality, epoch, epochs):
        message = '[Epoch {}/{}] Time elapsed: {:.2f}; '.format(epoch, epochs, time.time() - self.start_time)
        for k, v in quality.items():
            message += '{}: {:.4f}; '.format(k, v)

        print(message)
        with open(self.log_name, 'a') as log_file:
            log_file.write('{}\n'.format(message))
            
    def quality(self, name, quality, epoch, epochs):
        self.print_quality(quality, epoch, epochs)
        if self.use_vis :
            self.plot_quality(name, quality, epoch)
            