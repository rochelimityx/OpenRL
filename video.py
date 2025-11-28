import imageio
import os

class VideoRecoder(object):
    def __init__(self,dir_name,fps=25):
        self.dir_name=dir_name
        self.fps=fps
    
    def init(self,enabled=True):
        self.frames=[]
        self.enabled=self.dir_name is not None and enabled
    
    def record(self,env):
        if self.enabled:
            frame=env.render(mode="rgb_array")
            self.frames.append(frame)
    
    def save(self,file_name):
        if self.enabled:
            path=os.path.join(self.dir_name,file_name)
            imageio.mimsave(path,self.frames,fps=self.fps)