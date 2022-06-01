from threading import Thread as oldthread
from threading import Lock
import time
from ipycanvas import Canvas
from PIL import Image as PImage
from numba import cuda
import numpy as np
class Cleanup():
    def __init__(self):
        self.threads = []
    def add(self,*threads):
        for thread in threads:
            self.threads.append(thread)
    def reset(self):
        for t in self.threads:
            t.isalive=False
            t.join()
        self.threads=[]
    def hard_reset(self):
        for a in self.threads:
            try:
                a.start()
            except:
                pass
        self.reset()
cleaner = Cleanup()

class Thread(oldthread):
    def __init__(self):
        self.index=0
        self.isalive=True
        super(Thread, self).__init__()
        cleaner.add(self)
    def fps(self):
        iold = self.index
        time.sleep(1)
        return self.index-iold

    def run(self):
        while self.isalive:
            self.index+=1
            self.step()
            
@cuda.jit
def upscale(inarr,outarr):
    i,j,k=cuda.grid(3)
    iscale=outarr.shape[0]//inarr.shape[1]
    jscale=outarr.shape[1]//inarr.shape[2]
    outarr[i][j][k]=inarr[k][i//iscale][j//jscale]
class Render(Thread):
    def __init__(self, globalmem, canvas,dim=[512,512,3],binary=False,maxframes=1000):
        self.upscaled=np.zeros(dim)
        self.threads=(16,16)
        self.blocks=(int(np.ceil(dim[0] / 16)),int(np.ceil(dim[1] / 16)),3-2*binary)
        self.grid_global_mem = globalmem
        self.canvas = canvas
        self.isalive=True
        self.active=True
        self.maxframes=maxframes
        self.allframes=[]
        super(Render, self).__init__()
    def step(self):
        if self.active:
            upscale[self.blocks,self.threads](self.grid_global_mem,self.upscaled)
            gridf=self.upscaled
            self.canvas.put_image_data(gridf*255, 0, 0)
            
            if len(self.allframes)<self.maxframes:
                self.allframes+=[PImage.fromarray((gridf*255).astype('uint8'))]
            
            time.sleep(0.01)
        else:
            time.sleep(0.5)
    def make_gif(self,fp_out,start=0):
        if len(self.allframes)>0:
            img, *imgs = self.allframes[start:]
            img.save(fp=fp_out, format='GIF', append_images=imgs,
                     save_all=True, duration=20, loop=0)
