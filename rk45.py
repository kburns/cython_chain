import numpy as np
import os as os

class RK45():

    def __init__(self,force,state,
                 iter=0,time=0,dt=1e-10,tmax=0,
                 tol=1e-9,error=1e-9,
                 scheme='CK'):

        self.force = force
        self.state = state

        self.iter     = iter
        self.time     = time
        self.dt       = dt
        self.tmax     = tmax
        self.adaptive = True

        self.tol   = tol
        self.error = error

        self.s        = 6
        self.p        = 2
        self.order    = 1/(5+1)
        self.k        = np.zeros((self.s,len(self.state)))

        self.tableu(scheme)

        self.handler = False

    def __call__(self):

        if self.handler: self.output()

        self.force(self.state, output=self.k[0])
        for i in range(1,self.s):
            self.force( self.state + (self.dt*self.a[i-1,:i]) @ self.k[:i], output=self.k[i])

        self.state += (self.dt*self.b[0]) @ self.k

        self.iter  += 1
        self.time  += self.dt

        if self.adaptive:
            self.error  = np.max(np.abs((self.dt*self.b[1]) @ self.k))
            self.dt    *= (self.tol/self.error)**self.order


    def tableu(self,scheme):

        self.a = np.zeros((self.s-1,self.s-1))
        self.b = np.zeros((self.p,self.s))

        if scheme == 'F': # Fehlberg

            self.a[0,0:1] = [ 1/4]
            self.a[1,0:2] = [ 3/32,       9/32]
            self.a[2,0:3] = [ 1932/2197, -7200/2197,  7296/2197]
            self.a[3,0:4] = [ 439/216,   -8,          3680/513,  -845/4104]
            self.a[4,0:5] = [-8/27,       2,         -3544/2565,  1859/4104, -11/40]

            self.b[0] = [16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55] #5th-order
            self.b[1] = [25/216, 0, 1408/2565,  2197/4104,   -1/5,  0   ] #4th-order


        if scheme == 'CK': # Cash-Karp

            self.a[0,0:1] = [ 1/5]
            self.a[1,0:2] = [ 3/40,        9/40]
            self.a[2,0:3] = [ 3/10,       -9/10,     6/5]
            self.a[3,0:4] = [-11/54,       5/2,     -70/27,     35/27]
            self.a[4,0:5] = [ 1631/55296,  175/512,  575/13824, 44275/110592, 253/4096]

            self.b[0] = [37/378,     0, 250/621,     125/594,     0,         512/1771] #5th-order
            self.b[1] = [2825/27648, 0, 18575/48384, 13525/55296, 277/14336, 1/4   ]   #4th-order


        self.b[1] -= self.b[0]


    def __getitem__(self,item):

        if item == 'force':
            return self.force

        if item == 'state':
            return self.state

        if item == 'error':
            return self.error

        if item == 'iter':
            return self.iter

        if item == 'time':
            return self.time

        if item == 'dt':
            return self.dt


    def __setitem__(self,item,value):

        if item == 'dt':
            self.dt = value

        if item == 'tmax':
            self.tmax = value

        if item == 'tol':
            self.tol = value

        if item == 'scheme':
            self.tableu(value)


    def go(self): return self.time < self.tmax

    def add_handler(self,
                    freq=np.inf,
                    path='output/',
                    file='write',
                    digits=4,
                    records=1000,
                    dtmin=1e-9):

        self.handler = True

        self.freq  = freq
        self.dtmin = dtmin

        if path[-1]  !=  '/': path += '/'
        self.path   = path
        self.file   = path + file
        self.digits = digits

        if not path[:-1] in os.listdir(): os.mkdir(path)

        self.buffer     = np.zeros((records,len(self.state)))
        self.buffer[0]  = self.state
        self.record     = 1
        self.count      = 0

        for f in os.listdir(path):
            if (f.find(self.file) != -1) and (f.find('.npy') != -1):
                self.count += 1

        self.checkpoint = self.count*records*self.freq + self.freq
        self.update     = False

    def output(self):

        if self.record == len(self.buffer):
            file = f"{self.file}{self.count:0{self.digits}d}{'.npy'}"
            np.save(file,self.buffer)
            self.record  = 0
            self.count  += 1

        if self.update:
            self.buffer[self.record] = self.state
            self.record     += 1
            self.checkpoint += self.freq
            self.update = False

        if self.time + self.dt > self.checkpoint + self.dtmin:
            self.dt = self.checkpoint - self.time
            self.update = True



