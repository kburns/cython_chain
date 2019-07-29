import numpy           as np
from scipy.linalg  import solve_banded as solve

class Chain:
    """Festina Lente"""

    def __init__(self,n,friction=0):

        self.n       = n

        self.theta   = np.zeros(n)
        self.omega   = np.zeros(n)
        self.alpha   = np.zeros(n)
        self.tension = np.zeros(n)

        self.dif = np.zeros(n-1)
        self.sin = np.zeros(n-1)
        self.cos = np.zeros(n-1)

        self.L       = np.zeros((3,n))
        self.L[1,1:] = np.full(n-1,2)
        self.L[1,0]  = 1

        self.friction = friction
        if friction: self.torque = np.zeros(n)

    def __call__(self,z,output,g=1):

        self.theta, self.omega = z[:self.n], z[self.n:]

        self.dif           = np.diff(self.theta)
        self.sin, self.cos = np.sin(self.dif), -np.cos(self.dif)

        self.tension     = self.omega**2
        self.tension[0] += g*np.cos(self.theta[0])

        if self.friction:
            self.torque = 2*self.omega
            self.torque[:-1] -= self.omega[1:]
            self.torque[1:]  -=  self.omega[:-1]
            self.torque[-1]  -=  self.omega[-1]
            self.torque  *= -self.friction

            self.tension[:-1] += self.sin*self.torque[1:]
            self.tension[1:]  -= self.sin*self.torque[:-1]

        self.L[0,1:], self.L[2,:-1]  = self.cos, self.cos
        solve((1,1),self.L,self.tension,overwrite_b=True,overwrite_ab=False,check_finite=False)

        self.alpha[:-1] = self.sin*self.tension[1:]
        self.alpha[-1]  = 0
        self.alpha[1:] -= self.sin*self.tension[:-1]
        self.alpha[0]  -= g*np.sin(self.theta[0])

        if self.friction:
            self.alpha[0]   +=   self.torque[0]
            self.alpha[1:]  += 2*self.torque[1:]
            self.alpha[:-1] += self.cos*self.torque[1:]
            self.alpha[1:]  += self.cos*self.torque[:-1]

        output[:] = np.concatenate([self.omega,self.alpha])

    def __getitem__(self,item):

        if item == 'theta':   return self.theta
        if item == 'omega':   return self.omega
        if item == 'alpha':   return self.alpha
        if item == 'tension': return self.tension

        if item == 'torque':
            if not self.friction: np.zeros(self.n)
            return self.torque

        if item == 'friction':
            return self.friction

        if item == 'state':
            return np.concatenate([self.theta,self.omega])

        if item == 'force':
            return np.concatenate([self.omega,self.alpha])

        if item == 'x,y': return self.Cartesian()

        if item == 'x': return self.Cartesian()[0]

        if item == 'y': return self.Cartesian()[1]

        if item == 'u,v': return self.Cartesian(velocity=True)[2:]

        if item == 'u': return self.Cartesian(velocity=True)[2]

        if item == 'v': return self.Cartesian(velocity=True)[3]

        if item == 'x,y,u,v': return self.Cartesian(velocity=True)

        if item == 'e': return self.energy(total=False)

        if item == 'E': return self.energy(total=True)

        raise ValueError('item not found: {}'.format(item))

    def __setitem__(self,item,value):

        if item == 'theta': self.theta = value
        if item == 'omega': self.omega = value

        if item == 'state':
            self.theta, self.omega = value[:self.n], value[self.n:]

        if item == 'friction':
            self.friction = value
            if not self.friction: self.torque = np.zeros(self.n)


    def Cartesian(self,origin=True,velocity=False):

        x, y = np.cumsum(np.sin(self.theta)), -np.cumsum(np.cos(self.theta))

        if origin: x, y = np.insert(x,0,0), np.insert(y,0,0)

        if not velocity: return x, y

        u, v = np.cumsum(self.omega*np.cos(self.theta)), np.cumsum(self.omega*np.sin(self.theta))

        if origin: u, v = np.insert(u,0,0), np.insert(v,0,0)

        return x, y, u, v

    def energy(self,total=True,g=1):

        x, y, u, v = self.Cartesian(origin=False,velocity=True)

        e = 0.5*(u**2 + v**2) + g*y

        if total: return np.sum(e)

        return e

    def hanging_state(self,a,b,output=True):
        self.theta = np.pi - np.arctan2(a,np.linspace(-b,1-b,self.n))
        self.omega *= 0
        if output: return self['state']

    def folded_state(self,angle,output=True):
        self.theta[0::2] = angle
        self.theta[1::2] = angle - np.pi
        self.omega *= 0
        if output: return self['state']

    def info(self,directory):

        if not "info.txt" in os.listdir(directory):
            info = open(directory+"info.txt",'a')
            info.write('Simulation Parameters\n')
            info.write('---------------------\n')
            info.write(f' elements : {chain.n}\n')
            info.write(f' friction : {chain.friction}\n\n')
            info.close()

