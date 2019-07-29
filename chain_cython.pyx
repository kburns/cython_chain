# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

import numpy as np
from libc.math cimport sin as sinfunc
from libc.math cimport cos as cosfunc
import os


cdef void tension_solve(unsigned int n, double[:] c, double[:] a, double[:] T):
    """
    Computes in-place tri-diagonal solve for tension.

    Parameters
    ----------
    n = len(T)
    c = +cos(diff(theta)))

    """

    # upper diagonal band.
    cdef unsigned int n1 = n - 1
    cdef unsigned int n2 = n - 2
    cdef unsigned int i, j
    cdef double tmp

    for i in range(1, n1):
        j = i - 1
        tmp = 1 / (2 - a[j]*c[j])
        c[i] *= tmp
        T[i] += a[j]*T[j]
        T[i] *= tmp

    T[n1] += a[n2]*T[n2]
    T[n1] /= 2 - a[n2]*c[n2]

    for i in range(2, n+1):
        j = n - i
        T[j] += c[j]*T[j+1]


cdef class Chain:
    """Festina Lente"""

    cdef public unsigned int n
    cdef public double friction
    cdef public double[:] state
    cdef public double[:] theta_omega
    cdef public double[:] omega_alpha
    cdef public double[:] theta
    cdef public double[:] omega
    cdef public double[:] alpha
    cdef public double[:] tension
    cdef public double[:] torque
    cdef public double[:] sin
    cdef public double[:] cos
    cdef public double[:] coscopy

    def __init__(self, unsigned int n, double friction=0.0):

        self.n = n
        self.friction = friction

        self.state = np.zeros(3*n)
        self.theta_omega = self.state[:2*n]
        self.omega_alpha = self.state[n:]
        self.theta = self.state[:n]
        self.omega = self.state[n:2*n]
        self.alpha = self.state[2*n:]
        self.tension = np.zeros(n)
        self.sin = np.zeros(n-1)
        self.cos = np.zeros(n-1)
        self.coscopy = np.zeros(n-1)
        self.torque = np.zeros(n)

    def __call__(self, double[:] input, double[:] output, double g=1.0):

        # Local references
        cdef unsigned int n = self.n
        cdef double friction = self.friction
        cdef double[:] theta_omega = self.theta_omega
        cdef double[:] omega_alpha = self.omega_alpha
        cdef double[:] theta = self.theta
        cdef double[:] omega = self.omega
        cdef double[:] alpha = self.alpha
        cdef double[:] sin = self.sin
        cdef double[:] cos = self.cos
        cdef double[:] coscopy = self.coscopy
        cdef double[:] tension = self.tension
        cdef double[:] torque = self.torque

        # Copy input
        theta_omega[:] = input

        # Trig
        cdef unsigned int i
        cdef double diff

        for i in range(n-1):
            diff = theta[i+1] - theta[i]
            sin[i] = sinfunc(diff)
            cos[i] = cosfunc(diff)
            coscopy[i] = cos[i]

        # Tension and torque
        for i in range(n):
            tension[i] = omega[i]*omega[i]
        tension[0] = tension[0] + g * cosfunc(theta[0])

        if friction > 0.0:
            for i in range(n):
                torque[i] = 2.0 * omega[i]
            for i in range(n-1):
                torque[i] -= omega[i+1]
                torque[i+1] -= omega[i]
            torque[-1] -= omega[-1]
            for i in range(n):
                torque[i] *= (- friction)
            for i in range(n-1):
                tension[i] += sin[i]*torque[i+1]
                tension[i+1] -= sin[i]*torque[i]

        tension_solve(n, cos, coscopy, tension)

        # Acceleration
        alpha[0] = sin[0] * tension[1] - g * sinfunc(theta[0])
        for i in range(1, n-1):
            alpha[i] = (sin[i] * tension[i+1]) - (sin[i-1] * tension[i-1])
        alpha[n-1] = - (sin[n-2] * tension[n-2])

        if friction > 0.0:
            alpha[0] += torque[0]
            for i in range(n-1):
                alpha[i+1] += 2.0 * torque[i+1]
                alpha[i] += cos[i]*torque[i+1]
                alpha[i+1] += cos[i]*torque[i]

        # Copy output
        output[:] = omega_alpha

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
        self.theta[:] = np.pi - np.arctan2(a,np.linspace(-b,1-b,self.n))
        self.omega[:] = 0
        if output: return self['state']

    def folded_state(self,angle,output=True):
        self.theta[0::2] = angle
        self.theta[1::2] = angle - np.pi
        self.omega[:] = 0
        if output: return self['state']

    def info(self,directory):

        if not "info.txt" in os.listdir(directory):
            info = open(directory+"info.txt",'a')
            info.write('Simulation Parameters\n')
            info.write('---------------------\n')
            info.write(f' elements : {self.n}\n')
            info.write(f' friction : {self.friction}\n\n')
            info.close()

