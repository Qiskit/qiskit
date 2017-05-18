"""
Test the expressions for simplifying single qubit gates.

Author: Andrew Cross
"""
import sys
sys.path.insert(0, '../')

import qiskit.mapper as mapper

# from qiskit.mapper import compose_u3

import cmath
import math
import numpy as np
import random


I = complex(0.0, 1.0)


def Rz(lamb):
    """Z rotation."""
    return np.matrix([[cmath.exp(-I * lamb / 2.0), 0.0],
                      [0.0, cmath.exp(I * lamb / 2.0)]])


def Ry(lamb):
    """Y rotation."""
    return np.matrix([[math.cos(lamb / 2.0), -math.sin(lamb / 2.0)],
                      [math.sin(lamb / 2.0), math.cos(lamb / 2.0)]])


def u1(lamb):
    """u1 gate."""
    return Rz(lamb)


def u3(theta, phi, lamb):
    """u3 gate."""
    return np.dot(Rz(phi), np.dot(Ry(theta), Rz(lamb)))


def u2(phi, lamb):
    """u2 gate."""
    return u3(math.pi / 2.0, phi, lamb)


def testu2u2(phi1, lamb1, phi2, lamb2):
    """Test simplification rule for u2*u2=u3."""
    return np.dot(u2(phi1, lamb1),
                  u2(phi2, lamb2)) - u3(math.pi - lamb1 - phi2,
                                        phi1 + math.pi / 2.0,
                                        lamb2 + math.pi / 2.0)


MIN_V = 2.0
MAX_V = 0.0
samples = 50000
for samp in range(samples):
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    nrm = np.linalg.norm(testu2u2(phi1, lamb1, phi2, lamb2))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u2.u2: samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta1 = random.uniform(0.0, 2.0 * math.pi)
    theta2 = random.uniform(0.0, 2.0 * math.pi)
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3: samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta1 = random.uniform(0.0, 2.0 * math.pi)
    theta2 = random.uniform(0.0, 2.0 * math.pi)
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = math.pi - phi2
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 1): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta1 = random.uniform(0.0, 2.0 * math.pi)
    theta2 = -theta1
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 2, 1): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta2 = random.uniform(0.0, 2.0 * math.pi)
    theta1 = -theta2
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 2, 2): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta1 = random.uniform(0.0, 2.0 * math.pi)
    theta2 = -theta1 + 2 * math.pi
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 2, 3): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta2 = random.uniform(0.0, 2.0 * math.pi)
    theta1 = -theta2 + 2.0 * math.pi
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 2, 4): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta1 = random.uniform(0.0, 2.0 * math.pi)
    theta2 = -theta1 + math.pi
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 3, 1): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta2 = random.uniform(0.0, 2.0 * math.pi)
    theta1 = -theta2 + math.pi
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 3, 2): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta1 = random.uniform(0.0, 2.0 * math.pi)
    theta2 = -theta1 + 3 * math.pi
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 3, 3): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))

MIN_V = 2.0
MAX_V = 0.0
for samp in range(samples):
    theta2 = random.uniform(0.0, 2.0 * math.pi)
    theta1 = -theta2 + 3.0 * math.pi
    phi1 = random.uniform(0.0, 2.0 * math.pi)
    phi2 = random.uniform(0.0, 2.0 * math.pi)
    lamb1 = random.uniform(0.0, 2.0 * math.pi)
    lamb2 = random.uniform(0.0, 2.0 * math.pi)
    t, p, l = compose_u3(theta1, phi1, lamb1, theta2, phi2, lamb2)
    nrm = np.linalg.norm(np.dot(u3(theta1, phi1, lamb1),
                                u3(theta2, phi2, lamb2)) -
                         u3(t, p, l))
    if nrm < MIN_V:
        MIN_V = nrm
    if nrm > MAX_V:
        MAX_V = nrm
print("u3.u3 (case 3, 4): samp=%d min=%.15f max=%.15f" % (samples, MIN_V, MAX_V))
