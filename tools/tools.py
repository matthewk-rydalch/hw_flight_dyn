import numpy as np
    
def Euler2Quaternion(phi, th, psi):
    e0 = np.cos(psi/2.0)*np.cos(th/2.0)*np.cos(phi/2.0)+np.sin(psi/2.0)*np.sin(th/2.0)*np.sin(phi/2.0)
    e1 = np.cos(psi/2.0)*np.cos(th/2.0)*np.sin(phi/2.0)-np.sin(psi/2.0)*np.sin(th/2.0)*np.cos(phi/2.0)
    e2 = np.cos(psi/2.0)*np.sin(th/2.0)*np.cos(phi/2.0)+np.sin(psi/2.0)*np.cos(th/2.0)*np.sin(phi/2.0)
    e3 = np.sin(psi/2.0)*np.cos(th/2.0)*np.cos(phi/2.0)-np.cos(psi/2.0)*np.sin(th/2.0)*np.sin(phi/2.0)

    e = np.array([e0, e1, e2, e3])

    return e

def Quaternion2Euler(e):
    e0 = e.item(0)
    e1 = e.item(1)
    e2 = e.item(2)
    e3 = e.item(3)

    phi = np.arctan2(2*(e0*e1+e2*e3), e0**2+e3**2-e1**2-e2**2)
    if np.abs(2*(e0*e2-e1*e3)) > 1.:
        th = np.pi/2
    else:
       th = np.arcsin(2*(e0*e2-e1*e3))
    psi = np.arctan2(2*(e0*e3+e1*e2), e0**2+e1**2-e2**2-e3**2)
    psi = np.arctan2(2*(e0*e3+e1*e2), e0**2+e1**2-e2**2-e3**2)

    return phi, th, psi