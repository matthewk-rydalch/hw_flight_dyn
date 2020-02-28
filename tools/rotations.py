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

def Euler2Rotation(phi, theta, psi):
        """
        Converts euler angles to rotation matrix (R_b^i, i.e., body to inertial)
        """
        # only call sin and cos once for each angle to speed up rendering
        c_phi = np.cos(phi)
        s_phi = np.sin(phi)
        c_theta = np.cos(theta)
        s_theta = np.sin(theta)
        c_psi = np.cos(psi)
        s_psi = np.sin(psi)

        R_roll = np.array([[1, 0, 0],
                           [0, c_phi, s_phi],
                           [0, -s_phi, c_phi]])
        R_pitch = np.array([[c_theta, 0, -s_theta],
                            [0, 1, 0],
                            [s_theta, 0, c_theta]])
        R_yaw = np.array([[c_psi, s_psi, 0],
                          [-s_psi, c_psi, 0],
                          [0, 0, 1]])
        R = R_roll @ R_pitch @ R_yaw  # inertial to body (Equation 2.4 in book)
        return R.T  # transpose to return body to inertial

def Quaternion2Rotation(e):
    phi, th, psi = Quaternion2Euler(e)
    R = Euler2Rotation(phi, th, psi)

    return R