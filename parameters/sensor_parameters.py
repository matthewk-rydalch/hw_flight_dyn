import sys
sys.path.append('..')
import numpy as np

#-------- Accelerometer --------
#TODO put the actual std deviations back
accel_sigma = 0.0#0.0025*9.8  # standard deviation of accelerometers in m/s^2

#-------- Rate Gyro --------
gyro_x_bias = 0.  # bias on x_gyro
gyro_y_bias = 0.  # bias on y_gyro
gyro_z_bias = 0.  # bias on z_gyro
gyro_sigma = 0.0#0.13*np.pi/180.  # standard deviation of gyros in rad/sec

#-------- Pressure Sensor(Altitude) --------
static_pres_sigma = 0.0#0.01*1000  # standard deviation of static pressure sensors in Pascals
static_pres_beta = 0.0 # I added this

#-------- Pressure Sensor (Airspeed) --------
diff_pres_sigma = 0.0#0.002*1000  # standard deviation of diff pressure sensor in Pascals
diff_pres_beta = 0.0 # I added this

#-------- Magnetometer --------
mag_beta = 0.0#np.radians(1.0)
mag_sigma = 0.0#np.radians(0.03)

#-------- GPS --------
ts_gps = 1.0
# Is this gps beta right???
K_gps = 1. / 1100.  # 1 / s
gps_n_sigma = 0.0#0.21
gps_e_sigma = 0.0#0.21
gps_h_sigma = 0.0#0.40
gps_Vg_sigma = 0.0#0.05
gps_course_sigma = 0.0#gps_Vg_sigma / 10
