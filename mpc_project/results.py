import numpy as np
import matplotlib.pyplot as plt

def show_results(mav_n, mav_e, target_n, target_e, current_time, sim_time):

    num_tests = sim_time.shape[0]
    length = sim_time.shape[1]

    # print the time lag
    total_lag = current_time[:,length - 1] - sim_time[:,length - 1] - current_time[:,0]
    avg_lag = total_lag / length
    print('Average Time Lag = ', avg_lag)

    # plot error vs time
    test = ['TH = 50', 'TH = 25', 'TH = 5', 'Ref as WP']
    error = np.sqrt(
        (target_n - mav_n) ** 2 + (target_e - mav_e) ** 2)
    for index in range(num_tests):
        plt.plot(sim_time[index], error[index], label = test[index])
    plt.xlabel('time [s]')
    plt.ylabel('error [m]')
    plt.legend()
    plt.show()


def read_file(f):
    mav_n = []
    mav_e = []
    target_n = []
    target_e = []
    current_time = []
    sim_time = []

    for line in f:
        category = line.split()
        mav_n.append(float(category[0]))
        mav_e.append(float(category[1]))
        target_n.append(float(category[2]))
        target_e.append(float(category[3]))
        current_time.append(float(category[4]))
        sim_time.append(float(category[5]))

    return mav_n, mav_e, target_n, target_e, current_time, sim_time


mav_n_list = []
mav_e_list = []
target_n_list = []
target_e_list = []
current_time_list = []
sim_time_list = []

f_50=open('../../txt_files/Dir_TH_50.txt', 'r')
f_25=open('../../txt_files/Dir_TH_25.txt', 'r')
f_5=open('../../txt_files/Dir_TH_5.txt', 'r')
f_ref_wp=open('../../txt_files/Dir_RefWp.txt', 'r')

mav_n, mav_e, target_n, target_e, current_time, sim_time = read_file(f_50)
mav_n_list.append(mav_n)
mav_e_list.append(mav_e)
target_n_list.append(target_n)
target_e_list.append(target_e)
current_time_list.append(current_time)
sim_time_list.append(sim_time)

mav_n, mav_e, target_n, target_e, current_time, sim_time = read_file(f_25)
mav_n_list.append(mav_n)
mav_e_list.append(mav_e)
target_n_list.append(target_n)
target_e_list.append(target_e)
current_time_list.append(current_time)
sim_time_list.append(sim_time)

mav_n, mav_e, target_n, target_e, current_time, sim_time = read_file(f_5)
mav_n_list.append(mav_n)
mav_e_list.append(mav_e)
target_n_list.append(target_n)
target_e_list.append(target_e)
current_time_list.append(current_time)
sim_time_list.append(sim_time)

mav_n, mav_e, target_n, target_e, current_time, sim_time = read_file(f_ref_wp)
mav_n_list.append(mav_n)
mav_e_list.append(mav_e)
target_n_list.append(target_n)
target_e_list.append(target_e)
current_time_list.append(current_time)
sim_time_list.append(sim_time)

f_50.close()
f_25.close()
f_5.close()
f_ref_wp.close()

show_results(np.array(mav_n_list), np.array(mav_e_list), np.array(target_n_list), np.array(target_e_list), np.array(current_time_list), np.array(sim_time_list))
