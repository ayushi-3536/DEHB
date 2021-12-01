#%matplotlib inline
import numpy as np

from typing import Optional
from matplotlib import pyplot as plt



import os
import glob



def plot_hv(reg_hv,times,ax):


        return   plot_perf_over_time(
            ax,
            r,
            times,
            time_step_size=500,
            is_time_cumulated=True,

        )



def graph(path,result):
    file = ([np.loadtxt(path + '/' + f) for f in result])
    print(file)
    #print(file.shape)
    reward_all = [r[:, 0] for r in file]
    #reward_all = np.array([np.expand_dims(r,axis=1) for r in reward_all])
        # r= np.array(r)
        # r =
        # print(r)
    # print("rall",reward_all)
    # print("reward all shape",reward_all.shape)
    time_all = [r[:, 1] for r in file]

    shape_all = [len(i) for i in reward_all]
    max = np.max(shape_all)
    print("max :{}", max)
    # print("hv plot:{}", len(hv_all[0]))
    temp = np.pad(reward_all[0], (0, max - len(reward_all[0])), mode='maximum')
    print("temp:{}", len(temp))
    reward_all= np.array([np.pad(i, (0, max - len(i)), mode='maximum') for i in reward_all])
    time_all = np.array([np.pad(i, (0, max - len(i)), mode='maximum') for i in time_all])
    print("reward_all",reward_all)
    # reward_all = np.array([np.expand_dims(r, axis=1) for r in reward_all])
    # time_all = np.array([np.expand_dims(r, axis=1) for r in time_all])
    # print(reward_all)
    # print(reward_all.shape)
    # print(time_all.shape)

    # times = tuple(range(reg_hv.shape[1]))
    # times = np.tile(times, (10, 1))
    # print("times:{}", times)
    #
    # print("reg_hv shape :{}", reg_hv.shape)


    return reward_all,time_all

def plot_perf_over_time(
    ax: plt.Axes,
    results: np.ndarray,
    times: np.ndarray,
    time_step_size: int,
    is_time_cumulated: bool,
    show: bool = True,
    runtime_upper_bound: Optional[float] = None,
) -> None:
    """
    Args:
        results (np.ndarray):
            The performance of each experiment per evaluation.
            The shape must be (n_experiments, n_evals).
        times (np.ndarray):
            The runtime of each evaluation or cumulated runtime over each experiment.
            The shape must be (n_experiments, n_evals).
        time_step_size (int):
            How many time step size you would like to use for the visualization.
        is_time_cumulated (bool):
            Whether the `times` array already cumulated the runtime or not.
        label (str):
            The name of the plot.
        show (bool):
            Whether showing the plot or not.
            If you would like to pile plots, you need to make it False.
        color (str):
            Color of the plot.
        runtime_upper_bound (Optional[float]):
            The upper bound of runtime to show in the visualization.
            If None, we determine this number by the maximum of the data.
            You should specify this number as much as possible.
    """
    print("result",results.shape)
    n_experiments, n_evals = results.shape
    #n_experiments = results.shape
    results = np.maximum.accumulate(results, axis=-1)

    if not is_time_cumulated:
        times = np.cumsum(np.random.random((n_experiments)), axis=-1)
        #times = np.cumsum(np.random.random((n_experiments, n_evals)), axis=-1)
    if runtime_upper_bound is None:
        runtime_upper_bound = times[:, -1].max()

    dt = runtime_upper_bound / time_step_size

    perf_by_time_step = np.full((n_experiments, time_step_size), 1.0)
    curs = np.zeros(n_experiments, dtype=np.int32)

    for it in range(time_step_size):
        cur_time = it * dt
        for i in range(n_experiments):
            # while times[i][curs[i]] <= cur_time:
            #   curs[i] += 1
            while curs[i] < n_evals and times[i][curs[i]] <= cur_time:
                curs[i] += 1
            if curs[i]:
                perf_by_time_step[i][it] = results[i][curs[i] - 1]

    T = np.arange(time_step_size) * dt
    mean = perf_by_time_step.mean(axis=0)
    ste = perf_by_time_step.std(axis=0) / np.sqrt(n_experiments)
    return mean,ste,T

############plot ##################

_, ax = plt.subplots()
#path = 'C:\\Users\\ayush\\PycharmProjects\\DEHB\\DEHB\\'
path = '/work/dlclarge1/awad-dehb/DEHB/final/branin'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
r, times = graph(path,result)
mean,ste,T = plot_hv(r,times,ax)
ax.plot(T, mean, color="b", label="branin")
ax.fill_between(T, mean - ste, mean + ste, color="b", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
_, ax = plt.subplots()
path = '/work/dlclarge1/awad-dehb/DEHB/final/ackley'
#path = '/work/dlclarge1/awad-dehb/DEHB/final/branin'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
r, times = graph(path,result)
mean,ste,T = plot_hv(r,times,ax)
ax.plot(T, mean, color="b", label="ackley")
ax.fill_between(T, mean - ste, mean + ste, color="m", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
_, ax = plt.subplots()
path = '/work/dlclarge1/awad-dehb/DEHB/final/beale'
#path = '/work/dlclarge1/awad-dehb/DEHB/final/branin'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
r, times = graph(path,result)
mean,ste,T = plot_hv(r,times,ax)
ax.plot(T, mean, color="b", label="beale")
ax.fill_between(T, mean - ste, mean + ste, color="r", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
_, ax = plt.subplots()
path = '/work/dlclarge1/awad-dehb/DEHB/final/eggholder'
#path = '/work/dlclarge1/awad-dehb/DEHB/final/branin'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
r, times = graph(path,result)
mean,ste,T = plot_hv(r,times,ax)
ax.plot(T, mean, color="g", label="eggholder")
ax.fill_between(T, mean - ste, mean + ste, color="b", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
_, ax = plt.subplots()
path = '/work/dlclarge1/awad-dehb/DEHB/final/bukin'
#path = '/work/dlclarge1/awad-dehb/DEHB/final/branin'
extension = 'txt'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
print(result)
r, times = graph(path,result)
mean,ste,T = plot_hv(r,times,ax)
ax.plot(T, mean, color="b", label="bukin")
ax.fill_between(T, mean - ste, mean + ste, color="y", alpha=0.2)
ax.set_xscale("log")
# ax.set_yscale("log")
ax.grid()
plt.show()
plt.savefig('/work/dlclarge1/awad-dehb/DEHB/rewardvstime.pdf', dpi=450)


