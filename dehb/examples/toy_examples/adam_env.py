from ConfigSpace.hyperparameters import UniformFloatHyperparameter
from gym import Env, spaces
import numpy as np
import logging
import torch
from botorch.test_functions.synthetic import Beale, Branin, Ackley, Bukin, EggHolder
import argparse
import warnings
import  time
from dehb import DEHB
import json
from .ladam import LAdam
from .my_adam import Adam
from .my_nadam import NAdam

OPTIM_NAME_MAP = {'adam': Adam,
                  'nadam': NAdam}

ENV_ID = 0



parser = argparse.ArgumentParser()
parser.add_argument('--fix_seed', default='False', type=str, choices=['True', 'False'],
                    nargs='?', help='seed')
parser.add_argument('--run_id', default=1, type=int, nargs='?',
                    help='unique number to identify this run')
parser.add_argument('--runs', default=None, type=int, nargs='?', help='number of runs to perform')
parser.add_argument('--run_start', default=0, type=int, nargs='?',
                    help='run index to start with for multiple runs')
parser.add_argument('--iter', default=20, type=int, nargs='?',
                    help='number of DEHB iterations')
parser.add_argument('--output_path', default="./results", type=str, nargs='?',
                    help='specifies the path where the results will be saved')
strategy_choices = ['rand1_bin', 'rand2_bin', 'rand2dir_bin', 'best1_bin', 'best2_bin',
                    'currenttobest1_bin', 'randtobest1_bin',
                    'rand1_exp', 'rand2_exp', 'rand2dir_exp', 'best1_exp', 'best2_exp',
                    'currenttobest1_exp', 'randtobest1_exp']
parser.add_argument('--strategy', default="rand1_bin", choices=strategy_choices,
                    type=str, nargs='?',
                    help="specify the DE strategy from among {}".format(strategy_choices))
parser.add_argument('--mutation_factor', default=0.5, type=float, nargs='?',
                    help='mutation factor value')
parser.add_argument('--crossover_prob', default=0.5, type=float, nargs='?',
                    help='probability of crossover')
parser.add_argument('--boundary_fix_type', default='random', type=str, nargs='?',
                    help="strategy to handle solutions outside range {'random', 'clip'}")
parser.add_argument('--gens', default=1, type=int, nargs='?',
                    help='DE generations in each DEHB iteration')
parser.add_argument('--eta', default=3, type=int, nargs='?',
                    help='SH parameter')
parser.add_argument('--min_budget', default=1, type=float, nargs='?',
                    help='DEHB max budget')
parser.add_argument('--max_budget', default=200, type=float, nargs='?',
                    help='DEHB max budget')
parser.add_argument('--verbose', default='False', choices=['True', 'False'], nargs='?', type=str,
                    help='to print progress or not')
parser.add_argument('--folder', default=None, type=str, nargs='?',
                    help='name of folder where files will be dumped')
parser.add_argument('--version', default=None, type=str, nargs='?',
                    help='version of DEHB to run')
parser.add_argument('--loss', default='beale', type=str, nargs='?',
                    help='loss function')
class AdamOnSynthFuncEnv(Env):

    def __init__(self, fixed=False, act_low=1e-5, act_high=5, loss_func='beale', fixed_teacher=True):
        super(AdamOnSynthFuncEnv, self).__init__()
        global ENV_ID
        self._id = ENV_ID
        ENV_ID += 1
        self.logger = logging.getLogger(f'[chartreuse4]{self.__class__.__name__}-{self._id}[/]')
        self.curr_x = None
        self.opt = None
        self.loss_func = None
        self.__prev_state = None
        self._max_steps = 200
        self.total_steps = None
        self.__prev_fval = None
        self.__init_fval = None
        self._fixed = fixed
        self.__sum_r = 0

        self.__registered_teachers = []
        self.__teacher_kwargs = []
        self.__teacher_x = []
        self.__uninit_teachers = []
        self.__c_step = 0
        self._has_teachers = False
        if loss_func.lower() == 'beale':
            self.__loss_func = Beale
        elif loss_func.lower() == 'branin':
            self.__loss_func = Branin
        elif loss_func.lower() == 'ackley':
            self.__loss_func = Ackley
        elif loss_func.lower() == 'bukin':
            self.__loss_func = Bukin
        elif loss_func.lower() == 'eggholder':
            self.__loss_func = EggHolder

        self.__eval_starts = dict(
            Beale=[3., 3.],
            Branin=[14., -4.],
            Ackley=[14., -4.],
            Bukin=[-3., -14.],
            EggHolder=[-200., 175]
        )

        # TODO better way of setting the boxes
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.observation_space = spaces.Box(low=np.array(  # TODO figure out to automatically determine the bounds
                                                    [-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                                                     -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                                                     -np.inf, 0]),
                                                high=np.array(
                                                    [np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                                     np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, 1]),
                                                dtype=np.float32)
            self.action_space = spaces.Box(low=act_low, high=act_high, shape=(1, ), dtype=np.float32)
        self._fixed_teacher = fixed_teacher

    def register_teacher(self, optim, **kwargs):
        self._has_teachers = True
        if isinstance(optim, str):
            optim = OPTIM_NAME_MAP[optim]
        self.logger.info(f'Registering {optim.__name__} as teacher')
        try:
            self.__teacher_x.append(torch.tensor(self.curr_x.data.numpy().copy(), requires_grad=True))
            self.__registered_teachers.append(optim([self.__teacher_x[-1]], **kwargs))
            self.__uninit_teachers.append(optim)
            self.__teacher_kwargs.append(kwargs)
        except:
            self.__teacher_x.append(None)
            self.__registered_teachers.append(optim)
            self.__uninit_teachers.append(optim)
            self.__teacher_kwargs.append(kwargs)

    def render(self, mode="human"):
        raise NotImplementedError

    def seed(self, seed=None):
        self.logger.debug(f'Setting seed {seed}')
        self.rng = np.random.RandomState(seed)
        self._seed = seed
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    def teachers_steps(self):
        for teacher, teacher_x in zip(self.__registered_teachers, self.__teacher_x):
            tmp = self.loss_func(teacher_x)
            # print(teacher, tmp)
            teacher.zero_grad()
            tmp.backward(torch.ones_like(tmp))
            teacher.step()

    def get_teacher_actions(self):
        tvals = []
        for teacher in self.__registered_teachers:
            tvals.append(teacher.get_teacher())
        return tvals

    @torch.no_grad()
    def teacher_update_x(self):
        for idx in range(len(self.__registered_teachers)):
            if self.__teacher_x[idx] is None:  # Initializing teachers
                self.__teacher_x[idx] = torch.tensor(self.curr_x.data.numpy().copy(), requires_grad=True)
                self.__registered_teachers[idx] = self.__uninit_teachers[idx](
                    [self.__teacher_x[idx]], **self.__teacher_kwargs[idx])
            else:  # Update the actual x location of the student to the teachers.
                tx = self.curr_x.detach().clone()  # detach probably not necessary after no_grad
                self.__teacher_x[idx].data = tx.data

    def reset_teachers(self, lr):
        for idx in range(len(self.__registered_teachers)):
            self.__teacher_x[idx] = None
            self.__registered_teachers[idx] = None
            if not self._fixed_teacher:
                self.__teacher_kwargs[idx]['lr'] = lr

    def reset(self):
        self.loss_func = self.__loss_func()  # TODO create multi function version
        start_loc = [self.rng.uniform(*self.loss_func.bounds[:, 0]),
                     self.rng.uniform(*self.loss_func.bounds[:, 1])]
        start_stepsize = self.rng.choice(np.logspace(
            np.log10(self.action_space.low)[0], np.log10(self.action_space.high)[0], 1000)
        )  # if we do this across instances we can learn this otherwise sample it
        self.reset_teachers(start_stepsize)
        self.__c_step = 0

        self.__sum_r = 0
        if self._fixed:
            start_loc = self.__eval_starts[self.loss_func.__class__.__name__]
            start_stepsize = 1e-3
        self.curr_x = torch.tensor(start_loc, requires_grad=True)
        self.teacher_update_x()
        self.opt = LAdam([self.curr_x], lr=start_stepsize)

        self.__prev_state = None

        f_val = self.loss_func(self.curr_x)
        self.opt.zero_grad()
        f_val.backward(torch.ones_like(f_val))
        self.opt.step(learned_stepsize=None)  # Default Adam behaviour with the initial lr
        self.total_steps = 0
        self.__opt_val = self.loss_func(self.loss_func.optimizers[0])
        self.__prev_fval = np.max([f_val.data.numpy() - self.__opt_val.data.numpy(), 1e-8])
        self.__init_fval = np.max([f_val.data.numpy() - self.__opt_val.data.numpy(), 1e-8])

        self.teacher_update_x()
        self.teachers_steps()  # Calling teacher_steps after doing the actual optimizer step allows a kind of look ahead

        return self._compute_state()

    @torch.no_grad()
    def _compute_state(self):
        s = list(self.opt.state.values())
        p = self.opt.param_groups
        state = []
        for element in s:
            for key in element:
                if key == 'step':
                    pass
                if torch.is_tensor(element[key]):
                    if key == 'exp_avg_sq':
                        state.extend(torch.log10(element[key]).data.numpy())
                    elif key == 'exp_avg':
                        state.extend(torch.sign(element[key]).data.numpy())
                        state.extend(torch.log10(torch.abs(element[key])).data.numpy())
                    else:
                        state.extend(element[key].data.numpy())
                else:
                    state.append(element[key])

        for group in p:
            for param in group['params']:
                state.extend(torch.sign(param).data.numpy())
                state.extend(torch.log10(torch.abs(param)).data.numpy())
                state.extend(torch.sign(param.grad).data.numpy())
                state.extend(torch.log10(torch.abs(param.grad)).data.numpy())

        state.append(-1)
        if self.__prev_state is None:
            self.__prev_state = np.array(state, dtype=np.float32)
            self.__prev_state[-1] = self.total_steps / self._max_steps
            return self.__prev_state
        else:
            # state = np.subtract(state, self.__prev_state)
            state = np.array(state, dtype=np.float32)
            state[-1] = self.total_steps/self._max_steps
            self.__prev_state = state.copy()
            return state

    def step(self, action):
        start = time.time()
        # if not np.isscalar(action) and action is not None:
        #     action = action[0]
        print("action",action)
        action = action['action']
        print("action is",action)
        print(self.curr_x)
        f_val = self.loss_func(self.curr_x)
        self.opt.zero_grad()
        f_val.backward(torch.ones_like(f_val))
        self.opt.step(learned_stepsize=action)
        self.total_steps += 1
        tmp = np.max([f_val.data.numpy() - self.__opt_val.data.numpy(), 1e-8])
        self.__c_step += 1

        # TODO what is a good reward?
        r = -np.log10(tmp / self.__init_fval, dtype=np.float32)
        r = np.clip(r, -38, 45)  # constants to avoid np.log10 returning np.inf

        self.__prev_fval = tmp.copy()

        done = self.total_steps >= self._max_steps
        self.__sum_r += r

        self.teacher_update_x()
        self.teachers_steps()

        return self._compute_state(), r, done, time.time()-start,{}


import time
s = time.time()

args = parser.parse_args()
e = AdamOnSynthFuncEnv(fixed=True,loss_func=args.loss)
e.register_teacher(Adam)
e.seed(args.run_id)
e.reset()
# for i in range(10):
#     print("in loop")
#     e.step(e.action_space.sample())
e.logger.info(f'{e.observation_space}')
e.logger.info('DONE')

def f(config,budget):
    #print("budget",budget)
    compute_state, r, done,cost,_ = e.step(config)
    #, _= res
    print("compute state",compute_state)
    print("r:",r)
    print("done:",done)
    with open('dehb_run_'+str(args.run_id)+'_'+str(args.loss)+'.json', 'a+')as f:
        json.dump({'configuration': dict(config), 'r': r,'compute state':str(compute_state),'cost':cost}, f)
        f.write("\n")
    return r, cost





import ConfigSpace as CS
cs = CS.ConfigurationSpace()
val = UniformFloatHyperparameter('action', 1e-5, 5, default_value=0.001)
cs.add_hyperparameters([val])
# Initializing DEHB object
# for i in range(10):
#     print("sample",e.action_space.sample(),"cs sample",cs.sample_configuration()['action'])
dehb = DEHB(cs=cs, f=f, strategy=args.strategy,
            mutation_factor=args.mutation_factor, crossover_prob=args.crossover_prob,
            eta=args.eta, min_budget=1, max_budget=200,
            generations=args.gens, boundary_fix_type=args.boundary_fix_type)
dehb.run(iterations=1000)
