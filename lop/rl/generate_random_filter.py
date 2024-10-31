import os
import copy
import yaml
import pickle
import argparse
import subprocess
import numpy as np

import gymnasium as gym
import torch
from torch.optim import Adam

from lop.algos.rl.buffer import Buffer
from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.algos.rl.agent import Agent
from lop.algos.rl.ppo import PPO
from datetime import datetime
from itertools import product


def get_weights(net):
    """ Extract parameters from net, and return a list of tensors"""
    return [p.data for p in net.parameters()]


def get_random_weights(weights):
    """
        Produce a random direction that is a list of random Gaussian tensors
        with the same shape as the network's weights, so one direction entry per weight.
    """
    return [torch.randn(w.size(), device=w.device) for w in weights]


def normalize_direction(direction, weights, norm='filter'):
    """
        Rescale the direction so that it has similar norm as their corresponding
        model in different levels.

        Args:
          direction: a variables of the random direction for one layer
          weights: a variable of the original model for one layer
          norm: normalization method, 'filter' | 'layer' | 'weight'
    """
    if norm == 'filter':
        # Rescale the filters (weights in group) in 'direction' so that each
        # filter has the same norm as its corresponding filter in 'weights'.
        for d, w in zip(direction, weights):
            d.mul_(w.norm()/(d.norm() + 1e-10))
    elif norm == 'layer':
        # Rescale the layer variables in the direction so that each layer has
        # the same norm as the layer variables in weights.
        direction.mul_(weights.norm()/direction.norm())
    elif norm == 'weight':
        # Rescale the entries in the direction so that each entry has the same
        # scale as the corresponding weight.
        direction.mul_(weights)
    elif norm == 'dfilter':
        # Rescale the entries in the direction so that each filter direction
        # has the unit norm.
        for d in direction:
            d.div_(d.norm() + 1e-10)
    elif norm == 'dlayer':
        # Rescale the entries in the direction so that each layer direction has
        # the unit norm.
        direction.div_(direction.norm())


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
    """
        The normalization scales the direction entries according to the entries of weights.
    """
    assert(len(direction) == len(weights))
    for d, w in zip(direction, weights):
        if d.dim() <= 1:
            if ignore == 'biasbn':
                d.fill_(0) # ignore directions for weights with 1 dimension
            else:
                d.copy_(w) # keep directions for weights/bias that are only 1 per node
        else:
            normalize_direction(d, w, norm)


def create_random_direction(net,
                            # dir_type='weights',
                            ignore='biasbn', norm='filter'):
    """
        Setup a random (normalized) direction with the same dimension as
        the weights or states.

        Args:
          net: the given trained model
          dir_type: 'weights' or 'states', type of directions.
          ignore: 'biasbn', ignore biases and BN parameters.
          norm: direction normalization method, including
                'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

        Returns:
          direction: a random direction with the same dimension as weights or states.
    """

    # random direction
    # if dir_type == 'weights':
    weights = get_weights(net)  # a list of parameters.
    direction = get_random_weights(weights)
    normalize_directions_for_weights(direction, weights, norm, ignore)
    # elif dir_type == 'states':
    #     states = net.state_dict() # a dict of parameters, including BN's running mean/var.
    #     direction = get_random_states(states)
    #     normalize_directions_for_states(direction, states, norm, ignore)

    return direction


def load_checkpoint(path, device, learner):
    # Load step, model and optimizer states
    ckpt_dict = torch.load(path, map_location=device)
    opt_dict = ckpt_dict['opt']
    step = ckpt_dict['step']
    learner.pol.load_state_dict(ckpt_dict['actor'])
    learner.vf.load_state_dict(ckpt_dict['critic'])
    learner.opt.load_state_dict(opt_dict)
    return step, learner


def copy_and_set_weights(net, weights, directions=None, step=None):
    """
        Copy the network's weights and add it with a specified list of tensors
        or change weights along directions with a step size.
    """
    new_net = copy.deepcopy(net)
    if directions is None:
        # You cannot specify a step length without a direction.
        for (p, w) in zip(new_net.parameters(), weights):
            p.data.copy_(w.type(type(p.data)))
    else:
        assert step is not None, 'If a direction is specified then step must be specified as well'

        if len(directions) == 2:
            dx = directions[0]
            dy = directions[1]
            changes = [d0*step[0] + d1*step[1] for (d0, d1) in zip(dx, dy)]
        else:
            changes = [d*step for d in directions[0]]

        for (p, w, d) in zip(new_net.parameters(), weights, changes):
            p.data = w + torch.tensor(d).type(type(w)).to(w.device)
    return new_net

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, default='./cfg/ant/cbp.yml')
    parser.add_argument('-s', '--seed', required=False, type=int, default="1")
    parser.add_argument('-d', '--device', required=False, default='cuda')
    parser.add_argument('-m', '--model', required=False, default='data/ant/cbp_1/1.pth')
    parser.add_argument('-n', '--numsteps', required=False, type=int, default=10000)
    parser.add_argument('-o', '--output', required=False, default='data/ant/cbp_eval/random_filters_1.pkl')
    parser.add_argument('-b', '--batchsize', required=False, type=int, default=128)

    args = parser.parse_args()

    num_steps = args.numsteps

    if args.device: device = args.device
    else: device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = yaml.safe_load(open(args.config))
    cfg['seed'] = args.seed
    cfg['log_path'] = cfg['dir'] + str(args.seed) + '.log'
    cfg['ckpt_path'] = cfg['dir'] + str(args.seed) + '.pth'
    cfg['done_path'] = cfg['dir'] + str(args.seed) + '.done'

    bash_command = "mkdir -p " + cfg['dir']
    subprocess.Popen(bash_command.split(), stdout=subprocess.PIPE)

    # Set default values
    cfg.setdefault('wd', 0)
    cfg.setdefault('init', 'lecun')
    cfg.setdefault('to_log', [])
    cfg.setdefault('beta_1', 0.9)
    cfg.setdefault('beta_2', 0.999)
    cfg.setdefault('eps', 1e-8)
    cfg.setdefault('no_clipping', False)
    cfg.setdefault('loss_type', 'ppo')
    cfg.setdefault('frictions_file', 'cfg/frictions')
    cfg.setdefault('max_grad_norm', 1e9)
    cfg.setdefault('perturb_scale', 0)
    cfg['n_steps'] = int(float(cfg['n_steps']))
    cfg['perturb_scale'] = float(cfg['perturb_scale'])
    n_steps = cfg['n_steps']

    # Set default values for CBP
    cfg.setdefault('mt', 10000)
    cfg.setdefault('rr', 0)
    cfg['rr'] = float(cfg['rr'])
    cfg.setdefault('decay_rate', 0.99)
    cfg.setdefault('redo', False)
    cfg.setdefault('threshold', 0.03)
    cfg.setdefault('reset_period', 1000)
    cfg.setdefault('util_type_val', 'contribution')
    cfg.setdefault('util_type_pol', 'contribution')
    cfg.setdefault('pgnt', (cfg['rr']>0) or cfg['redo'])
    cfg.setdefault('vgnt', (cfg['rr']>0) or cfg['redo'])

    # Initialize env
    seed = cfg['seed']
    friction = -1.0
    if cfg['env_name'] in ['SlipperyAnt-v2', 'SlipperyAnt-v3']:
        xml_file = os.path.abspath(cfg['dir'] + f'slippery_ant_{seed}.xml')
        cfg.setdefault('friction', [0.02, 2])
        cfg.setdefault('change_time', int(2e6))

        with open(cfg['frictions_file'], 'rb+') as f:
            frictions = pickle.load(f)
        friction_number = 0
        new_friction = frictions[seed][friction_number]

        if friction < 0:  # If no saved friction, use the default value 1.0
            friction = 1.0
        env = gym.make(cfg['env_name'], friction=new_friction, xml_file=xml_file)
        print(f'Initial friction: {friction:.6f}')
    else:
        env = gym.make(cfg['env_name'])
    env.name = None

    np.random.seed(seed)
    random_state = np.random.get_state()
    torch_seed = np.random.randint(1, 2 ** 31 - 1)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

    # Initialize algorithm
    opt = Adam
    num_layers = len(cfg['h_dim'])
    o_dim = env.observation_space.shape[0]
    a_dim = env.action_space.shape[0]
    pol = MLPPolicy(o_dim, a_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg['init'])
    vf = MLPVF(o_dim, act_type=cfg['act_type'], h_dim=cfg['h_dim'], device=device, init=cfg['init'])
    np.random.set_state(random_state)
    buf = Buffer(o_dim, a_dim, cfg['bs'], device=device)

    learner = PPO(pol, buf, cfg['lr'], g=cfg['g'], vf=vf, lm=cfg['lm'], Opt=opt,
                  u_epi_up=cfg['u_epi_ups'], device=device, n_itrs=cfg['n_itrs'], n_slices=cfg['n_slices'],
                  u_adv_scl=cfg['u_adv_scl'], clip_eps=cfg['clip_eps'],
                  max_grad_norm=cfg['max_grad_norm'], init=cfg['init'],
                  wd=float(cfg['wd']),
                  betas=(cfg['beta_1'], cfg['beta_2']), eps=float(cfg['eps']), no_clipping=cfg['no_clipping'],
                  loss_type=cfg['loss_type'], perturb_scale=cfg['perturb_scale'],
                  util_type_val=cfg['util_type_val'], replacement_rate=cfg['rr'], decay_rate=cfg['decay_rate'],
                  vgnt=cfg['vgnt'], pgnt=cfg['pgnt'], util_type_pol=cfg['util_type_pol'], mt=cfg['mt'],
                  redo=cfg['redo'], threshold=cfg['threshold'], reset_period=cfg['reset_period']
                  )

    print("Loading checkpoint")
    step, learner = load_checkpoint(args.model, args.device, learner)

    tau_start = -1
    tau_end = 1
    # DEBUGGING
    n_taus = 20

    pol_x_direction = create_random_direction(learner.pol)
    pol_y_direction = create_random_direction(learner.pol)
    weights = get_weights(learner.pol)  # a list of parameters.
    pol_directions = [pol_x_direction, pol_y_direction]

    # val_x_direction = create_random_direction(learner.vf)
    # val_y_direction = create_random_direction(learner.vf)

    tau_array = np.linspace(tau_start, tau_end, num=n_taus)
    taus_array = list(product(tau_array, tau_array))
    epi_steps = 0
    # print(len(tau_array))
    print("******************************")

    rets_data = []
    jacobian_values_data = []
    jacobian_actions_data = []
    value_estimates_data = []

    for idx, taus in enumerate(taus_array):
        perturbed_pol = copy_and_set_weights(learner.pol, weights, directions=pol_directions, step=taus)

        agent = Agent(perturbed_pol, learner, device=device)
        ret = 0.0
        rets = []
        termination_steps = []
        o, _ = env.reset()

        samples = []

        samples.append(o)

        for step in range(num_steps):
            a, logp, dist, new_features = agent.get_action(o)
            op, r, done, trunc, infos = env.step(a)
            epi_steps += 1

            o = op
            samples.append(o)
            ret += r
            if done or trunc:
                # print(step, "(", epi_steps, ") {0:.2f}".format(ret))
                rets.append(ret)
                termination_steps.append(step)
                ret = 0
                epi_steps = 0
                o, _ = env.reset()

        ret_np = np.array(rets)

        rets_data.append([taus, np.mean(ret_np), np.std(ret_np)])

        jacobian_policy_norms = []
        jacobian_values = []
        value_estimates = []

        for i in range(len(samples)):
            o_inputs = np.array(samples[i])
            o_inputs = torch.from_numpy(o_inputs).float().to(device)
            o_inputs.requires_grad = True

            vals = learner.vf.value(o_inputs)
            actions_preds = learner.pol.mean_net(o_inputs)

            for action_idx in range(a_dim):
                actions_by_idx = actions_preds[action_idx]
                jacobian_action = torch.autograd.grad(actions_by_idx, o_inputs, create_graph=True)[0]
                jacobian_action_idx = torch.norm(jacobian_action, p=2).detach().cpu().numpy()
                if len(jacobian_policy_norms) < action_idx + 1:
                    jacobian_policy_norms.append(jacobian_action_idx)
                else:
                    jacobian_policy_norms[action_idx] = np.append(jacobian_policy_norms[action_idx],
                                                                  jacobian_action_idx)

            jacobian_value = torch.autograd.grad(vals, o_inputs, create_graph=True)[0]
            jacobian_value_norms = torch.norm(jacobian_value, p=2).detach().cpu().numpy()
            jacobian_values = np.append(jacobian_values, jacobian_value_norms)

            value_estimates = np.append(value_estimates, vals.detach().cpu().numpy())

        #Convert to np array
        jacobian_values = np.array(jacobian_values)
        jacobian_policy_norms = np.array(jacobian_policy_norms)
        value_estimates = np.array(value_estimates)

        #get mean and std and append to the data array
        jacobian_values_data.append([*taus, np.mean(jacobian_values), np.std(jacobian_values)])
        jacobian_actions_data.append([taus, np.mean(jacobian_values),  np.mean(jacobian_policy_norms, axis=0),
                                      np.std(jacobian_policy_norms, axis=0)])
        value_estimates_data.append([taus, np.mean(value_estimates), np.std(value_estimates)])


        if idx % 20 == 0:
            # datetime object containing current date and time
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

            print("{} Done with tau: {}".format(dt_string, taus))

    estimated_vals = {"returns": rets_data,
                      "action_jacobians": jacobian_actions_data,
                      "values_jacobian": jacobian_values_data,
                      "value_estimates": value_estimates_data}

    with open(args.output, 'wb') as f_out:
        pickle.dump(estimated_vals, f_out, pickle.HIGHEST_PROTOCOL)
        print("Saved data at: {}".format(args.output))

