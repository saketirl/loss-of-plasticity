import os
import yaml
import pickle
import argparse
import subprocess
import numpy as np

import gym
import torch
from torch.optim import Adam

from lop.algos.rl.buffer import Buffer
from lop.nets.policies import MLPPolicy
from lop.nets.valuefs import MLPVF
from lop.algos.rl.agent import Agent
from lop.algos.rl.ppo import PPO
from datetime import datetime



def average_weighted_tau(tau, model1_dict, model2_dict, device):
    combined_state_dict = {}

    for key in model1_dict.keys():
        if key in model2_dict.keys():
            param1 = model1_dict[key].to(device)
            param2 = model2_dict[key].to(device)
            combined_param = (1 - tau) * param1 + tau * param2
            combined_state_dict[key] = combined_param
        else:
            raise KeyError(f"Parameter {key} not found in both critic checkpoints.")

    return combined_state_dict


def load_checkpoint(tau, path1, path2, device, learner):
    # Load step, model and optimizer states
    ckpt_dict_1 = torch.load(path1, map_location=device)
    ckpt_dict_2 = torch.load(path2, map_location=device)
    actor_tau = average_weighted_tau(tau, ckpt_dict_1['actor'], ckpt_dict_2['actor'], device)
    critic_tau = average_weighted_tau(tau, ckpt_dict_1['critic'], ckpt_dict_2['critic'], device)
    opt_dict = ckpt_dict_1['opt']
    step = ckpt_dict_1['step']
    learner.pol.load_state_dict(actor_tau)
    learner.vf.load_state_dict(critic_tau)
    learner.opt.load_state_dict(opt_dict)
    return step, learner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=False, type=str, default='./cfg/ant/cbp.yml')
    parser.add_argument('-s', '--seed', required=False, type=int, default="1")
    parser.add_argument('-d', '--device', required=False, default='cuda')
    parser.add_argument('-m', '--model1', required=False, default='data/ant/cbp_1/1.pth')
    parser.add_argument('-f', '--model2', required=False, default='data/ant/cbp_2/2.pth')
    parser.add_argument('-n', '--numsteps', required=False, type=int, default=10000)
    parser.add_argument('-o', '--output', required=False, default='data/ant/cbp_eval/eval_1_2.pkl')
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

    tau_start = -0.25
    tau_end = 1.25
    tau_spacing = 0.01
    tau_array = np.arange(tau_start, tau_end + tau_spacing, tau_spacing)
    epi_steps = 0
    print(len(tau_array))
    print("******************************")

    rets_data = []
    jacobian_values_data = []
    jacobian_actions_data = []
    value_estimates_data = []

    for idx, spaced_tau in enumerate(tau_array):
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

        step, learner = load_checkpoint(spaced_tau, args.model1, args.model2, args.device, learner)

        agent = Agent(learner.pol, learner, device=device)
        ret = 0.0
        rets = []
        termination_steps = []
        o = env.reset()

        samples = []

        samples.append(o)

        for step in range(num_steps):
            a, logp, dist, new_features = agent.get_action(o)
            op, r, done, infos = env.step(a)
            epi_steps += 1

            o = op
            samples.append(o)
            ret += r
            if done:
                # print(step, "(", epi_steps, ") {0:.2f}".format(ret))
                rets.append(ret)
                termination_steps.append(step)
                ret = 0
                epi_steps = 0
                o = env.reset()

        ret_np = np.array(rets)

        rets_data.append([spaced_tau, np.mean(ret_np), np.std(ret_np)])

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
        jacobian_values_data.append([spaced_tau, np.mean(jacobian_values), np.std(jacobian_values)])
        jacobian_actions_data.append([spaced_tau, np.mean(jacobian_values),  np.mean(jacobian_policy_norms, axis=0),
                                      np.std(jacobian_policy_norms, axis=0)])
        value_estimates_data.append([spaced_tau, np.mean(value_estimates_data), np.std(value_estimates)])


        if idx % 20 == 0:
            # datetime object containing current date and time
            now = datetime.now()
            # dd/mm/YY H:M:S
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

            print("{} Done with tau: {}".format(dt_string, spaced_tau))

    estimated_vals = {"returns": rets_data,
                      "action_jacobians": jacobian_actions_data,
                      "values_jacobian": jacobian_values_data,
                      "value_estimates": value_estimates_data}

    with open(args.output, 'wb') as f_out:
        pickle.dump(estimated_vals, f_out, pickle.HIGHEST_PROTOCOL)
        print("Saved data at: {}".format(args.output))


if __name__ == "__main__":
    main()
