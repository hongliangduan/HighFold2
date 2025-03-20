import os
import gc

# os.environ['TF_FORCE_UNIFIED_MEMORY'] = '1'
os.environ['CUDA_VISIBLE_DEVICES']='1'
# os.environ['TF_FORCE_UNIFIED_MEMORY']='1'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION']='.80'
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]='platform'
# os.environ['XLA_FLAGS']='--xla_gpu_force_compilation_parallelism=1'

import jax
import optax
import pandas as pd
import torch
from data_process import CustomPandasDataset, create_batch_from_dataset
from alphafold.model import data, config, model
import haiku as hk
import jax.numpy as jnp
import numpy as np
from utils import flatten_dict, norm_grads_per_example, collate_fn, save_pdb, del_pdbs

jax.tree_multimap = jax.tree_map 

model_name = 'model_5_multimer_v3'
data_dir = 'alphafold_multimer'

jax_key = jax.random.PRNGKey(0)
model_config = config.model_config(model_name)
model_config.model.num_ensemble_train = 1
model_config.model.global_config.use_remat = True

class PoolLayer(hk.Module):
    def __init__(self, head=8):
        super(PoolLayer, self).__init__()
        # self.bond_qury = hk.Sequential([
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        # ])
        # self.bond_key = hk.Sequential([
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        # ])
        # self.bond_value = hk.Sequential([
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        #     jax.nn.relu,
        #     hk.Linear(6),
        # ])
        self.head = head
        self.layernorm = hk.LayerNorm(axis=1, create_scale=True, create_offset=True)
    
        self.linear1 = hk.Sequential([
            hk.Linear(6),
            jax.nn.relu,
            hk.Linear(6),
            jax.nn.relu,
            hk.Linear(1),
        ])
        self.linear2 = hk.Sequential([
            hk.Linear(1),
            jax.nn.relu,
            hk.Linear(64),
            jax.nn.relu,
            hk.Linear(128),
        ])
        self.atom_qury = hk.Sequential([
            hk.Linear(16),
            jax.nn.relu,
            hk.Linear(24),
            jax.nn.relu,
            hk.Linear(32),
        ])
        self.atom_key = hk.Sequential([
            hk.Linear(16),
            jax.nn.relu,
            hk.Linear(24),
            jax.nn.relu,
            hk.Linear(32),
        ])
        self.atom_value = hk.Sequential([
            hk.Linear(16),
            jax.nn.relu,
            hk.Linear(24),
            jax.nn.relu,
            hk.Linear(32),
        ])
        self.linear3 = hk.Sequential([
            hk.Linear(24),
            jax.nn.relu,
            hk.Linear(16),
            jax.nn.relu,
            hk.Linear(11),
        ])
        self.linear4 = hk.Sequential([
            hk.Linear(11),
            jax.nn.relu,
            hk.Linear(11),
            jax.nn.relu,
            hk.Linear(21),
        ])

    def __call__(self, bond_feat, atom_feat, M):
        F = (1.0 - M + 1e-6) / (M - 1e-6)
        
        # N, N, n = bond_feat.shape
        # bond_feat = jnp.reshape(bond_feat, (N*N, 6))
        # bond_qury = self.bond_qury(bond_feat) # [N*N, 6]
        # bond_key = self.bond_key(bond_feat) # [N*N, 6]
        # bond_value = self.bond_value(bond_feat) # [N*N, 6]
        # score = jax.nn.softmax(jnp.matmul(bond_qury, bond_key.T) / jnp.sqrt(6)) # [N*N, N*N]
        # bond = jnp.matmul(score, bond_value) # [N*N, 6]
        # bond = jnp.reshape(bond, (N, N, 6))
        
        bond = self.linear1(bond_feat) # [N, N, 1]
        Ms_bond1 = jax.nn.softmax(jnp.transpose(bond, (0, 2, 1)) + F[:, :, None], axis=0)  # [N, Nr, N]
        bond_pool = jnp.matmul(bond.transpose(0, 2, 1), Ms_bond1.transpose(0, 2, 1))  # [N, 1, Nr]

        Ms_bond2 = jax.nn.softmax(bond_pool + F[:, :, None], axis=0) # [N, Nr, Nr]
        bond_pool = jnp.matmul(bond_pool.transpose(2, 1, 0), Ms_bond2.transpose(1, 0, 2)) # [Nr, 1, Nr]
        bond_pool = self.linear2(bond_pool.transpose(0, 2, 1)) # [Nr, Nr, 1]
        # bond_pool = jnp.squeeze(bond_pool, axis=2)

        x_clone = atom_feat
        atom_qury = self.atom_qury(atom_feat) # [N, 32]
        atom_key = self.atom_key(atom_feat) # [N, 32]
        atom_value = self.atom_value(atom_feat) # [N, 32]

        head_dim = atom_qury.shape[-1] // self.head
        Q = atom_qury.reshape(-1, self.head, head_dim).transpose(1, 0, 2)  # [head, N, head_dim]
        K = atom_key.reshape(-1, self.head, head_dim).transpose(1, 0, 2)  # [head, N, head_dim]
        V = atom_value.reshape(-1, self.head, head_dim).transpose(1, 0, 2)  # [head, N, head_dim]
        
        score = jax.nn.softmax(jnp.matmul(Q, K.transpose(0, 2, 1)) / jnp.sqrt(head_dim)) # [head, N, N]
        atom = jnp.matmul(score, V) # [head, N, head_dim]
        atom = atom.transpose(1, 0, 2).reshape(-1, atom_qury.shape[-1])  # [N, 32]
        atom = self.linear3(atom) # [N, 11]
        # atom = atom + x_clone
        # atom = self.layernorm(atom)
        atom = self.linear4(atom)
        Ms_atom = jax.nn.softmax(atom[:, None, :] + F[:, :, None], axis=0) # [N,Nr,21]
        atom_pool = jnp.matmul(atom.T[:, None, :], Ms_atom.transpose(2, 0, 1)) # [21,1,Nr]
        atom_pool = jnp.squeeze(atom_pool, axis=1).T
        return bond_pool, atom_pool


def pool_fn(bond_feat, atom_feat, M):
    model = PoolLayer()
    return model(bond_feat, atom_feat, M)

pool = hk.transform(pool_fn, apply_rng=True)

rng = jax.random.PRNGKey(42)
bond_feat_init = jnp.ones((18, 18, 6))
atom_feat_init = jnp.ones((18, 11))
M_init = jnp.ones((18, 2))
pool_params = pool.init(rng, bond_feat_init, atom_feat_init, M_init)

af2_model_params = data.get_model_haiku_params(model_name=model_name, data_dir=data_dir)

model_params = hk.data_structures.merge(af2_model_params, pool_params)
model_runner = model.RunModel(model_config, af2_model_params, is_training=True, compute_loss=True)

def get_loss_fn(model_params, key, processed_feature_dict):
    pool_params, af2_model_param = hk.data_structures.partition(lambda m, n, p: m[:9] != "alphafold", model_params)
    # print(pool_params)
    L = len(processed_feature_dict['aatype'])
    bond_rep_plus_init = jnp.zeros((L, L, 128), dtype = 'bfloat16')
    atom_rep_plus_init = jnp.zeros((L, 21), dtype = 'bfloat16')
    bond_rep_plus = bond_rep_plus_init
    atom_rep_plus = atom_rep_plus_init
    for key in processed_feature_dict['ligand_feats_dict']:
        start, end = processed_feature_dict['ligand_feats_dict'][key]['start_and_end']
        bond_feat = processed_feature_dict['ligand_feats_dict'][key]['bond_feat']
        atom_feat = processed_feature_dict['ligand_feats_dict'][key]['atom_feat']
        mask = processed_feature_dict['ligand_feats_dict'][key]['mask']
        bond_rep, atom_rep = jax.jit(pool.apply)(pool_params, rng, bond_feat, atom_feat, mask)
        bond_rep_plus = bond_rep_plus.at[start:end,start:end,:].set(bond_rep.astype('bfloat16'))
        # bond_rep_plus = bound_rep_plus_init.at[start:end,start:end,0].set(bond_rep.astype('bfloat16'))
        atom_rep_plus = atom_rep_plus.at[start:end,:].set(atom_rep.astype('bfloat16'))
    processed_feature_dict['bond_rep_plus'] = bond_rep_plus
    processed_feature_dict['atom_rep_plus'] = atom_rep_plus
    del processed_feature_dict['ligand_feats_dict']
    del processed_feature_dict['deletion_mean']
    del processed_feature_dict['entity_mask']
    loss, predicted_dict = model_runner.apply(af2_model_param, key, processed_feature_dict)
    loss = loss[0]
    if processed_feature_dict['is_continuous']:
        # print('is_continuous')
        loss = loss
    else:
        # print('is_discrete')
        loss = loss * 0.5
    del processed_feature_dict['is_continuous']
    return loss, predicted_dict


def train_step(model_params, key, batch):
    (loss, predicted_dict), grads = jax.value_and_grad(get_loss_fn, has_aux=True)(model_params, key, batch)
    return loss, grads, predicted_dict

# load data
train_df = pd.read_csv('./train_set.csv')
valid_df = pd.read_csv('./valid_set.csv')
training_set  = CustomPandasDataset(train_df, create_batch_from_dataset, 220)
valid_set  = CustomPandasDataset(valid_df, create_batch_from_dataset)
train_params_loader = {
    'shuffle': True,
    'pin_memory': False,
    'batch_size': 1,
    'num_workers': 0,
    'collate_fn': collate_fn,
}
valid_params_loader = {
    'shuffle': False,
    'pin_memory': False,
    'batch_size': 1,
    'num_workers': 0,
    'collate_fn': collate_fn,
}
train_loader = torch.utils.data.DataLoader(training_set, **train_params_loader)
valid_loader = torch.utils.data.DataLoader(valid_set, **valid_params_loader)

# optimizer
# scheduler = optax.linear_schedule(1e-3, 1e-3, 1000, 0)
# lr_coef = 0.025
# chain_me = [
#     optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),
#     optax.scale_by_schedule(scheduler),
#     optax.scale(-1.0*lr_coef),
# ]
lr = 7.5e-5
chain_me = [
    optax.scale_by_adam(b1=0.9, b2=0.999, eps=1e-6),
    optax.scale(-1.0 * lr),
]
gradient_transform = optax.chain(*chain_me)
replicated_params = jax.tree_map(lambda x: jnp.array(x), model_params)
opt_state = gradient_transform.init(replicated_params)
# opt_state = gradient_transform.init(pool_params)

def grad_update(grads_sum, grads_sum_count, opt_state, replicated_params):
    grads_sum = jax.tree_map(lambda x: x/grads_sum_count, grads_sum)
    grads_sum = norm_grads_per_example(grads_sum, l2_norm_clip=0.1)
    updates, opt_state = gradient_transform.update(grads_sum, opt_state)
    replicated_params = optax.apply_updates(replicated_params, updates)
    grads_sum, grads_sum_count = None, 0

# train
valid_set_losses = []
training_set_losses = []
lowest_valid_loss = 1e6
global_step = 0
for e in range(30):
    grads_sum, grads_sum_count = None, 0
    training_set_loss = 0

    for n, batch in enumerate(train_loader):
        jax_key, subkey = jax.random.split(jax_key)
        del batch['pdbid']
        loss, grads, predicted_dict = train_step(replicated_params, subkey, batch)
        training_set_loss += loss
        global_step += 1
        print(f'sample {n} loss: {loss}')

        # grads = norm_grads_per_example(grads, l2_norm_clip=10)
        # updates, opt_state = gradient_transform.update(grads, opt_state)
        # replicated_params = optax.apply_updates(replicated_params, updates)

        # gradient accumulation
        if grads_sum_count == 0:
            grads_sum = grads
        else:
            grads_sum = jax.tree_multimap(lambda x, y: x+y, grads_sum, grads)
        grads_sum_count += 1

        # update params
        if grads_sum_count >= 8:
            grads_sum = jax.tree_map(lambda x: x/grads_sum_count, grads_sum)
            grads_sum = norm_grads_per_example(grads_sum, l2_norm_clip=0.1)
            updates, opt_state = gradient_transform.update(grads_sum, opt_state)
            replicated_params = optax.apply_updates(replicated_params, updates)
            # replicated_params = optax.apply_updates(pool_params, updates)
            grads_sum, grads_sum_count = None, 0
        # jax.clear_backends()
        # gc.collect()
        # del batch
        batch.clear()

        # validation
        if global_step % 120 == 0:
            valid_set_loss = 0
            for n, batch in enumerate(valid_loader):
                pdbid = batch['pdbid']
                del batch['pdbid']
                loss, grads, predicted_dict = train_step(replicated_params, subkey, batch)
                valid_set_loss += loss
                save_pdb(predicted_dict, batch, './validation_predictions', global_step, pdbid)
                # jax.clear_backends()
                # gc.collect()
                # del batch
            
            print(f'validation set loss: {valid_set_loss}')
            valid_set_losses.append(valid_set_loss)
            loss_dict = {'training_set_loss': np.array(training_set_losses), 'valid_set_loss': np.array(valid_set_losses)}
            np.save('loss_dict.npy', loss_dict)


            # save params
            if valid_set_loss < lowest_valid_loss:
                lowest_valid_loss = valid_set_loss
                save_steps = global_step

                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')

                param_fname = f'./checkpoints/params_model_5_multimer_v3_{global_step}_unnatural.npz'
                save_params = jax.tree_map(lambda x: np.asarray(x), replicated_params)
                flattened_params = flatten_dict(save_params)
                np.savez(param_fname, **flattened_params)

            del_pdbs('./validation_predictions', save_steps)



    print(f'training set loss: {training_set_loss}')
    training_set_losses.append(training_set_loss)
    loss_dict = {'training_set_loss': np.array(training_set_losses), 'valid_set_loss': np.array(valid_set_losses)}
    np.save('loss_dict.npy', loss_dict)