import torch
import numpy as np
import os
import json
import random
import augmentations
from datetime import datetime


class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(
			tau * param.data + (1 - tau) * target_param.data
		)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'timestamp': str(datetime.now()),
		'args': str(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f)


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def array_init(capacity, dims, dtype):
	"""Preallocate array in memory"""
	chunks = 20
	zero_dim_size = int(capacity / chunks)
	array = np.zeros((capacity, *dims), dtype=dtype)
	temp = np.ones((zero_dim_size, *dims), dtype=dtype)
	
	for i in range(chunks):
		array[i*zero_dim_size:(i+1)*zero_dim_size] = temp

	return array


class ReplayBuffer(object):
	"""Buffer to store environment transitions"""
	def __init__(self, obs_shape, action_shape, capacity, batch_size):
		self.capacity = capacity
		self.batch_size = batch_size

		self.obs = array_init(capacity, obs_shape, dtype=np.uint8)
		self.next_obs = array_init(capacity, obs_shape, dtype=np.uint8)
		self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self.rewards = np.empty((capacity, 1), dtype=np.float32)
		self.not_dones = np.empty((capacity, 1), dtype=np.float32)

		self.idx = 0
		self.full = False

	def add(self, obs, action, reward, next_obs, done):
		np.copyto(self.obs[self.idx], obs)
		np.copyto(self.actions[self.idx], action)
		np.copyto(self.rewards[self.idx], reward)
		np.copyto(self.next_obs[self.idx], next_obs)
		np.copyto(self.not_dones[self.idx], not done)

		self.idx = (self.idx + 1) % self.capacity
		self.full = self.full or self.idx == 0

	def _get_idxs(self, n=None):
		if n is None:
			n = self.batch_size
		return np.random.randint(
			0, self.capacity if self.full else self.idx, size=n
		)

	def sample_soda(self, n=None):
		return torch.as_tensor(self.obs[self._get_idxs(n)]).cuda().float()

	def sample_curl(self, n=None):
		idxs = self._get_idxs(n)

		obs = torch.as_tensor(self.obs[idxs]).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		next_obs = torch.as_tensor(self.next_obs[idxs]).cuda().float()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		pos = augmentations.random_crop(obs.clone())
		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones, pos

	def sample(self, n=None):
		idxs = self._get_idxs(n)

		obs = torch.as_tensor(self.obs[idxs]).cuda().float()
		actions = torch.as_tensor(self.actions[idxs]).cuda()
		rewards = torch.as_tensor(self.rewards[idxs]).cuda()
		next_obs = torch.as_tensor(self.next_obs[idxs]).cuda().float()
		not_dones = torch.as_tensor(self.not_dones[idxs]).cuda()

		obs = augmentations.random_crop(obs)
		next_obs = augmentations.random_crop(next_obs)

		return obs, actions, rewards, next_obs, not_dones
