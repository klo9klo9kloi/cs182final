"""Utility functions that are useful for implementing DQN.
"""
import gym
import numpy as np
import random
import math

def sample_n_unique(sampling_f, n):
    """Helper function. Given a function `sampling_f` that returns comparable
    objects, sample n such unique objects.
    """
    res = []
    while len(res) < n:
        candidate = sampling_f()
        if candidate not in res:
            res.append(candidate)
    return res


def linear_interpolation(l, r, alpha):
    return l + alpha * (r - l)

def minimize_and_clip(optimizer, objective, var_list, clip_val=10):
    """Minimized `objective` using `optimizer` w.r.t. variables in `var_list`
    while ensure the norm of the gradients for each variable is clipped to
    `clip_val`.
    """
    gradients = optimizer.compute_gradients(objective, var_list=var_list)
    for i, (grad, var) in enumerate(gradients):
        if grad is not None:
            gradients[i] = (tf.clip_by_norm(grad, clip_val), var)
    return optimizer.apply_gradients(gradients)


def get_wrapper_by_name(env, classname):
    currentenv = env
    while True:
        if classname in currentenv.__class__.__name__:
            return currentenv
        elif isinstance(env, gym.Wrapper):
            currentenv = currentenv.env
        else:
            raise ValueError("Couldn't find wrapper named %s"%classname)


# ------------------------------------------------------------------------------
# SCHEDULES
# ------------------------------------------------------------------------------

class Schedule(object):

    def value(self, t):
        """Value of the schedule at time t"""
        raise NotImplementedError()


class ConstantSchedule(object):

    def __init__(self, value):
        """Value remains constant over time.

        Parameters
        ----------
        value: float
            Constant value of the schedule
        """
        self._v = value

    def value(self, t):
        """See Schedule.value"""
        return self._v


class LinearSchedule(object):

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p            = final_p
        self.initial_p          = initial_p

    def value(self, t):
        """See Schedule.value"""
        fraction  = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)


class PiecewiseSchedule(object):

    def __init__(self, endpoints, interpolation=linear_interpolation, outside_value=None):
        """Piecewise schedule.
        
        Parameters
        ----------
        endpoints: [(int, int)]
            list of pairs `(time, value)` meanining that schedule should output
            `value` when `t==time`. All the values for time must be sorted in
            an increasing order. When t is between two times, e.g. `(time_a, value_a)`
            and `(time_b, value_b)`, such that `time_a <= t < time_b` then value outputs
            `interpolation(value_a, value_b, alpha)` where alpha is a fraction of
            time passed between `time_a` and `time_b` for time `t`.
        interpolation: lambda float, float, float: float
            a function that takes value to the left and to the right of t according
            to the `endpoints`. Alpha is the fraction of distance from left endpoint to
            right endpoint that t has covered. See linear_interpolation for example.
        outside_value: float
            if the value is requested outside of all the intervals sepecified in
            `endpoints` this value is returned. If None then AssertionError is
            raised when outside value is requested.
        """
        idxes = [e[0] for e in endpoints]
        assert idxes == sorted(idxes)
        self._interpolation = interpolation
        self._outside_value = outside_value
        self._endpoints      = endpoints

    def value(self, t):
        """See Schedule.value"""
        for (l_t, l), (r_t, r) in zip(self._endpoints[:-1], self._endpoints[1:]):
            if l_t <= t and t < r_t:
                alpha = float(t - l_t) / (r_t - l_t)
                return self._interpolation(l, r, alpha)

        # t does not belong to any of the pieces, so doom.
        assert self._outside_value is not None
        return self._outside_value

# ------------------------------------------------------------------------------
# REPLAY BUFFER
# ------------------------------------------------------------------------------

class ReplayBuffer(object):

    def __init__(self, size, frame_history_len, cartpole=False):
        """This is a memory efficient implementation of the replay buffer.

        The sepecific memory optimizations use here are:
            - only store each frame once rather than k times
              even if every observation normally consists of k last frames
            - store frames as np.uint8 (actually it is most time-performance
              to cast them back to float32 on GPU to minimize memory transfer
              time)
            - store frame_t and frame_(t+1) in the same buffer.

        For the tipical use case in Atari Deep RL buffer with 1M frames the total
        memory footprint of this buffer is 10^6 * 84 * 84 bytes ~= 7 gigabytes

        Warning! Assumes that returning frame of zeros at the beginning
        of the episode, when there is less frames than `frame_history_len`,
        is acceptable.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        frame_history_len: int
            Number of memories to be retried for each observation.
        cartpole: bool
            If True, CartPole, else Pong
        """
        self.cartpole = cartpole
        self.size = size
        self.frame_history_len = frame_history_len
        self.next_idx = 0
        self.num_in_buffer = 0
        self.obs = None
        self.action = None
        self.reward = None
        self.done = None

    def can_sample(self, batch_size):
        """Returns true if `batch_size` different transitions can be sampled from the buffer."""
        return batch_size + 1 <= self.num_in_buffer

    def _encode_sample(self, idxes):
        obs_batch      = np.concatenate([self._encode_observation(idx)[None] for idx in idxes], 0)
        act_batch      = self.action[idxes]
        rew_batch      = self.reward[idxes]
        next_obs_batch = np.concatenate([self._encode_observation(idx + 1)[None] for idx in idxes], 0)
        done_mask      = np.array([1.0 if self.done[idx] else 0.0 for idx in idxes], dtype=np.float32)

        return obs_batch, act_batch, rew_batch, next_obs_batch, done_mask


    def sample(self, batch_size):
        """Sample `batch_size` different transitions.

        i-th sample transition is the following:

        when observing `obs_batch[i]`, action `act_batch[i]` was taken,
        after which reward `rew_batch[i]` was received and subsequent
        observation  next_obs_batch[i] was observed, unless the epsiode
        was done which is represented by `done_mask[i]` which is equal
        to 1 if episode has ended as a result of that action.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        act_batch: np.array
            Array of shape (batch_size,) and dtype np.int32
        rew_batch: np.array
            Array of shape (batch_size,) and dtype np.float32
        next_obs_batch: np.array
            Array of shape
            (batch_size, img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8
        done_mask: np.array
            Array of shape (batch_size,) and dtype np.float32
        """
        assert self.can_sample(batch_size)
        idxes = sample_n_unique(lambda: random.randint(0, self.num_in_buffer - 2), batch_size)
        return self._encode_sample(idxes)

    def encode_recent_observation(self):
        """Return the most recent `frame_history_len` frames.

        Returns
        -------
        observation: np.array
            Array of shape (img_h, img_w, img_c * frame_history_len)
            and dtype np.uint8, where observation[:, :, i*img_c:(i+1)*img_c]
            encodes frame at time `t - frame_history_len + i`
        """
        assert self.num_in_buffer > 0
        return self._encode_observation((self.next_idx - 1) % self.size)

    # TEMP FIXED: this used to break if the last index in the buffer is sampled and then you try to 
    # encode the next obs state e.g. replay_buffer_size=100, if you sample 99 and then try to encode 100, 
    # there are currently no checks in the code below that fill in the missing context, so it returns self.obs[101-n:101] 
    # which is only n-1 states and not n states.
    # It hasn't been a problem for assignment 4 before because replay buffer size has always been greater than training 
    # timestep budget, and also the sampling method is hardcoded to only return up to index size-2. p
    # probably not a problem for procgen though since a replay buffer of size 1M is still
    # computationally feasible, but just a wierd discovery.
    def _encode_observation(self, idx):
        end_idx   = idx + 1 # make noninclusive
        start_idx = end_idx - self.frame_history_len
        # this checks if we are using low-dimensional observations, such as RAM
        # state, in which case we just directly return the latest RAM.
        if len(self.obs.shape) == 2:
            return self.obs[end_idx-1]
        # if our buffer is still filling up so we can't sample from end
        if start_idx < 0 and self.num_in_buffer != self.size:
            start_idx = 0
        for idx in range(start_idx, end_idx - 1):
            if self.done[idx % self.size]:
                start_idx = idx + 1
        missing_context = self.frame_history_len - (end_idx - start_idx)

        # CASE 1: buffer still filling up and sampled an index near left boundary of buffer OR
        # we sampled an index near a done boundary
        # -> pad missing context with zeros
        if missing_context > 0:
            frames = [np.zeros_like(self.obs[0]) for _ in range(missing_context)]
            for idx in range(start_idx, end_idx):
                frames.append(self.obs[idx % self.size])
            return np.concatenate(frames, 2)
        # CASE 2: buffer is full and sampled an index near left boundary of buffer
        # -> pad from end of buffer
        elif start_idx < 0 and self.num_in_buffer == self.size:
            # print('case 2')
            frames = [self.obs[idx] for idx in range(start_idx, 0)] + [self.obs[idx] for idx in range(end_idx)]
            return np.concatenate(frames, 2)
        # CASE 3: buffer is full and sampled and index near right boundary of buffer
        # and we already looped around enough times
        # -> pad from start of buffer
        elif end_idx > self.size and self.next_idx >= (end_idx - self.size) and self.num_in_buffer == self.size:
            # print('case 3')
            frames = [self.obs[idx] for idx in range(start_idx, self.size)] + [self.obs[idx] for idx in range(end_idx-self.size)]
            return np.concatenate(frames, 2)
        # CASE 4: ^ but we haven't looped around enough times yet
        # -> pad end with zeros
        elif end_idx > self.size: 
            frames = [self.obs[idx] for idx in range(start_idx, self.size)] + [np.zeros_like(self.obs[0]) for _ in range(end_idx-self.size)]
            return np.concatenate(frames, 2)
        else:
            # this optimization has potential to saves about 30% compute time \o/
            img_h, img_w = self.obs.shape[1], self.obs.shape[2]
            return self.obs[start_idx:end_idx].transpose(1, 2, 0, 3).reshape(img_h, img_w, -1)

    def store_frame(self, frame):
        """Store a single frame in the buffer at the next available index, overwriting
        old frames if necessary.

        Parameters
        ----------
        frame: np.array
            Array of shape (img_h, img_w, img_c) and dtype np.uint8
            the frame to be stored

        Returns
        -------
        idx: int
            Index at which the frame is stored. To be used for `store_effect` later.
        """
        self.size = int(self.size)
        if self.obs is None:
            self.obs    = np.empty([self.size] + list(frame.shape), dtype=np.float32 if self.cartpole else np.uint8)
            self.action = np.empty([self.size],                     dtype=np.int32)
            self.reward = np.empty([self.size],                     dtype=np.float32)
            self.done   = np.empty([self.size],                     dtype=np.bool)
        self.obs[self.next_idx] = frame

        ret = self.next_idx
        self.next_idx = (self.next_idx + 1) % self.size
        self.num_in_buffer = min(self.size, self.num_in_buffer + 1)

        return ret

    def store_effect(self, idx, action, reward, done):
        """Store effects of action taken after obeserving frame stored
        at index idx. The reason `store_frame` and `store_effect` is broken
        up into two functions is so that once can call `encode_recent_observation`
        in between.

        Paramters
        ---------
        idx: int
            Index in buffer of recently observed frame (returned by `store_frame`).
        action: int
            Action that was performed upon observing this frame.
        reward: float
            Reward that was received when the actions was performed.
        done: bool
            True if episode was finished after performing that action.
        """
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

class PrioritizedReplayBuffer(ReplayBuffer):
    # SumTree code modeled after https://github.com/google/dopamine/blob/master/dopamine/replay_memory/sum_tree.py
    def __init__(self, size, frame_history_len, alpha, cartpole=False):
        super().__init__(size, frame_history_len, cartpole)

        self.nodes = []
        tree_depth = int(math.ceil(np.log2(size)))
        level_size = 1
        for _ in range(tree_depth + 1):
          nodes_at_this_depth = np.zeros(level_size)
          self.nodes.append(nodes_at_this_depth)

          level_size *= 2

        self.alpha = alpha
        self.max_priority_value = 1

    def _total_priority(self):
        return self.nodes[0][0]

    def sample(self, batch_size):
        assert self.can_sample(batch_size)

        bounds = np.linspace(0., 1., batch_size + 1)
        assert len(bounds) == batch_size + 1
        segments = [(bounds[i], bounds[i+1]) for i in range(batch_size)]

        idxes = []
        priorities = [] 
        for segment in segments:
            priority = np.random.uniform(segment[0], segment[1])
            idx, actual_priority = self.search(priority * self._total_priority())

            idxes.append(idx)
            priorities.append(actual_priority)
        return self._encode_sample(idxes), np.array(priorities), np.array(idxes)

    def store_effect(self, idx, action, reward, done):
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx]   = done

        self._update_priority((self.max_priority_value), idx)

    def update_priorities(self, new_priorities, indices):
        assert(len(new_priorities) == len(indices))
        for i in range(len(new_priorities)):
            self._update_priority( (new_priorities[i]+1e-8)**self.alpha, indices[i])

    def _update_priority(self, priority, index):
        self.max_priority_value = max(self.max_priority_value, priority)

        delta = priority - self.nodes[-1][index]

        current_index = index
        for j in range(len(self.nodes)-1, -1, -1):
            nodes_at_this_depth = self.nodes[j]
            nodes_at_this_depth[current_index] += delta
            current_index = current_index // 2

    def search(self, value):
        current_index = 0

        for nodes_at_this_depth in self.nodes[1:]:
          left_child = current_index * 2
          left_sum = nodes_at_this_depth[left_child]
          if value < left_sum: 
            current_index = left_child
          else:  
            current_index = left_child + 1
            value -= left_sum
        return current_index, self.nodes[-1][current_index]
