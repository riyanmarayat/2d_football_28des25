import numpy as np
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from gymnasium import spaces

from core.simulator import Simulator
from .setup import create_players, create_ball, create_field, create_recorder

class FootballEnv(AECEnv):
    metadata = {'render_modes': ['human'], 'name': 'football_arc_v0'}

    def __init__(self):
        super().__init__()
        # define possible agents
        self._agent_selector = None
        self.possible_agents = [f'player_{i}' for i in range(22)]
        self.agents = []
        self.agent_name_mapping = {agent: i for i, agent in enumerate(self.possible_agents)}

        self.observation_spaces = {
            agent: spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32) for agent in self.possible_agents # Example observation space with 20 features
        }
        self.action_spaces = {
            agent: spaces.Discrete(6) for agent in self.possible_agents # 6 possible actions (e.g., pass, shoot, dribble, etc.)
        }
        self.sim = None

    def reset(self, seed=None, options=None):
        self.agents = self.possible_agents[:]
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        players = create_players(self.agents)
        ball = create_ball()
        field = create_field()
        recorder = create_recorder()

        #reset internal simulator
        self.sim = Simulator(players, ball, field, recorder) #edit this line to instantiate your simulator
        self.sim.reset()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agents: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.observations = {agent: self.sim.get_observation_by_name (agent) for agent in self.agents}

        return self.observations[self.agent_selection], {}

    def step(self, action):
        agent = self.agent_selection

        #apply action to the simulator
        self.sim.apply_action_by_name(agent, action)
        self.sim.step()

        self.rewards[agent] = self.sim.get_reward(agent)
        self.terminations[agent] = self.sim.check_terminated()
        self.truncations[agent] = False
        self.observations[agent] = self.sim.get_observation_by_name(agent)

        #advance to the next agent
        self.agent_selection = self._agent_selector.next()
        return (self.observations[self.agent_selection],
                self.rewards[agent],
                self.terminations[agent],
                self.truncations[agent],
                self.infos[agent])

    def observe(self, agent):
        return self.observations[agent]

    def render(self):
        if self.sim:
            self.sim.render()

    def close(self):
        if self.sim:
            self.sim.close()