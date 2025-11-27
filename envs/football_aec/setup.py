from agents.base_agent import FootballAgent
from core.ball import Ball
from core.field import Field
from core.recorder import Recorder

def create_players(agent_names):
    players = []
    for name in agent_names:
        team = "left" if int(name.split("_")[1]) < 11 else "right"
        # buat instance FootballAgent; default role = 'midfielder'
        agent = FootballAgent(team=team, role="midfielder")
        agent.name = name
        players.append(agent)
    return players

def create_ball():
    return Ball()

def create_field():
    return Field()

def create_recorder():
    return Recorder()