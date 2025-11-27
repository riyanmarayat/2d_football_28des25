import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import os
from tqdm import tqdm

experiment = [
    ['elu',        "./Astroids_ELU.jit.pth"],
]

MAX_STEPS = 10_000
RUN_COUNT =  5_000

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CSVManager:
    def __init__(self, filename: str):
        self.filename = filename
    
    def create_template(self) -> None:
        if os.path.exists(self.filename) == False:
            print("File Does Not Exist !")
            x = pd.DataFrame({'placeholder' : []})
            x.to_csv(self.filename, index = False)

    def read_csv(self) -> pd.DataFrame:
        self.create_template()
        return pd.read_csv(self.filename)

    def write_csv(self, data: pd.DataFrame):
        data.to_csv(self.filename, index=False)

    def update(self, experiment_name : str, values : list[float]):
        data = self.read_csv()
        data.insert(0, experiment_name, values)
        self.write_csv(data)

def model_act(model : torch.jit.ScriptModule, input : np.ndarray) -> int:
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    with torch.no_grad():
        action_values : torch.Tensor = model(state)
    selected = np.argmax(action_values.to('cpu').data.numpy())
    return int(selected)

def run_evn(env : gym.Env, model : torch.jit.ScriptModule) -> float:
    observation, info = env.reset()
    action_rewards = 0
    for _ in range(MAX_STEPS):
        action = model_act(model, observation)
        observation, reward, terminated, truncated, info = env.step(action)
        action_rewards += float(reward)
        
        if terminated or truncated:
            break    
    
    return float(action_rewards)

def calculate_sturges_bins(data : list):
    n = len(data)
    return int(np.ceil(1 + np.log2(n)))


if __name__ == "__main__":
    print("Running Experiments")  
    
    data_store = CSVManager("./Combined_Results/results_combined.csv")
    env = gym.make('ALE/Asteroids-ram-v5')
    
    for experiement_name, file_path in experiment:
        print("Running :", experiement_name)
        print("> File  :", file_path)

        experiment_scores = []
        model : torch.jit.ScriptModule = torch.jit.load(file_path, map_location = device)
        
        for _ in tqdm(RUN_COUNT):
            # run the experiments
            score = run_evn(env, model)
            experiment_scores.append(score)
        
        data_store.update(experiement_name, experiment_scores)

        bines = calculate_sturges_bins(experiment_scores)
        plt.hist(experiment_scores, bins = bines, edgecolor='black')
        plt.title(f'Histogram of Scores For "{experiement_name}"')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.savefig(f"./Combined_Results/histogram_{experiement_name}.jpg")
        plt.clf()
    
    print("Performing Fully Random Run !")
    import random
    experiement_name  = "random"
    experiment_scores = []
    for _ in tqdm(range(RUN_COUNT)):
        # run the experiments
        observation, info = env.reset()
        action_rewards = 0
        for _ in range(MAX_STEPS):
            action = random.randrange(0, (14 + 1)) #select random 0 - 14
            observation, reward, terminated, truncated, info = env.step(action)
            action_rewards += float(reward)
            if terminated or truncated:
                break    
        experiment_scores.append(action_rewards)
    
    data_store.update(experiement_name, experiment_scores)
    bines = calculate_sturges_bins(experiment_scores)
    plt.hist(experiment_scores, bins = bines, edgecolor='black')
    plt.title(f'Histogram of Scores For "{experiement_name}"')
    plt.xlabel('Score')
    plt.ylabel('Frequency')
    plt.savefig(f"./Combined_Results/histogram_{experiement_name}.jpg")

    plt.clf()

    print("Complete !")