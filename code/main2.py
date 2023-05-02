from stable_baselines3 import PPO
from luxai2021.env.agent import AgentFromStdInOut
from luxai2021.env.lux_env import LuxEnvironment
from luxai2021.game.constants import LuxMatchConfigs_Default
from PPOBot import PPOBot

if __name__ == "__main__":

    configs = LuxMatchConfigs_Default
    model = PPO.load(f"other/model_final.zip")
    opponent = AgentFromStdInOut()
    player = PPOBot(mode="inference", model=model)

    # Run the environment
    env = LuxEnvironment(configs, player, opponent)
    env.reset()
