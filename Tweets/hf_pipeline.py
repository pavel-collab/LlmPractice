import hydra
import wandb

from omegaconf import DictConfig

@hydra.main(config_path="./conf", config_name="config", version_base="1.2 ")
def main(cfg: DictConfig):
    ...