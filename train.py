import hydra
import torch
import os
import wandb
import gym
from tqdm import tqdm
import numpy as np
from video import VideoRecoder
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import utils

def make_agent(obs_spec,action_spec,cfg):
    cfg.state_dim=obs_spec.shape[0]
    cfg.action_dim=action_spec.n
    return hydra.utils.instantiate(cfg)

class WorkSpace(object):
    def __init__(self,cfg):
        self.work_dir=Path.cwd()
        print(f"workspace : ",{self.work_dir})
        self.video_dir=self.work_dir/ "videos"
        self.video_dir.mkdir(exist_ok=True)
        self.video=VideoRecoder(self.video_dir if cfg.save_video else None)
        os.environ["device"]=cfg.device
        self.cfg=cfg
        self.device=torch.device(cfg.device)
        utils.set_seed_everywhere(cfg.seed)
        self.setup()
        self.train_env=gym.make(cfg.task)
        self.train_env.seed(cfg.seed)
        
        self.eval_env=gym.make(cfg.task)
        self.eval_env.seed(cfg.seed+100)

        self.agent=make_agent(self.train_env.observation_space,self.train_env.action_space,self.cfg.algo)

        self._global_step = 0
    

    def setup(self):
        if self.cfg.use_wandb:
            exp_name="_".join([self.cfg.experiment,str(self.cfg.seed)])
            wandb.init(project="Policy gradient",group="series of PPO",name=exp_name,config=dict(self.cfg),save_code=True)

    def eval(self,video,env,step,eval_mode):
        episode_rewards=[]
        print(f"\n Starting evaluation for {self.cfg.eval_episodes} episodes ...")
        for i in range(self.cfg.eval_episodes):
            obs=env.reset()
            video.init(enabled=(i==0))
            done=False
            episode_reward = 0
            episode_step =0
            while not done:
                with utils.eval_mode(self.agent):
                    action=self.agent.take_action(obs)
                obs,reward,done,_=env.step(action)
                video.record(env)
                episode_reward+=reward
                episode_step+=1
            _test_env=f"_{eval_mode}"
            video.save(f"{step}{_test_env}.mp4")
            episode_rewards.append(episode_reward)
        mean_reward=np.mean(episode_rewards)
        print("mean_reward : ",mean_reward)
        return mean_reward

            
    @property
    def global_step(self):
        return self._global_step


        
    def train(self):
        return_list=[]
        for i in range(10):
            with tqdm(total=int(self.cfg.train_episodes/10),desc="Iteration %d "% i) as pbar:
                for i_episode in range(int(self.cfg.train_episodes/10)):
                    episode_return=0
                    transition_dict={"states":[],"actions":[],"next_states":[],"rewards":[],"dones":[]}
                    state=self.train_env.reset()
                    done=False
                    while not done:
                        action=self.agent.take_action(state)
                        next_state,reward,done,_=self.train_env.step(action)
                        transition_dict["states"].append(state)
                        transition_dict["actions"].append(action)
                        transition_dict["next_states"].append(next_state)
                        transition_dict["rewards"].append(reward)
                        transition_dict["dones"].append(done)
                        state=next_state
                        episode_return+=reward
                        self._global_step += 1
                    return_list.append(episode_return)
                    return_log_dict={"return":episode_return}
                    wandb.log(return_log_dict,step=i*int(self.cfg.train_episodes/10)+i_episode)
                    self.agent.update(transition_dict)
                    if (i_episode+1)%10==0:
                        pbar.set_postfix({"episode": "%d" %(self.cfg.train_episodes/10*i+i_episode+1),"return": "%.3f"% np.mean(return_list[-10:])})
                    pbar.update(1)
        if self.cfg.save_snapshot: #and (self.global_step % int(1e5) == 0):
            self.eval(self.video,self.eval_env,self.global_step,eval_mode="train")
            self.save_snapshot()
        
        wandb.finish()
        return return_list


    def save_snapshot(self):
        snapshot=self.work_dir / "snapshot.pt"
        keys_to_save=["agent","_global_step"]
        payload={k:self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload,f)
    
    def load_snapshot(self):
        snapshot=self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload=torch.load(f)
        
        for k,v in payload.items():
            self.__dict__[k]=v


@hydra.main(config_path="cfgs",config_name="config")
def main(cfg):
    from train import WorkSpace as W
    root_dir=Path.cwd()
    workspace=W(cfg)
    snapshot=root_dir/"snapshot.pt"
    if snapshot.exists():
        print(f"resuming : {snapshot}")
        workspace.load_snapshot()
    workspace.train()



if __name__=="__main__":
    main()