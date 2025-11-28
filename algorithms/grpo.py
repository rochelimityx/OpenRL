import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import compute_advantage
import numpy as np





class Actor(nn.Module):
    def __init__(self,state_dim,hidden_dim,action_dim):
        super(Actor,self).__init__()
        self.fc1=nn.Linear(state_dim,hidden_dim)
        self.fc_mu=nn.Linear(hidden_dim,action_dim)
        self.fc_std=nn.Linear(hidden_dim,action_dim)

    
    def forward(self,x):
        x=F.relu(self.fc1(x))
        mu=2.0*torch.tanh(self.fc_mu(x)) #将输出限制在（-1,1）;再乘以2让范围变成(-2,2)
        std=F.softplus(self.fc_std(x)) #log(1+exp(x));保证输出是正数；标准差必须为正数
        return mu,std








class GRPO(object):
    def __init__(self,state_dim,hidden_dim,action_dim,actor_lr,lmbda,epochs,gamma,eps,device):
        self.actor=Actor(state_dim,hidden_dim,action_dim).to(device)
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters(),lr=actor_lr)
        

        self.gamma=gamma
        self.lmbda=lmbda
        self.epochs=epochs
        self.eps=eps
        self.device=device
    
    def take_action(self,state):
        state = torch.tensor([state],dtype=torch.float).to(self.device)
        mu,sigma=self.actor(state)
        action_dist=torch.distributions.Normal(mu,sigma)
        action = action_dist.sample()
        return [action.item()]#将单个元素的张量转化为Python原生数据类型
    
    def group_normalize(self,advantages,group_size=32):
        n=len(advantages)
        groups=[advantages[i:i+group_size] for i in range(0,n,group_size)]
        norm_adv=[]
        for g in groups:
            g=(g-torch.mean(g))/(torch.std(g)+1e-8)
            norm_adv.extend(g) # 添加g中的元素
        return torch.tensor(norm_adv,dtype=torch.float).to(self.device)

    def update(self,transition_dict):
        states=torch.tensor(transition_dict["states"],dtype=torch.float).to(self.device)
        actions=torch.tensor(transition_dict["actions"],dtype=torch.float).view(-1,1).to(self.device)
        rewards=torch.tensor(transition_dict["rewards"],dtype=torch.float).view(-1,1).to(self.device)#一定要注意维度的对齐，否则容易触发广播机制
        rewards=(rewards+8.0)/8.0#和TRPO一样，对奖励进行修改，方便训练
    
        returns=[]
        G=0
        for r in reversed(rewards):
            G=r+self.gamma*G
            returns.insert(0,G)
        returns=torch.tensor(returns)
        advantages=returns-torch.mean(returns)
        advantages=self.group_normalize(advantages,group_size=16)

    

        mu,std=self.actor(states)
        action_dists=torch.distributions.Normal(mu.detach(),std.detach())
        old_log_probs=action_dists.log_prob(actions)#计算已有动作在该分布下的对数概率

        for _ in range(self.epochs):
            mu,std=self.actor(states)
            action_dists=torch.distributions.Normal(mu,std)
            log_probs=action_dists.log_prob(actions)
            ratio=torch.exp(log_probs-old_log_probs)
            surr1=ratio*advantages
            surr2=torch.clamp(ratio,1-self.eps,1+self.eps)*advantages
            actor_loss=torch.mean(-torch.min(surr1,surr2))
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()




