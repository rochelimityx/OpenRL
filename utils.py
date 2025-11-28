import torch
import numpy as np
import random



def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)





def compute_advantage(gamma,lmbda,td_delta):
    td_delta=td_delta.detach().numpy()
    advantage_list=[]
    advantage=0.0
    for delta in td_delta[::-1]:#注意从后往前计算GAE
        advantage=gamma*lmbda*advantage+delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list,dtype=torch.float)

class eval_mode(object):
    def __init__(self,*models):#构造函数接受任意数量的位置参数
        self.models=models

    def __enter__(self):#进入with块时调用
        self.prev_states=[]
        for model in self.models:
            self.prev_states.append(model.training)#记录当前模型的 training 属性（True 表示 training 模式，False 表示 eval 模式）。这是读取 nn.Module.training 属性。
            model.train(False)
    
    def __exit__(self,*args):#离开with块的时候调用
        for model,state in zip(self.models,self.prev_states):#把每个模型的 training 标志恢复为进入上下文前保存的状态（True 或 False）。
            model.train(state)
        return False