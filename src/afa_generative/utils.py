import torch
import torch.nn as nn
from torch.distributions import RelaxedOneHotCategorical
from sklearn.model_selection import train_test_split
import collections
import numpy as np
import math
import copy
import yaml


def ReadYAML(Filename):
    with open(Filename,'r') as ymlfile:
        # cfg=yaml.load(ymlfile)
        cfg = yaml.safe_load(ymlfile)
    return cfg

def restore_parameters(model, best_model):
    '''Move parameters from best model to current model.'''
    for param, best_param in zip(model.parameters(), best_model.parameters()):
        param.data = best_param

def generate_uniform_mask(batch_size, num_features):
    '''Generate binary masks with cardinality chosen uniformly at random.'''
    unif = torch.rand(batch_size, num_features)
    ref = torch.rand(batch_size, 1)
    return (unif > ref).float()

def base_generate_mask_incomplete(batch_data,mask_prop=0.25,rs=40):
    mask=np.zeros((batch_data.shape[0],batch_data.shape[1]))
    mask_other=np.zeros((batch_data.shape[0],batch_data.shape[1]))
    for row_idx in range(batch_data.shape[0]):
        pos = np.where(np.abs(batch_data[row_idx,:])>0)[0] # Positions of all observed data
        pos_train, pos_test, pos_train, pos_test = train_test_split(pos, pos, test_size=mask_prop) # select which is masked for train
        mask[row_idx, pos_train] = 1
        mask_other[row_idx, pos_test] = 1
    return mask,mask_other

def make_onehot(x):
    '''Make an approximately one-hot vector one-hot.'''
    argmax = torch.argmax(x, dim=1)
    onehot = torch.zeros(x.shape, dtype=x.dtype, device=x.device)
    onehot[torch.arange(len(x)), argmax] = 1
    return onehot

def square_dict(D):
    D_squared=collections.OrderedDict()
    for key,value in D.items():
        D_squared[key]=value**2
    return D_squared

def get_mask(deleted_column_batch_data,flag_tensor=True):
    if flag_tensor==True:
        return (torch.abs(deleted_column_batch_data)>0.).float()
    else:
        return (np.abs(deleted_column_batch_data)>0.).astype(float)

def grap_modify_grad(list_p,W_dict_size,Data_N):
    grad_list=[]
    for p in list_p:
        if p.grad is not None:
            grad_list.append(-p.grad.clone().detach()/(W_dict_size*Data_N))
        else:
            grad_list.append('None')
    return grad_list

def zero_grad(list_p):
    for p in list_p:
        if p.grad is not None:
            p.grad.data.zero_()

def assign_grad(list_p,list_grad):
    idx=0
    for p in list_p:
        if type(list_grad[idx])!=type('None'):
            p.grad.data=list_grad[idx].data.clone()
        else:
            p.grad.data=None
        idx+=1

def Update_W_sample(W_dict,W_sample,sample_num,maxsize=20):
    if len(W_sample)>=maxsize:
        key,value=W_sample.popitem(last=False)
        #print('%s is evicted'%(key))
        W_sample['sample_%s'%(sample_num)]=copy.deepcopy(W_dict)
    else:
        W_sample['sample_%s' % (sample_num)] = copy.deepcopy(W_dict)
    return W_sample

# Use our evaluation script instead?
def Test_UCI_batch(PNP,test_input,test_target,sigma_out_scale=0.1,split=3,flag_model='PNP_BNN',size=10,Infer_model=None,W_sample=None):
    batch_size = int(math.ceil(test_input.shape[0] / split))
    pre_idx = 0
    PNP_copy = PNP#copy.deepcopy(PNP)
    total_se = 0
    total_MAE = 0
    total_NLL =0
    for counter in range(split+1):
        idx=min((counter+1)*batch_size,test_input.shape[0])
        if pre_idx==idx:
            break
        data_input=test_input[pre_idx:idx,:]
        data_target=test_target[pre_idx:idx,:]
        # get mask
        mask_test_input = get_mask(data_input)
        mask_test_target = get_mask(data_target)
        if flag_model=='PNP_BNN':
            pred_mean,_=PNP_copy.completion(data_input,mask_test_input,sigma_out_scale,size=size)
            pred_mean=pred_mean*mask_test_target
            _,pred_tot_ll=PNP_copy.test_log_likelihood(data_input,data_target,mask_test_input,sigma_out_scale,size=size)
        elif flag_model=='PNP':
            sigma_out = sigma_out_scale * torch.ones(1, test_input.shape[1])
            pred_mean = PNP_copy.completion(data_input, mask_test_input, sigma_out) * mask_test_target
            _,pred_tot_ll=PNP_copy.test_log_likelihood(data_input,data_target,mask_test_input,sigma_out)
        elif flag_model=='PNP_SGHMC':
            if Infer_model.flag_LV:
                pred_mean,_ = PNP_copy.completion(Infer_model,data_input, mask_test_input,W_sample, size_Z=size,record_memory=False)

            else:
                pred_mean = PNP_copy.completion(Infer_model,data_input, mask_test_input,W_sample, size_Z=size,record_memory=False)
            pred_mean=torch.mean(torch.mean(pred_mean,dim=0),dim=0)
            pred_mean = pred_mean * mask_test_target


            _, pred_tot_ll = PNP_copy.test_log_likelihood(Infer_model,X_in=data_input, X_test=data_target,W_sample=W_sample, mask=mask_test_input,sigma_out=sigma_out_scale,
                                                          size=size)
        else:
            raise NotImplementedError

        # pred_test = torch.tensor(pred_mean.data)
        pred_test = pred_mean.clone().detach()
        ae = torch.sum(torch.abs(pred_test - data_target))
        # ae = torch.tensor(ae.data)
        ae = ae.clone().detach()
        se = torch.sum((pred_test - data_target) ** 2)
        # se = torch.tensor(se.data)
        se = se.clone().detach()
        total_se += se
        total_MAE += ae
        total_NLL+=-pred_tot_ll
        pre_idx = idx
    total_mask_target = get_mask(test_target)
    RMSE = torch.sqrt(1. / (torch.sum(total_mask_target)) * total_se)
    # RMSE = torch.tensor(RMSE.data)
    RMSE = RMSE.clone().detach()
    # MAE = torch.tensor((1. / torch.sum(total_mask_target) * total_MAE).data)
    MAE = (1. / torch.sum(total_mask_target) * total_MAE).clone().detach()
    # NLL=torch.tensor(1. / (torch.sum(total_mask_target)) *total_NLL)
    NLL = (1. / (torch.sum(total_mask_target)) *total_NLL).clone().detach()
    return RMSE, MAE, NLL

def test_UCI_AL(model,max_selection,sample_x,test_input,test_pool,test_target,sigma_out,search='Target',model_name='PNP_BNN',**kwargs):
    RMSE_Results=np.zeros(max_selection+1)
    MAE_Results=np.zeros(max_selection+1)
    NLL_Results=np.zeros(max_selection+1)
    if model_name=='PNP_SGHMC':
        W_sample=kwargs['W_sample']
        Infer_model=model.Infer_model
        # Evaluate on zero selection
        RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                   sigma_out_scale=sigma_out,
                                                   split=10, flag_model=model_name, size=25,Infer_model=Infer_model,W_sample=W_sample)
    else:
        RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                       sigma_out_scale=sigma_out,
                                                       split=10, flag_model=model_name, size=25)

    RMSE_Results[0] = RMSE_test.cpu().data.numpy()
    MAE_Results[0] = MAE_test.cpu().data.numpy()
    NLL_Results[0] = NLL_test.cpu().data.numpy()
    for num_selected in range(max_selection):
        # Active Learning
        if search=='Target' and num_selected>-1:
            if model_name=='PNP_BNN':
                test_input, index_array, test_pool = model.base_active_learning_z_target_BNN(
                    active_sample_number=sample_x,
                    test_input=test_input,
                    pool_data_tensor=test_pool, target_data_tensor=test_target)
            elif model_name=='PNP_SGHMC':
                test_input, index_array, test_pool = model.active_learn_target_test(flag_same_pool=True,test_input=test_input,pool_data_tensor=test_pool,target_data_tensor=test_target,size_z=10,split=10,W_sample=W_sample)
            else:
                raise NotImplementedError
        elif search=='Random' :
            if model_name=='PNP_BNN':
                test_input, index_array, test_pool = model.base_random_select(
                    active_sample_number=sample_x, test_input=test_input, pool_data_tensor=test_pool)
            elif model_name=='PNP_SGHMC':
                test_input, index_array, test_pool = model.random_select_test( test_input=test_input,pool_data_tensor=test_pool)
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        # Clear memory
        test_input=torch.tensor(test_input.data)
        test_pool=torch.tensor(test_pool.data)
        if model_name=='PNP_SGHMC':
            RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                                          sigma_out_scale=sigma_out,
                                                                          split=10, flag_model=model_name, size=25,Infer_model=Infer_model,W_sample=W_sample)
        else:
            RMSE_test, MAE_test, NLL_test = Test_UCI_batch(model.model, test_input, test_target,
                                                           sigma_out_scale=sigma_out,
                                                           split=10, flag_model=model_name, size=25)
        RMSE_Results[num_selected+1]=RMSE_test.cpu().data.numpy()
        MAE_Results[num_selected+1]=MAE_test.cpu().data.numpy()
        NLL_Results[num_selected+1]=NLL_test.cpu().data.numpy()
    return RMSE_Results,MAE_Results,NLL_Results

def Compute_AUIC_1D(const=None,**kwargs):
    if type(const)==type(None):
        BALD = kwargs['BALD']
        RAND = kwargs['RAND']
        diff=RAND-BALD
        diff_roll=np.roll(diff,-1)
        area = np.sum((0.5 * (diff + diff_roll))[0:-1], axis=0)
    else:
        Results=kwargs['Results']
        #Results=Results.cpu().data.numpy()
        diff=Results-const
        diff_roll=np.roll(diff,-1)
        area = np.sum((0.5 * (diff + diff_roll))[0:-1], axis=0)
    return area

def get_choice(observed_train_data):
    num_selected=torch.sum(torch.abs(observed_train_data)>0.,dim=0) # obs_dim
    return num_selected.cpu().data.numpy()

def reduce_size(samples,perm1,perm2,flag_one_row=False):
    # Note the samples have the shape N_w x N_z x N x obs or N_z x N x obs or N_w x Nz x Np x N x obs
    if flag_one_row==True:
        if len(samples.shape)==3:
            sample_reduce_1=samples[perm1,:,:]
            sample_reduce_2=sample_reduce_1[:,perm2,:]
        else:
            raise NotImplementedError
    else:
        if len(samples.shape)==4:
            sample_reduce_1=samples[perm1,:,:,:] # reduced x n_z x N x obs
            sample_reduce_2=sample_reduce_1[:,perm2,:,:]
        elif len(samples.shape)==3:
            sample_reduce_2=samples[perm2,:,:]
        elif len(samples.shape) == 5:

            sample_reduce_1 = samples[perm1,:, :, :, :]  # reduced x n_z x Np x N x obs
            sample_reduce_2 = sample_reduce_1[:, perm2, :,:, :] # reduce x reduce x Np x N x obs

    return sample_reduce_2

def remove_zero_row_2D(data):
    sum_data=torch.sum(torch.abs(data),dim=1) # N
    return torch.tensor(data[sum_data>0,:].data)

def assign_zero_row_2D_with_target_reverse(data,data_target,value=0.):
    # This is the reverse operation s.t. the observed user will be assigned to 0.
    data=torch.tensor(data.data)
    sum_data=torch.sum(torch.abs(data_target),dim=1)
    data[sum_data>=0.001,:]=value
    return data # Can be seen as the entries of data matrix with the positions of the unobserved rows of the data_target are 0

def BALD_Select_Explore(BALD_weight,step):
    # This implement the selection basded on the probability defined by BALD value
    active_entry_num=torch.sum(BALD_weight>0.)

    if active_entry_num==0.: # Every thing is zero, means no way to draw it
        flag_full = True
        idx = []
        select_num=0
    elif active_entry_num<step:
        flag_full=False
        idx=torch.multinomial(BALD_weight,active_entry_num)
        select_num=active_entry_num
    else:
        flag_full = False
        idx = torch.multinomial(BALD_weight, step)
        select_num=step
    return idx,flag_full,select_num

def assign_zero_row_2D_with_target(data,data_target,value=0.):
    data=torch.tensor(data.data)
    sum_data=torch.sum(torch.abs(data_target),dim=1)
    data[sum_data<=0.001,:]=value
    return data # Can be seen as the entries of data matrix with the positions of the unobserved rows of the data_target are 0

def BALD_Select(BALD):
    BALD=torch.tensor(BALD.data)
    num_unobserved=torch.sum(BALD<-10)

    # Active Select
    _, idx = torch.sort(BALD.view(1, -1), descending=True)  # Flattened idx # 1 x tot
    return idx,num_unobserved


class MaskLayer(nn.Module):
    '''
    Mask layer for tabular data.
    
    Args:
      append:
      mask_size:
    '''
    def __init__(self, append, mask_size=None):
        super().__init__()
        self.append = append
        self.mask_size = mask_size

    def forward(self, x, m):
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
    

MaskLayer1d = MaskLayer
    
    
class MaskLayerGrouped(nn.Module):
    '''
    Mask layer for tabular data with feature grouping.
    
    Args:
      group_matrix:
      append:
    '''
    def __init__(self, group_matrix, append):
        # Verify group matrix.
        assert torch.all(group_matrix.sum(dim=0) == 1)
        assert torch.all((group_matrix == 0) | (group_matrix == 1))
        
        # Initialize.
        super().__init__()
        self.register_buffer('group_matrix', group_matrix.float())
        self.append = append
        self.mask_size = len(group_matrix)
        
    def forward(self, x, m):
        out = x * (m @ self.group_matrix)
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out
    
    
MaskLayer1dGrouped = MaskLayerGrouped


class MaskLayer2d(nn.Module):
    '''
    Mask layer for 2d image data.
    
    Args:
      append:
      mask_width:
      patch_size:
    '''

    # TODO change argument order, including in CIFAR notebooks
    def __init__(self, append, mask_width, patch_size):
        super().__init__()
        self.append = append
        self.mask_width = mask_width
        self.mask_size = mask_width ** 2
        
        # Set up upsampling.
        self.patch_size = patch_size
        if patch_size == 1:
            self.upsample = nn.Identity()
        elif patch_size > 1:
            self.upsample = nn.Upsample(scale_factor=patch_size)
        else:
            raise ValueError('patch_size should be int >= 1')

    def forward(self, x, m):
        # Reshape if necessary.
        if len(m.shape) == 2:
            m = m.reshape(-1, 1, self.mask_width, self.mask_width)
        elif len(m.shape) != 4:
            raise ValueError(f'cannot determine how to reshape mask with shape = {m.shape}')
        
        # Apply mask.
        m = self.upsample(m)
        out = x * m
        if self.append:
            out = torch.cat([out, m], dim=1)
        return out


class StaticMaskLayer1d(torch.nn.Module):
    '''
    Mask a fixed set of indices from 1d tabular data.
    
    Args:
      inds: array or tensor of indices to select.
    '''
    def __init__(self, inds):
        super().__init__()
        self.inds = inds
        
    def forward(self, x):
        return x[:, self.inds]


class StaticMaskLayer2d(torch.nn.Module):
    '''
    Mask a fixed set of pixels from 2d image data.
    
    Args:
      mask: mask indicating which parts of the image to remove at a patch level.
      patch_size: size of patches in the mask.
    '''

    def __init__(self, mask, patch_size):
        super().__init__()
        self.patch_size = patch_size
        mask = mask.float()

        # Reshape if necessary.
        if len(mask.shape) == 4:
            assert mask.shape[0] == 1
            assert mask.shape[1] == 1
        elif len(mask.shape) == 3:
            assert mask.shape[0] == 1
            mask = torch.unsqueeze(mask, 0)
        elif len(mask.shape) == 2:
            mask = torch.unsqueeze(torch.unsqueeze(mask, 0), 0)
        else:
            raise ValueError(f'unable to reshape mask with size {mask.shape}')
        assert mask.shape[-1] == mask.shape[-2]

        # Upsample mask.
        if patch_size == 1:
            mask = mask
        elif patch_size > 1:
            mask = torch.nn.Upsample(scale_factor=patch_size)(mask)
        else:
            raise ValueError('patch_size should be int >= 1')
        self.register_buffer('mask', mask)
        self.mask_size = self.mask.shape[2] * self.mask.shape[3]

    def forward(self, x):
        out = x * self.mask
        return out


class Flatten(object):
    '''Flatten image input.'''
    def __call__(self, pic):
        return torch.flatten(pic)


class ConcreteSelector(nn.Module):
    '''Output layer for selector models.'''

    def __init__(self, gamma=0.2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, temp, deterministic=False):
        if deterministic:
            # TODO this is somewhat untested, but seems like best way to preserve argmax
            return torch.softmax(logits / (self.gamma * temp), dim=-1)
        else:
            dist = RelaxedOneHotCategorical(temp, logits=logits / self.gamma)
            return dist.rsample()
