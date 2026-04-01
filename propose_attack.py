import copy
import numpy as np
import torch
from utils import clip_image_values
from torch.autograd import Variable
from scipy.fftpack import dct, idct
import math




class Proposed_attack():
    def __init__(self, model, src_img, mean, std, lb, ub, 
                 tar_img = None, dim_reduc_factor=4, attack_method = 'CGBA_H',
                 iteration=93, initial_query=30, tol=0.0001, sigma=0.0002, 
                 verbose_control='Yes'):
        self.model = model
        self.src_img = src_img
        self.src_lbl = torch.argmax(self.model.forward(Variable(self.src_img, requires_grad=True)).data).item()
        self.tar_img = tar_img
        if tar_img != None:
            self.tar_lbl = torch.argmax(self.model.forward(Variable(self.tar_img, requires_grad=True)).data).item()
        self.dim_reduc_factor = dim_reduc_factor
        self.iteration =iteration
        self.N0 = initial_query
        self.mean = mean
        self.std = std
        self.lb = lb
        self.ub = ub
        self.tol = tol
        self.sigma = sigma
        self.grad_estimator_batch_size = 40
        self.verbose_control = verbose_control
        self.attack_method = attack_method

        # print(f'Source imge lbl: {self.src_lbl}     Targeted image lbl: {self.tar_lbl}')
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.all_queries = 0
        
        
        
    def is_adversarial(self, image):
        predict_label = torch.argmax(self.model.forward(Variable(image, requires_grad=True)).data).item()
        self.all_queries += 1
        if self.tar_img == None:
            is_adv = predict_label != self.src_lbl
        else:
            is_adv = predict_label == self.tar_lbl
        if is_adv:
            return 1
        else:
            return -1
    
    
    
    def find_random_adversarial(self, image):
        """替换原来的随机初始化，改用低频结构化方向"""
        num_calls = 0
        step = 1.0

        while True:
            # 1. 低频随机方向（4×4上采样）
            H, W = image.shape[2], image.shape[3]
            v_low = torch.randn(image.shape[0], image.shape[1], 4, 4).to(self.device)
            v_dir = F.interpolate(
                v_low, size=(H, W),
                mode='bilinear', align_corners=False
            )
            v_dir = v_dir / torch.norm(v_dir)

            # 2. 指数步长向外试探
            d_test = step
            x_test = clip_image_values(image + d_test * v_dir, self.lb, self.ub)
            num_calls += 1

            while self.is_adversarial(x_test) == -1 and d_test < 100:
                d_test *= 2.0
                x_test = clip_image_values(image + d_test * v_dir, self.lb, self.ub)
                num_calls += 1

            # 3. 找到对抗样本，二分搜索精确定位边界
            if self.is_adversarial(x_test) == 1:
                d_low  = 0.0
                d_high = d_test

                for _ in range(10):
                    d_mid = (d_low + d_high) / 2.0
                    x_mid = clip_image_values(image + d_mid * v_dir, self.lb, self.ub)
                    num_calls += 1
                    if self.is_adversarial(x_mid) == 1:
                        d_high = d_mid
                    else:
                        d_low  = d_mid

                x_init = clip_image_values(image + d_high * v_dir, self.lb, self.ub)
                return x_init, num_calls 
    
    
    
    def bin_search(self, x_0, x_random):  
        num_calls = 0
        adv = x_random
        cln = x_0      
        while True:         
            mid = (cln + adv) / 2.0
            num_calls += 1           
            if self.is_adversarial(mid)==1:
                adv = mid
            else:
                cln = mid   
            if torch.norm(adv-cln).cpu().numpy()<self.tol or num_calls>=100:
                break       
        return adv, num_calls 
    
    
    
    def normal_vector_approximation_batch(self, x_boundary, q_max, random_noises):    
        '''
        To estimate the normal vector on the boundary point, x_boundary, at each iteration
        '''
        grad_tmp = [] # estimated gradients in each estimate_batch
        z = [] # sign of grad_tmp
        outs = []
        num_batchs = math.ceil(q_max/self.grad_estimator_batch_size)
        last_batch = q_max - (num_batchs-1)*self.grad_estimator_batch_size        
        for j in range(num_batchs):            
            if j == num_batchs-1:
                current_batch = random_noises[self.grad_estimator_batch_size * j:]
                noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*last_batch + self.sigma*current_batch.cpu().numpy()   
            else:
                current_batch = random_noises[self.grad_estimator_batch_size * j:self.grad_estimator_batch_size * (j + 1)]
                noisy_boundary = [x_boundary[0,:,:,:].cpu().numpy()]*self.grad_estimator_batch_size +self.sigma*current_batch.cpu().numpy()   
            noisy_boundary_tensor = torch.tensor(noisy_boundary).to(self.device)   
            predict_labels = torch.argmax(self.model.forward(noisy_boundary_tensor),1).cpu().numpy().astype(int)              
            outs.append(predict_labels)  
        outs = np.concatenate(outs, axis=0)
        self.all_queries = self.all_queries+q_max
        for i, predict_label in enumerate(outs):
            if self.tar_img == None:
                if predict_label == self.src_lbl:
                    z.append(1)
                    grad_tmp.append(random_noises.cpu().numpy()[i])
                else:
                    z.append(-1)
                    grad_tmp.append(-random_noises.cpu().numpy()[i])  
            if self.tar_img != None:
                if predict_label != self.tar_lbl:
                    z.append(1)
                    grad_tmp.append(random_noises.cpu().numpy()[i])
                else:
                    z.append(-1)
                    grad_tmp.append(-random_noises.cpu().numpy()[i]) 
        grad = -(1/q_max)*sum(grad_tmp)       
        grad_f = torch.tensor(grad).to(self.device)[None, :,:,:]    
        return grad_f, sum(z)    



    def go_to_boundary_CGBA_H(self, x_s, eta_o, x_b):   
        num_calls = 1
        eta = eta_o/torch.norm(eta_o)
        v = (x_b - x_s)/torch.norm(x_b - x_s)
        dot_val = torch.dot(eta.reshape(-1), v.reshape(-1)).clamp(-1.0, 1.0)
        theta = torch.acos(dot_val)
        #theta = torch.acos(torch.dot(eta.reshape(-1), v.reshape(-1)))  
        while True:
            m = (torch.sin(theta)*torch.cos(theta/(pow(2, num_calls)))/torch.sin(theta/(pow(2,num_calls)))-torch.cos(theta)).item()
            zeta = (eta + m*v)/torch.norm(eta + m*v)
            perturbed = x_s + zeta*torch.norm(x_b-x_s)*torch.dot(zeta.reshape(-1), v.reshape(-1)) 
            perturbed = clip_image_values(perturbed, self.lb, self.ub)
            num_calls += 1
            if self.is_adversarial(perturbed) == 1:
                break
            if num_calls >=40:
                return x_b, num_calls
        perturbed, bin_query = self.bin_search(self.src_img, perturbed)
        return perturbed, num_calls-1 + bin_query



    def go_to_boundary_CGBA(self, x_s, eta_o, x_b):
        num_calls = 1
        eta = eta_o/torch.norm(eta_o)
        v = (x_b - x_s)/torch.norm(x_b - x_s)
        dot_val = torch.dot(eta.reshape(-1), v.reshape(-1)).clamp(-1.0, 1.0)
        theta = torch.acos(dot_val)
        #theta = torch.acos(torch.dot(eta.reshape(-1), v.reshape(-1)))
        while True:   
            m = (torch.sin(theta.cpu())*torch.cos(torch.tensor([(math.pi/2)])*(1 - 1/pow(2, num_calls)))/torch.sin(torch.tensor([(math.pi/2)])*(1 - 1/pow(2, num_calls)))-torch.cos(theta.cpu())).item()
            zeta = (eta + m*v)/torch.norm(eta + m*v)
            p_near_boundary = x_s + zeta*torch.norm(x_b-x_s)*torch.dot(v.reshape(-1), zeta.reshape(-1)) 
            p_near_boundary = clip_image_values(p_near_boundary, self.lb, self.ub)
            if self.is_adversarial(p_near_boundary) == -1:
                break
            num_calls += 1
            if num_calls>=40:
                return x_b, num_calls
        perturbed , n_calls = self.SemiCircular_boundary_search(x_s, x_b, p_near_boundary)
        return perturbed, num_calls+n_calls

    

    def SemiCircular_boundary_search(self, x_0, x_b, p_near_boundary):
        num_calls = 0
        norm_dis = torch.norm(x_b-x_0)
        boundary_dir = (x_b-x_0)/torch.norm(x_b-x_0)
        clean_dir = (p_near_boundary - x_0)/torch.norm(p_near_boundary - x_0)
        adv_dir = boundary_dir
        adv = x_b
        clean = x_0
        while True:
            mid_dir = adv_dir + clean_dir
            mid_dir = mid_dir/torch.norm(mid_dir)
            dot_val = (torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1)) /(torch.linalg.norm(boundary_dir) * torch.linalg.norm(mid_dir))).clamp(-1.0, 1.0)
            theta = torch.acos(dot_val)
            #theta = torch.acos(torch.dot(boundary_dir.reshape(-1), mid_dir.reshape(-1))/ (torch.linalg.norm(boundary_dir)*torch.linalg.norm(mid_dir)))
            d = torch.cos(theta)*norm_dis
            x_mid = x_0 + mid_dir*d
            num_calls +=1
            if self.is_adversarial(x_mid)==1:
                adv_dir = mid_dir
                adv = x_mid  
            else:
                clean_dir = mid_dir  
                clean = x_mid                             
            if torch.norm(adv-clean).cpu().numpy()<self.tol:
                break
            if num_calls >100:
                break   
        adv = clip_image_values(adv, self.lb, self.ub)   
        return adv, num_calls
    
    
    
    def find_random(self, x, n):
        image_size = x.shape
        out = torch.zeros(n, 3, int(image_size[-2]), int(image_size[-1]))
        for i in range(n):
            x = torch.zeros(image_size[0], 3, int(image_size[-2]), int(image_size[-1]))
            fill_size = int(image_size[-1]/self.dim_reduc_factor)
            x[:, :, :fill_size, :fill_size] = torch.randn(image_size[0], x.size(1), fill_size, fill_size)
            if self.dim_reduc_factor > 1.0:
                x = torch.from_numpy(idct(idct(x.cpu().numpy(), axis=3, norm='ortho'), axis=2, norm='ortho'))
            out[i] = x
        return out



    def Attack(self):
        norms = []
        n_query = []
        grad = 0   
        total_grad_queries     = 0  # ← 新增
        total_boundary_queries = 0  # ← 新增

        x_inv = self.inv_tf(copy.deepcopy(self.src_img.cpu()[0,:,:,:].squeeze()), self.mean, self.std)
        if self.tar_img == None:
            x_random, query_random= self.find_random_adversarial(self.src_img)
        if self.tar_img != None:
            x_random, query_random= self.tar_img, 0
        x_b, query_b = self.bin_search(self.src_img, x_random)
        x_b_inv = self.inv_tf(copy.deepcopy(x_b.cpu()[0,:,:,:].squeeze()), self.mean, self.std) 
        norm_initial = torch.norm(x_b_inv - x_inv)
        norms.append(norm_initial)
        q_num = query_random + query_b
        print('Initial boundary norm', torch.norm(norm_initial).item())
        print('initial query', q_num)
        n_query.append(q_num)
        size = self.src_img.shape
    
        for i in range(self.iteration):
            q_opt = int(self.N0*np.sqrt(i+1)) 
            if self.dim_reduc_factor < 1.0:
                raise Exception("The dimension reduction factor should be greater than 1 for reduced dimension, and should be 1 for Full dimensional image space.")
            if self.dim_reduc_factor > 1.0:
                random_vec_o = self.find_random(self.src_img, q_opt) 
            else:
                random_vec_o = torch.randn(q_opt,3,size[-2],size[-1]) 

            grad_oi, ratios = self.normal_vector_approximation_batch(x_b, q_opt, random_vec_o)
        
            q_num = q_num + q_opt
            total_grad_queries += q_opt  # ← 新增

            if self.attack_method == 'CGBA':
                x_adv, qs = self.go_to_boundary_CGBA(self.src_img, grad_oi, x_b)
            if self.attack_method == 'CGBA_H':
                x_adv, qs = self.go_to_boundary_CGBA_H(self.src_img, grad_oi, x_b)

            q_num = q_num + qs
            total_boundary_queries += qs  # ← 新增

            x_b = x_adv
            x_adv_inv = self.inv_tf(copy.deepcopy(x_adv.cpu()[0,:,:,:].squeeze()), self.mean, self.std)            
            norm = torch.norm(x_inv - x_adv_inv)

            if i%4==0 or i==self.iteration-1:
                if self.verbose_control == 'Yes':
                    print('iteration -> ' + str(i) + 
                         '   Queries ' + str(q_num) + 
                        ' norm is -> ' + f'{norm.item():.3f}' +
                        f'   grad_q={q_opt}  boundary_q={qs}')  # ← 新增

            norms.append(norm)
            n_query.append(q_num)
        
        # ── 最终汇总 ──────────────────────────────────────────────
        print(f'\n── Query num ──────────────────────────────────')
        print(f'Gradient estimation queries : {total_grad_queries}')
        print(f'Boundary search queries     : {total_boundary_queries}')
        print(f'Total queries               : {q_num}')
        print(f'────────────────────────────────────────────────')

        x_adv = clip_image_values(x_adv, self.lb, self.ub)           
        return x_adv, n_query, norms



    def inv_tf(self, x, mean, std):   
        '''
        To rescale the pixels of x within 0 and 1
        '''
        for i in range(len(mean)):    
            x[i] = np.multiply(x[i], std[i], dtype=np.float32)
            x[i] = np.add(x[i], mean[i], dtype=np.float32)   
        x = np.swapaxes(x, 0, 2)      
        x = np.swapaxes(x, 0, 1)    
        return x
  
    
  
    
  
    
  
    
  
