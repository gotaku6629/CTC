"""
default
    Adaptive Asynchronous Temporal Fields Base model
but, now change into
    Adaptive Synchronous Temoiral Fields Base model 
"""
import torch.nn as nn
import torch
from torch.autograd import Variable

class BasicModule(nn.Module):
    def __init__(self, inDim, outDim, hidden_dim = 1000, dp_rate = 0.3):
        super(BasicModule, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(inDim, hidden_dim), # 1024, 1000
            nn.ReLU(),
            nn.Dropout(p = dp_rate),
            nn.Linear(hidden_dim, outDim) # 1000, 16x5
        )
    def forward(self, x):
        return self.layers(x)
        

class AsyncTFBase(nn.Module):
    def __init__(self, dim, s_classes, o_classes, v_classes, temporal, batch_size, _BaseModule = BasicModule):
        super(AsyncTFBase, self).__init__()
        self.s_classes = s_classes
        self.o_classes = o_classes
        self.v_classes = v_classes
        
        self.num_low_rank = 5
        self.temporal = temporal
        self.batch_size = batch_size

        self.s = nn.Sequential(
            nn.Linear(dim, 1000),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1000, 1000),
            nn.ReLU(),
            nn.Dropout(p = 0.3),
            nn.Linear(1000, self.s_classes)
            )
        self.o = nn.Linear(dim, self.o_classes)
        self.v = nn.Linear(dim, self.v_classes)
        
        ### label compatibility matrix
        ## spatio
        self.so_a = _BaseModule(dim, self.s_classes * self.num_low_rank)
        self.so_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
               
        self.ov_a = _BaseModule(dim, self.o_classes * self.num_low_rank) 
        self.ov_b = _BaseModule(dim, self.num_low_rank * self.v_classes) 
        
        self.vs_a = _BaseModule(dim, self.v_classes * self.num_low_rank) 
        self.vs_b = _BaseModule(dim, self.num_low_rank * self.s_classes) 
        
        ## temporal
        self.ss_a = _BaseModule(dim, self.s_classes * self.num_low_rank) 
        self.ss_b = _BaseModule(dim, self.num_low_rank * self.s_classes) 
        
        self.oo_a = _BaseModule(dim, self.o_classes * self.num_low_rank)
        self.oo_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        
        self.vv_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vv_b = _BaseModule(dim, self.num_low_rank * self.v_classes) 
        
        self.so_t_a = _BaseModule(dim, self.s_classes * self.num_low_rank) 
        self.so_t_b = _BaseModule(dim, self.num_low_rank * self.o_classes) 
        
        self.ov_t_a = _BaseModule(dim, self.o_classes * self.num_low_rank)
        self.ov_t_b = _BaseModule(dim, self.num_low_rank * self.v_classes)
        
        self.vs_t_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vs_t_b = _BaseModule(dim, self.num_low_rank * self.s_classes) 
        
        self.os_t_a = _BaseModule(dim, self.o_classes * self.num_low_rank) 
        self.os_t_b = _BaseModule(dim, self.num_low_rank * self.s_classes) 
        
        self.vo_t_a = _BaseModule(dim, self.v_classes * self.num_low_rank)
        self.vo_t_b = _BaseModule(dim, self.num_low_rank * self.o_classes)
        
        self.sv_t_a = _BaseModule(dim, self.s_classes * self.num_low_rank)
        self.sv_t_b = _BaseModule(dim, self.num_low_rank * self.v_classes) 



    def forward(self, rgb_feat):
        # rgb_feat = [10, 2, 1024]  #[time, batch, feature]

        # unary(Node)
        s = torch.cat([self.s(rgb_feat[i]).unsqueeze(0) for i in range(rgb_feat.shape[0])], dim=0)
        o = torch.cat([self.o(rgb_feat[i]).unsqueeze(0) for i in range(rgb_feat.shape[0])], dim=0)
        v = torch.cat([self.v(rgb_feat[i]).unsqueeze(0) for i in range(rgb_feat.shape[0])], dim=0)        
        
        feat = rgb_feat
        # pairwise(Edge)
        # spatio
        so_a = torch.cat([self.so_a(feat[i]).view(-1, self.s_classes, self.num_low_rank).unsqueeze(0) for i in range(feat.shape[0])], dim=0) # [3, 25, 16]
        so_b = torch.cat([self.so_b(feat[i]).view(-1, self.num_low_rank, self.o_classes).unsqueeze(0) for i in range(feat.shape[0])], dim=0) # [3, 25, 38]
        so = torch.cat([torch.bmm(so_a[i], so_b[i]).unsqueeze(0) for i in range(feat.shape[0])], dim=0)
        
        ov_a = torch.cat([self.ov_a(feat[i]).view(-1, self.o_classes, self.num_low_rank).unsqueeze(0) for i in range(feat.shape[0])], dim=0)
        ov_b = torch.cat([self.ov_b(feat[i]).view(-1, self.num_low_rank, self.v_classes).unsqueeze(0) for i in range(feat.shape[0])], dim=0)
        ov = torch.cat([torch.bmm(ov_a[i], ov_b[i]).unsqueeze(0) for i in range(feat.shape[0])], dim=0)
        
        vs_a = torch.cat([self.vs_a(feat[i]).view(-1, self.v_classes, self.num_low_rank).unsqueeze(0) for i in range(feat.shape[0])], dim=0)
        vs_b = torch.cat([self.vs_b(feat[i]).view(-1, self.num_low_rank, self.s_classes).unsqueeze(0) for i in range(feat.shape[0])], dim=0)
        vs = torch.cat([torch.bmm(vs_a[i], vs_b[i]).unsqueeze(0) for i in range(feat.shape[0])], dim=0)
        
        # temporal
        '''
            0時刻目はナシ, 1,2,...時刻目から次の時刻のノードを指すように!! 
        '''
        # t=0
        ss_a = self.ss_a(feat[0]).view(-1, self.s_classes, self.num_low_rank) # [25, 16]
        ss_b = self.ss_b(feat[0]).view(-1, self.num_low_rank, self.s_classes)
        ss_0 = torch.bmm(ss_a, ss_b).unsqueeze(0)
        
        oo_a = self.oo_a(feat[0]).view(-1, self.o_classes, self.num_low_rank)
        oo_b = self.oo_b(feat[0]).view(-1, self.num_low_rank, self.o_classes)
        oo_0 = torch.bmm(oo_a, oo_b).unsqueeze(0)
        
        vv_a = self.vv_a(feat[0]).view(-1, self.v_classes, self.num_low_rank)
        vv_b = self.vv_b(feat[0]).view(-1, self.num_low_rank, self.v_classes)
        vv_0 = torch.bmm(vv_a, vv_b).unsqueeze(0)

        so_t_a = self.so_t_a(feat[0]).view(-1, self.s_classes, self.num_low_rank)
        so_t_b = self.so_t_b(feat[0]).view(-1, self.num_low_rank, self.o_classes)
        so_t_0 = torch.bmm(so_t_a, so_t_b).unsqueeze(0)
        
        ov_t_a = self.ov_t_a(feat[0]).view(-1, self.o_classes, self.num_low_rank)
        ov_t_b = self.ov_t_b(feat[0]).view(-1, self.num_low_rank, self.v_classes)
        ov_t_0 = torch.bmm(ov_t_a, ov_t_b).unsqueeze(0)
        
        vs_t_a = self.vs_t_a(feat[0]).view(-1, self.v_classes, self.num_low_rank)
        vs_t_b = self.vs_t_b(feat[0]).view(-1, self.num_low_rank, self.s_classes)
        vs_t_0 = torch.bmm(vs_t_a, vs_t_b).unsqueeze(0)
        
        os_t_a = self.os_t_a(feat[0]).view(-1, self.o_classes, self.num_low_rank)
        os_t_b = self.os_t_b(feat[0]).view(-1, self.num_low_rank, self.s_classes)
        os_t_0 = torch.bmm(os_t_a, os_t_b).unsqueeze(0)
        
        vo_t_a = self.vo_t_a(feat[0]).view(-1, self.v_classes, self.num_low_rank)
        vo_t_b = self.vo_t_b(feat[0]).view(-1, self.num_low_rank, self.o_classes)
        vo_t_0 = torch.bmm(vo_t_a, vo_t_b).unsqueeze(0)
        
        sv_t_a = self.sv_t_a(feat[0]).view(-1, self.s_classes, self.num_low_rank)
        sv_t_b = self.sv_t_b(feat[0]).view(-1, self.num_low_rank, self.v_classes)
        sv_t_0 = torch.bmm(sv_t_a, sv_t_b).unsqueeze(0)

        # t = 1, 2, ... , 8, 9      feat = [10, 2, 1024]  #[time, batch, feature]
        # feat[i]=[2, 1024] → ss_a(feat)=[2, 85] → ss_a(feat).view=[2, 17, 5] → ss_a(feat).view.unsqueese(0)=[1, 2, 17, 5] → concat=[9, 2, 17, 5]
        ss_a = torch.cat([self.ss_a(feat[i]).view(-1, self.s_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('ss_a.shape={}'.format(ss_a.shape)) # [9,2,17,5]
        ss_b = torch.cat([self.ss_b(feat[i+1]).view(-1, self.num_low_rank, self.s_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('ss_b.shape={}'.format(ss_b.shape)) # [9,2,5,17]
        ss = torch.cat([torch.bmm(ss_a[i], ss_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('ss.shape={}'.format(ss.shape))  # [9,2,17,17]

        oo_a = torch.cat([self.oo_a(feat[i]).view(-1, self.o_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('oo_a.shape={}'.format(oo_a.shape)) # [9,2,39,5]
        oo_b = torch.cat([self.oo_b(feat[i+1]).view(-1, self.num_low_rank, self.o_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('oo_b.shape={}'.format(oo_b.shape)) # [9,2,5,39]
        oo = torch.cat([torch.bmm(oo_a[i], oo_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('oo.shape={}'.format(oo.shape)) # [9,2,39,39]

        vv_a = torch.cat([self.vv_a(feat[i]).view(-1, self.v_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vv_a.shape={}'.format(vv_a.shape)) # [9,2,34,5]
        vv_b = torch.cat([self.vv_b(feat[i+1]).view(-1, self.num_low_rank, self.v_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vv_b.shape={}'.format(vv_b.shape)) # [9,2,5,34]
        vv = torch.cat([torch.bmm(vv_a[i], vv_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vv.shape={}'.format(vv.shape)) # [9,2,34,34]

        so_t_a = torch.cat([self.so_t_a(feat[i]).view(-1, self.s_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('so_t_a.shape={}'.format(so_t_a.shape)) # [9,2,17,5]
        so_t_b = torch.cat([self.so_t_b(feat[i+1]).view(-1, self.num_low_rank, self.o_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('so_t_b.shape={}'.format(so_t_b.shape)) # [9,2,5,39]
        so_t = torch.cat([torch.bmm(so_t_a[i], so_t_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('so_t.shape={}'.format(so_t.shape)) # [9,2,17,39]

        ov_t_a = torch.cat([self.ov_t_a(feat[i]).view(-1, self.o_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('ov_t_a.shape={}'.format(ov_t_a.shape)) # [9,2,39,5]
        ov_t_b = torch.cat([self.ov_t_b(feat[i+1]).view(-1, self.num_low_rank, self.v_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('ov_t_b.shape={}'.format(ov_t_b.shape)) # [9,2,5,34]
        ov_t = torch.cat([torch.bmm(ov_t_a[i], ov_t_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('ov_t.shape={}'.format(ov_t.shape))     # [9,2,39,54]
    
        vs_t_a = torch.cat([self.vs_t_a(feat[i]).view(-1, self.v_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vs_t_a.shape={}'.format(vs_t_a.shape)) # [9,2,34,5]
        vs_t_b = torch.cat([self.vs_t_b(feat[i+1]).view(-1, self.num_low_rank, self.s_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vs_t_b.shape={}'.format(vs_t_b.shape)) # [9,2,5,17]
        vs_t = torch.cat([torch.bmm(vs_t_a[i], vs_t_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vs_t.shape={}'.format(vs_t.shape))  # [9,2,34,17]

        os_t_a = torch.cat([self.os_t_a(feat[i]).view(-1, self.o_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('os_t_a.shape={}'.format(os_t_a.shape)) # [9,2,39,5]
        os_t_b = torch.cat([self.os_t_b(feat[i+1]).view(-1, self.num_low_rank, self.s_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('os_t_b.shape={}'.format(os_t_b.shape)) # [9,2,5,17]
        os_t = torch.cat([torch.bmm(os_t_a[i], os_t_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('os_t.shape={}'.format(os_t.shape)) # [9,2,39,17]

        vo_t_a = torch.cat([self.vo_t_a(feat[i]).view(-1, self.v_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vo_t_a.shape={}'.format(vo_t_a.shape)) # [9,2,34,5]
        vo_t_b = torch.cat([self.vo_t_b(feat[i+1]).view(-1, self.num_low_rank, self.o_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vo_t_b.shape={}'.format(vo_t_b.shape)) # [9,2,34,39]
        vo_t = torch.cat([torch.bmm(vo_t_a[i], vo_t_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('vo_t.shape={}'.format(vo_t.shape)) # [9,2,34,39]

        sv_t_a = torch.cat([self.sv_t_a(feat[i]).view(-1, self.s_classes, self.num_low_rank).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('sv_t_a.shape={}'.format(sv_t_a.shape)) # [9,2,17,5]
        sv_t_b = torch.cat([self.sv_t_b(feat[i+1]).view(-1, self.num_low_rank, self.v_classes).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('sv_t_b.shape={}'.format(sv_t_b.shape)) # [9,2,5,34]
        sv_t = torch.cat([torch.bmm(sv_t_a[i], sv_t_b[i]).unsqueeze(0) for i in range(0, feat.shape[0]-1)], dim=0)
        #print('sv_t.shape={}'.format(sv_t.shape)) # [9,2,17,34]

        # concat(0時刻 + 1, 2, ... , 8, 9時刻)
        ss = torch.cat([ss_0, ss], dim=0)
        oo = torch.cat([oo_0, oo], dim=0)
        vv = torch.cat([vv_0, vv], dim=0)
        so_t = torch.cat([so_t_0, so_t], dim=0)
        ov_t = torch.cat([ov_t_0, ov_t], dim=0)
        vs_t = torch.cat([vs_t_0, vs_t], dim=0)
        os_t = torch.cat([os_t_0, os_t], dim=0)
        vo_t = torch.cat([vo_t_0, vo_t], dim=0)
        sv_t = torch.cat([sv_t_0, sv_t], dim=0)

        # print debug
        """
        print('s.shape={}'.format(s.shape)) # [10,2,17]
        print('o.shape={}'.format(o.shape)) # [10,2,39]
        print('v.shape={}'.format(v.shape)) # [10,2,34]
        print('so.shape={}'.format(so.shape)) # [10,2,17,39]
        print('ov.shape={}'.format(ov.shape))
        print('vs.shape={}'.format(vs.shape))
        print('ss.shape={}'.format(ss.shape))
        print('oo.shape={}'.format(oo.shape))
        print('vv.shape={}'.format(vv.shape))
        print('so_t.shape={}'.format(so_t.shape))
        print('ov_t.shape={}'.format(ov_t.shape))
        print('vs_t.shape={}'.format(vs_t.shape))
        print('os_t.shape={}'.format(os_t.shape))
        print('vo_t.shape={}'.format(vo_t.shape))
        print('sv_t.shape={}'.format(sv_t.shape))
        """        

        return s, o, v, so, ov, vs, ss, oo, vv, so_t, ov_t, vs_t, os_t, vo_t, sv_t