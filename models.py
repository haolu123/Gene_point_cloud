import numpy as np
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F

class GSNet(nn.Module):
    def __init__(self, k=2, out_k=3) -> None:
        super(GSNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 64, 1)
        self.conv3 = torch.nn.Conv1d(64, out_k, 1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.out_k = out_k
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = self.conv3(x)
        return x

class SNet(nn.Module):
    def __init__(self, k=3):
        super(SNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc4 = nn.Linear(k*k, 1)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.bn6 = nn.BatchNorm1d(k*k)
        self.k = k
    
    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = F.relu(self.bn6(self.fc3(x)))
        x = self.fc4(x)
        x = F.pad(x, (0, self.k*self.k-1), 'constant', 0)

        iden = np.eye(self.k)
        # Set the first element to 0
        iden[0, 0] = 0
        iden_tensor = Variable(torch.from_numpy(iden.flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden_tensor = iden_tensor.cuda()
        x = x + iden_tensor
        x = x.view(-1, self.k, self.k)
        return x


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x

class attmil(nn.Module):

    def __init__(self, inputd=1024, hd1=512, hd2=256):
        super(attmil, self).__init__()

        self.hd1 = hd1
        self.hd2 = hd2
        self.feature_extractor = nn.Sequential(
            torch.nn.Conv1d(inputd, hd1, 1),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            torch.nn.Conv1d(hd1, hd2,1),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            torch.nn.Conv1d(hd1, hd2, 1),
            nn.Sigmoid()
        )

        self.attention_weights = torch.nn.Conv1d(hd2, 1, 1)



    def forward(self, x):
        x = self.feature_extractor(x) # b*512*n

        A_V = self.attention_V(x)  # b*256*n
        A_U = self.attention_U(x)  # b*256*n
        A = self.attention_weights(A_V * A_U) # element wise multiplication # b*1*n
        A = A.permute(0, 2, 1)  # b*n*1
        A = F.softmax(A, dim=1)  # softmax over n

        # M = torch.matmul(A, x)  # 1x512
        # M = M.view(-1, self.hd1) # 512

        # Y_prob = self.classifier(M)

        # return Y_prob, A
        return A # batch_size x 1 x n
    
class PointNetfeat(nn.Module):
    def __init__(self, input_dim = 4, fstn_dim = 64, global_feat = True, feature_transform = False, atention_pooling_flag = False):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=input_dim)
        self.conv1 = torch.nn.Conv1d(input_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=fstn_dim)
        if atention_pooling_flag:
            self.atention_pooling = attmil()
        self.atention_pooling_flag = atention_pooling_flag

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))


        if self.atention_pooling_flag:
            A = self.atention_pooling(x)
            x = torch.bmm(x, A)
        else:
            x = torch.max(x, 2, keepdim=True)[0] ######## think about how to change it to attention pooling
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat

class PointNetCls(nn.Module):
    def __init__(self, gene_idx_dim = 2, gene_space_num = 3, class_num=10, feature_transform=False, atention_pooling_flag = False):
        super(PointNetCls, self).__init__()
        self.gstn = GSNet(k=gene_idx_dim)
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(input_dim = gene_space_num+1, global_feat=True, feature_transform=feature_transform, atention_pooling_flag = atention_pooling_flag)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, class_num)
        self.dropout = nn.Dropout(p=0.3)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x_feature, x_gene_idx):
        x_gene_idx = self.gstn(x_gene_idx)
        x = torch.cat([x_feature, x_gene_idx], 1)
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return x, trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, k = 2, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.k = k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        x, trans, trans_feat = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2,1).contiguous()
        x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.k)
        return x, trans, trans_feat

    
def feature_transform_regularizer(trans):
    d = trans.size()[1]
    batchsize = trans.size()[0]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2,1)) - I, dim=(1,2)))
    return loss

if __name__ == '__main__':
    device = torch.device("cuda:0")
    
    # test GSNet
    sim_data_gene_idx = Variable(torch.rand(8, 60660, 2))
    sim_data_gene_idx = sim_data_gene_idx.transpose(2, 1)
    gstn = GSNet(k=2, out_k=3)
    gene_space = gstn(sim_data_gene_idx)
    print('gsnet', gene_space.size())

    # test Concat
    sim_data_feature = Variable(torch.rand(8, 60660))
    sim_data_feature = sim_data_feature.unsqueeze(1)
    x = torch.cat([sim_data_feature, gene_space], 1)
    print('x', x.size())

    # test fstn
    trans = STNkd(k=4)
    out = trans(x)
    print('stn', out.size())
    print('loss', feature_transform_regularizer(out))

    # test PointNetfeat (max pooling)
    pointfeat = PointNetfeat(input_dim = 4, fstn_dim = 64, global_feat = True, feature_transform = False, atention_pooling_flag = False)
    pointfeat.to(device)
    x = x.to(device)
    out, _, _ = pointfeat(x)

    print('global feat', out.size())

    # test PointNetfeat (attention pooling)
    pointfeat = PointNetfeat(input_dim = 4, fstn_dim = 64, global_feat = True, feature_transform = False, atention_pooling_flag = True)
    pointfeat.to(device)
    x = x.to(device)
    out, _, _ = pointfeat(x)
    print('global feat', out.size())

    # test PointNetCls
    x_feature, x_gene_idx = Variable(torch.rand(2, 60660)), Variable(torch.rand(2, 60660, 2))
    x_gene_idx = x_gene_idx.transpose(2, 1)
    x_feature = x_feature.unsqueeze(1)
    cls_pointnet = PointNetCls(gene_idx_dim = 2, gene_space_num = 3, class_num=10, feature_transform=False, atention_pooling_flag = False)
    cls_pointnet.to(device)
    x_feature = x_feature.to(device)
    x_gene_idx = x_gene_idx.to(device)
    out, _, _ = cls_pointnet(x_feature, x_gene_idx)
    print('cls_pointnet', out.size())

def transpose_input(x_feature, x_gene_idx):
    x_gene_idx = x_gene_idx.transpose(2, 1)
    x_feature = x_feature.unsqueeze(1)
    return x_feature.float(), x_gene_idx.float()