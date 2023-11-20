#%%
from dataloader import load_data
from models import *
import torch.optim as optim
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix

data_dir = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/training_data_6_tumors.csv"
batch_size = 8
max_epoch = 250
feature_transform = False
eval_interval = 1
atention_pooling_flag = False
outf = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/Point_cloud_gene_expression/saved_models"
gene_space_dim = 3
gene_number_name_mapping, number_to_label,feature_num, train_loader, val_loader, test_loader = load_data(file_path=data_dir, batch_size=batch_size)

class_num = len(number_to_label.keys())
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = PointNetCls(gene_idx_dim = 2, 
                    gene_space_num = gene_space_dim, 
                    class_num=class_num, 
                    feature_transform=feature_transform, 
                    atention_pooling_flag = atention_pooling_flag)

model_state_dict = torch.load(outf+f"/cls_model_geneSpaceD_3_transfeat_False_attenpool_False_best.pth")
model.load_state_dict(model_state_dict)

total_correct = 0
total_testset = 0
confusion_matrix_all = np.zeros((class_num, class_num))
for i,data in enumerate(val_loader, 0):
    features1_count, features2_gene_idx, labels = data
    features1_count, features2_gene_idx = transpose_input(features1_count, features2_gene_idx)
    features1_count, features2_gene_idx, labels = features1_count.to(device), features2_gene_idx.to(device), labels.to(device)
    model = model.eval()
    pred, _, _ = model(features1_count,features2_gene_idx)
    pred_choice = pred.data.max(1)[1]
    correct = torch.sum(pred_choice == labels)
    total_correct += correct.item()
    total_testset += features1_count.shape[0]

    pred_labels_np = pred_choice.cpu().numpy()
    labels_np = labels.cpu().numpy()
    for idx_batch in range(labels_np.shape[0]):
        label_i = labels_np[idx_batch]
        pred_i = pred_labels_np[idx_batch]
        confusion_matrix_all[label_i,pred_i] += 1
    print("accuracy {}".format(correct.item()/float(batch_size)))
print("final accuracy {}".format(total_correct / float(total_testset)))
print(confusion_matrix_all)