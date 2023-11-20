#%%
from dataloader import load_data
from models import *
import torch.optim as optim
import torch.nn.functional as F
import torch
from sklearn.metrics import confusion_matrix

data_dir = f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/training_data_6_tumors.csv"
batch_size = 6
max_epoch = 250
feature_transform = False
eval_interval = 2
atention_pooling_flag = True
outf = "/isilon/datalake/cialab/scratch/cialab/Hao/work_record/Project1_GM/codes/Point_cloud_gene_expression/saved_models"
gene_space_dim = 3
LOSS_SELECT = 'CE' # 'CE' or 'NLL'
WEIGHT_LOSS_FLAG = True
MULTI_GPU_FLAG = False





gene_number_name_mapping, number_to_label,feature_num, train_loader, val_loader, test_loader = load_data(file_path=data_dir, batch_size=batch_size)

class_num = len(number_to_label.keys())
samples_per_class = [feature_num[i] for i in range(len(feature_num))]
# Calculate class weights
total_samples = sum(samples_per_class)
class_weights = [total_samples / samples_per_class[i] for i in range(len(samples_per_class))]

# Normalize weights so that their sum equals the number of classes
weight_sum = sum(class_weights)
normalized_weights = torch.tensor([w / weight_sum * len(feature_num) for w in class_weights])
print("Normalized Class Weights:", normalized_weights)
#%%
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = PointNetCls(gene_idx_dim = 2, 
                    gene_space_num = gene_space_dim, 
                    class_num=class_num, 
                    feature_transform=feature_transform, 
                    atention_pooling_flag = atention_pooling_flag)

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
model.to(device)

def get_loss_criterion(LOSS_SELECT, WEIGHT_LOSS_FLAG, normalized_weights):
    if LOSS_SELECT == 'CE':
        if WEIGHT_LOSS_FLAG:
            criterion = nn.CrossEntropyLoss(weight=normalized_weights)
        else:
            criterion = nn.CrossEntropyLoss()
    elif LOSS_SELECT == 'NLL':
        if WEIGHT_LOSS_FLAG:
            criterion = nn.NLLLoss(weight=normalized_weights)
        else:
            criterion = nn.NLLLoss()
    else:
        raise ValueError("Invalid LOSS_SELECT value.")
    return criterion
#%% train
best_acc = 0
for epoch in range(max_epoch):
    scheduler.step()
    for i , data in enumerate(train_loader, 0):
        features1_count, features2_gene_idx, labels = data
        features1_count, features2_gene_idx = transpose_input(features1_count, features2_gene_idx)
        features1_count, features2_gene_idx, labels = features1_count.to(device), features2_gene_idx.to(device), labels.to(device)
        optimizer.zero_grad()
        model = model.train()
        pred, trans, trans_feat = model(features1_count, features2_gene_idx)
        normalized_weights = normalized_weights.to(device)
        criterion = get_loss_criterion(LOSS_SELECT, WEIGHT_LOSS_FLAG, normalized_weights)
        if LOSS_SELECT == 'NLL':
            pred = F.log_softmax(pred, dim=1)
        loss = criterion(pred, labels)
        # loss = F.nll_loss(F.log_softmax(pred, dim=1), labels)
        if feature_transform:
            loss += feature_transform_regularizer(trans_feat) * 0.001
        loss.backward()
        optimizer.step()
        pred_labels = torch.argmax(pred, dim=1)
        correct = torch.sum(pred_labels == labels)/float(batch_size)
        print(f"[{epoch}: {i}/{len(train_loader)}] train loss: {loss.item()} accuracy: {correct.item()}")

    
    if epoch % eval_interval == 0:
        confusion_matrix_all = np.zeros((class_num, class_num))
        correct_all = 0
        for i, data in enumerate(val_loader, 0):
            features1_count, features2_gene_idx, labels = data
            features1_count, features2_gene_idx = transpose_input(features1_count, features2_gene_idx)
            features1_count, features2_gene_idx, labels = features1_count.to(device), features2_gene_idx.to(device), labels.to(device)
            model = model.eval()
            pred, _, _ = model(features1_count, features2_gene_idx)
            pred_labels = torch.argmax(pred, dim=1)
            correct = torch.sum(pred_labels == labels)/float(batch_size)
            correct_all += torch.sum(pred_labels == labels)
            print(f"[{epoch}: {i}/{len(val_loader)}] val accuracy: {correct.item()}")
            pred_labels_np = pred_labels.cpu().numpy()
            labels_np = labels.cpu().numpy()
            for idx_batch in range(labels_np.shape[0]):
                label_i = labels_np[idx_batch]
                pred_i = pred_labels_np[idx_batch]
                confusion_matrix_all[label_i,pred_i] += 1
        correct_all = correct_all.cpu().numpy()
        if correct_all > best_acc:
            best_acc = correct_all
            torch.save(model.state_dict(), f"{outf}/cls_model_geneSpaceD_{gene_space_dim}_transfeat_{feature_transform}_attenpool_{atention_pooling_flag}_best.pth")
        print(confusion_matrix_all)
    torch.save(model.state_dict(), f"{outf}/cls_model_geneSpaceD_{gene_space_dim}_transfeat_{feature_transform}_attenpool_{atention_pooling_flag}_{epoch}.pth")

total_correct = 0
total_testset = 0
confusion_matrix_all = np.zeros((class_num, class_num))
for i,data in enumerate(test_loader, 0):
    features1_count, features2_gene_idx, labels = data
    features1_count, features2_gene_idx = transpose_input(features1_count, features2_gene_idx)
    features1_count, features2_gene_idx, labels = features1_count.to(device), features2_gene_idx.to(device), labels.to(device)
    model = model.eval()
    pred, _, _ = model(features1_count,features2_gene_idx)
    pred_choice = pred.data.max(1)[1]
    correct = torch.sum(pred_choice == labels)
    total_correct += correct.item()
    total_testset += features1_count.size()[0]

    pred_labels_np = pred_choice.cpu().numpy()
    labels_np = labels.cpu().numpy()
    for idx_batch in range(labels_np.shape[0]):
        label_i = labels_np[idx_batch]
        pred_i = pred_labels_np[idx_batch]
        confusion_matrix_all[label_i,pred_i] += 1

print("final accuracy {}".format(total_correct / float(total_testset)))
print(confusion_matrix_all)