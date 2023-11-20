import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np

def load_data(file_path, batch_size=8):
    # 1. Read the CSV file with MultiIndex
    data_dir = file_path
    df = pd.read_csv(data_dir, header=[0, 1], index_col=0)
    df.columns = pd.MultiIndex.from_tuples(df.columns)

    gene_names = df.index.values
    gene_name_number_mapping = {gene_names[i]: i for i in range(len(gene_names))}
    gene_number_name_mapping = {i: gene_names[i] for i in range(len(gene_names))}
    gene_numbers = np.arange(len(gene_names))

    # 2. Reshape and Preprocess the Data
    # Flatten the DataFrame
    data = []
    feature_num = {}
    for col in df.columns:
        label = col[0]  # First level of the MultiIndex is the class name
        features = df[col].values
        data.append((features, label))
        if label not in feature_num:
            feature_num[label] = 0
        else:
            feature_num[label] += 1
    # Separate features and labels
    features, labels = zip(*data)

    # Create a set of unique labels and sort it to maintain consistency
    unique_labels = sorted(set(labels))

    # Create a mapping dictionary from label to number
    label_to_number = {label: num for num, label in enumerate(unique_labels)}

    # Map your labels to numbers
    numerical_labels = [label_to_number[label] for label in labels]

    # To get the reverse mapping (from number to label), you can use:
    number_to_label = {num: label for label, num in label_to_number.items()}

    labels = numerical_labels
    feature_num = {label_to_number[key]: value for key, value in feature_num.items() if key in label_to_number}

    gene_numbers_len = len(gene_numbers)
    gene_numbers_len = np.round(np.sqrt(gene_numbers_len)) + 1
    # print(gene_numbers_len)
    gene_num_2d = np.zeros((len(gene_numbers), 2))
    for i in range(len(gene_numbers)):
        gene_num_2d[i, 0] = i // gene_numbers_len
        gene_num_2d[i, 1] = i % gene_numbers_len
    # print(gene_num_2d)

    features_mean = np.mean(features)
    features_std = np.std(features)
    features_normalized = (features - features_mean) / features_std

    gene_numbers_mean = np.mean(gene_numbers)
    gene_numbers_std = np.std(gene_numbers)
    gene_numbers_normalized = (gene_numbers - gene_numbers_mean) / gene_numbers_std 

    gene_num_2d_mean = np.mean(gene_num_2d)
    gene_num_2d_std = np.std(gene_num_2d)
    gene_num_2d_normalized = (gene_num_2d - gene_num_2d_mean) / gene_num_2d_std

    # 3. Create a Custom Dataset
    class TumorDataset(Dataset):
        def __init__(self, features_count, features_gene_idx, labels):
            self.features_count = features_count
            self.features_idx = features_gene_idx
            self.labels = labels

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, idx):
            sample_feature1 = self.features_count[idx]
            sample_feature2 = self.features_idx[idx]
            label = self.labels[idx]
            return sample_feature1, sample_feature2, label

    # 4. Split Dataset
    X_train, X_temp, y_train, y_temp = train_test_split(features_normalized, labels, test_size=0.3, random_state=42, stratify=labels)  # feature normalization
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Ensuring the test set has equal number of samples for each class
    class_counts = Counter(y_test)
    min_class_count = min(class_counts.values())
    indices = {label: np.where(y_test == label)[0][:min_class_count] for label in class_counts}
    balanced_indices = np.concatenate(list(indices.values()))
    X_test_balanced = [X_test[i] for i in balanced_indices]
    y_test_balanced = [y_test[i] for i in balanced_indices]

    gene_numbers_norm_tile_train = np.tile(gene_numbers_normalized, (X_train.shape[0], 1))
    gene_numbers_norm_tile_val = np.tile(gene_numbers_normalized, (X_val.shape[0], 1))
    gene_numbers_norm_tile_test = np.tile(gene_numbers_normalized, (len(X_test), 1))

    gene_num_2d_norm_tile_train = np.tile(gene_num_2d_normalized, (X_train.shape[0], 1, 1))
    gene_num_2d_norm_tile_val = np.tile(gene_num_2d_normalized, (X_val.shape[0], 1, 1))
    gene_num_2d_norm_tile_test = np.tile(gene_num_2d_normalized, (len(X_test), 1, 1))
    # Create PyTorch Datasets
    train_dataset = TumorDataset(X_train, gene_num_2d_norm_tile_train, y_train)
    val_dataset = TumorDataset(X_val, gene_num_2d_norm_tile_val, y_val)
    test_dataset = TumorDataset(X_test, gene_num_2d_norm_tile_test, y_test)

    # 5. Create DataLoaders

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return gene_number_name_mapping, number_to_label,feature_num, train_loader, val_loader, test_loader

if __name__ == '__main__':
    # Check the data loader.
    gene_number_name_mapping, number_to_label,feature_num, train_loader, val_loader, test_loader = load_data(file_path=f"/isilon/datalake/cialab/original/cialab/image_database/d00154/Tumor_gene_counts/training_data_6_tumors.csv", batch_size=8)
    for data_check in train_loader:
        # Unpack the data
        features1_check,features2_chekc, labels_check = data_check

        # Print the first element of the batch
        print("First feature batch:", features1_check[0])
        print("First feature batch:", features2_chekc[0])
        print("First label batch:", labels_check[0])

        # Break the loop after the first batch
        break