# import of libraries
import numpy as np
import matplotlib.pyplot as plt

 
# global declarations
data_path = "wifi_db/clean_dataset.txt"
MAX_DEPTH = 4
initial_data = np.loadtxt(data_path)


#definition of functions

#function for calculating entropy of a subset of data
def entropy(y):
    _, counts = np.unique(y, return_counts=True)
    probabilities = counts / len(y)
    return -np.sum(probabilities * np.log2(probabilities))

#function for calculating entropy gain given feature and threshold
def information_gain(X, y, feature_index, threshold):
    mask = X[:, feature_index] <= threshold
    left_y = y[mask]
    right_y = y[~mask]

    if len(left_y) == 0 or len(right_y) == 0:
        return 0

    total_entropy = entropy(y)
    left_entropy = entropy(left_y)
    right_entropy = entropy(right_y)

    left_weight = len(left_y) / len(y)
    right_weight = len(right_y) / len(y)

    return total_entropy - (left_weight * left_entropy + right_weight * right_entropy)

#function for finding best feature and threshold using the entropy and informationgain functions above
def find_best_split(X, y):
    best_feature = None
    best_threshold = None
    best_gain = -1

    for feature_index in range(X.shape[1]):
        thresholds = np.unique(X[:, feature_index])
        for threshold in thresholds:
            gain = information_gain(X, y, feature_index, threshold)
            if gain > best_gain:
                best_gain = gain
                best_feature = feature_index
                best_threshold = threshold

    return best_feature, best_threshold

#recursive function to build tree from root. 
def build_tree(X, y, depth=0, max_depth=None):
    if len(set(y)) == 1:
        return {'class': y[0]}

    if max_depth is not None and depth == max_depth:
        return {'class': np.argmax(np.bincount(y))}

    best_feature, best_threshold = find_best_split(X, y)

    if best_feature is None:
        return {'class': np.argmax(np.bincount(y))}

    mask = X[:, best_feature] <= best_threshold
    left_subtree = build_tree(X[mask], y[mask], depth + 1, max_depth)
    right_subtree = build_tree(X[~mask], y[~mask], depth + 1, max_depth)

    return {'feature': best_feature, 'threshold': best_threshold,
            'left': left_subtree, 'right': right_subtree}

#recursive function to predict y with given X and tree
def predict_tree(tree, X):
    if 'class' in tree:
        return tree['class']
    if X[tree['feature']] <= tree['threshold']:
        return predict_tree(tree['left'], X)
    else:
        return predict_tree(tree['right'], X)

#plotting tree with matplotlib
def plot_tree_manual(tree, x=0, y=0, layer=0, max_depth=MAX_DEPTH):
    if 'class' in tree:
        plt.text(x, y, f'Class {tree["class"]}', bbox=dict(facecolor='lightgreen', alpha=0.5),
                 ha='center', va='center', fontsize=6)
    else:
        plt.text(x, y, f'Feature {tree["feature"]} <= {tree["threshold"]}', bbox=dict(facecolor='lightblue', alpha=0.5),
                 ha='center', va='center', fontsize=6)

        '''the distance on matplotlib was manually tuned for a tree up to depth 4 such that there
         are no overlaps, some smarter ones can be implemented instead'''
        if layer < max_depth:
            if (layer == 0):
                dx = 1000
                dy = 1
            elif (layer == 1):
                dx = 500
                dy = 2
            elif (layer == 2):
                dx = 250
                dy = 2
            elif (layer == 3):
                dx= 125
                dy = 2
            else:
                dx = 60
                dy = 3

            # Draw left child
            plt.plot([x, x-dx], [y, y-dy], 'k-')
            plot_tree_manual(tree['left'], x-dx, y-dy, layer+1, max_depth)

            # Draw right child
            plt.plot([x, x+dx], [y, y-dy], 'k-')
            plot_tree_manual(tree['right'], x+dx, y-dy, layer+1, max_depth)



# main calls
np.random.shuffle(initial_data) #randomize order of the data

X = initial_data[:, :-1]  # Select all rows, and all columns except the last one
y = initial_data[:, -1]   # Select all rows, and only the last column
y = y.astype('int64')     

# Define the number of folds
num_folds = 10

# Calculate the size of each fold
fold_size = len(y) // num_folds

# Initialize an array to store the validation accuracies
validation_accuracies = []
# Initialize the confusion matrix
confusion_matrix = [
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0],
    [0,0,0,0]
]

# iteration for cross validation
for fold in range(num_folds):
    # Define the indices for the current fold
    start_idx = fold * fold_size
    end_idx = (fold + 1) * fold_size
    
    # Split the data into training and validation sets
    X_train = np.concatenate((X[:start_idx], X[end_idx:]), axis=0)
    y_train = np.concatenate((y[:start_idx], y[end_idx:]), axis=0)
    
    X_val = X[start_idx:end_idx]
    y_val = y[start_idx:end_idx]
    
    # Train model
    tree = build_tree(X_train, y_train, max_depth = MAX_DEPTH)
    
    # Make predictions
    predictions = [predict_tree(tree, validation_x) for validation_x in X_val]

    # Evaluate
    accuracy = np.sum(predictions == y_val) / len(y_val)
    for index in range(len(predictions)):
        confusion_matrix[y_val[index]-1][predictions[index]-1] += 1
    validation_accuracies.append(accuracy)

# Calculate the mean and standard deviation of the validation accuracies
mean_accuracy = np.mean(validation_accuracies)
std_dev_accuracy = np.std(validation_accuracies)

print(f'Mean accuracy: {mean_accuracy}')
print(f'Standard deviation: {std_dev_accuracy}')
print ('confusion matrix:', confusion_matrix)

# Retrain the model with the full dataset to maximize performance
tree = build_tree(X, y, max_depth=MAX_DEPTH) 

# draw figure with the new tree
plt.figure(figsize=(10, 10))
plt.axis('off')
plot_tree_manual(tree)
plt.savefig('decision_tree.png') # save new tree to local file

print("complete")