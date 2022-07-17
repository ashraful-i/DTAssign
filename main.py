from datetime import date

import pandas as pd
import numpy as np
from pprint import pprint

# Import the dataset and define the feature as well as the target datasets / columns#
# dataset = pd.read_csv('accelerometer.csv', sep=',', names=['wconfid','pctid','x','y','z', ])
dataset = pd.read_csv('accelerometer.csv', sep=',')
df = dataset.iloc[1:, :]

# print(df)
# dataset = dataset.to_csv(header=None,index=False)

def entropy_calc(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy = np.sum(
        [(-counts[i] / np.sum(counts)) * np.log2(counts[i] / np.sum(counts)) for i in range(len(elements))])
    return entropy


def InfoGain(data, split_attribute_name, target_name="wconfid"):
    # Calculate the entropy of the total dataset
    total_entropy = entropy_calc(data[target_name])

    # Calculate the entropy of the dataset

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts = np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    Weighted_Entropy = np.sum(
        [(counts[i] / np.sum(counts)) * entropy_calc(
            data.where(data[split_attribute_name] == vals[i]).dropna()[target_name])
         for i in range(len(vals))])

    # Calculate the information gain
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


def ID3(data, originaldata, features, target_attribute_name="wconfid", parent_node_class=None):
    # print("ID3")
    # Define the stopping criteria --> If one of this is satisfied, we want to return a leaf node#

    # If all target_values have the same value, return this value
    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]

        # If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) == 0:
        return np.unique(originaldata[target_attribute_name])[
            np.argmax(np.unique(originaldata[target_attribute_name], return_counts=True)[1])]

        # If the feature space is empty, return the mode target feature value of the direct parent node --> Note that
    # the direct parent node is that node which has called the current run of the ID3 algorithm and hence
    # the mode target feature value is stored in the parent_node_class variable.

    elif len(features) == 0:
        return parent_node_class

        # If none of the above holds true, grow the tree!

    else:
        # Set the default value for this node --> The mode target feature value of the current node
        parent_node_class = np.unique(data[target_attribute_name])[
            np.argmax(np.unique(data[target_attribute_name], return_counts=True)[1])]
        # print(parent_node_class)
        # Select the feature which best splits the dataset
        item_values = [InfoGain(data, feature, target_attribute_name) for feature in
                       features]  # Return the information gain values for the features in the dataset
        # print(item_values)
        best_feature_index = np.argmax(item_values)
        # print(best_feature_index)
        best_feature = features[best_feature_index]
        # print(best_feature)
        # Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        # gain in the first run
        tree = {best_feature: {}}

        # Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        # print("features ")
        # print(features)
        # Grow a branch under the root node for each possible value of the root node feature

        for value in np.unique(data[best_feature]):
            value = value
            # Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()

            # Call the ID3 algorithm for each of those sub_datasets with the new parameters --> Here the recursion comes in!
            subtree = ID3(sub_data, dataset, features, target_attribute_name, parent_node_class)

            # Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree

        # print(tree)
        return (tree)


def predict(query, tree_elm, default=1):
    for key in list(query.keys()):
        if key in list(tree_elm.keys()):

            try:
                result = tree_elm[key][query[key]]
            except:
                return default

            result = tree_elm[key][query[key]]

            if isinstance(result, dict):
                return predict(query, result)
            else:
                return result


def train_test_split(dataset_tmp):
    count_row = dataset_tmp.shape[0]
    training_cnt = int(count_row * 80 / 100)
    # print(count_row)
    training_dataset = dataset_tmp.iloc[:training_cnt].reset_index(drop=True)
    # We drop the index respectively relabel the index
    # starting form 0, because we do not want to run into errors regarding the row labels / indexes
    testing_dataset = dataset_tmp.iloc[training_cnt:].reset_index(drop=True)

    return training_dataset, testing_dataset


# print(dataset)
training_data = train_test_split(df)[0]
testing_data = train_test_split(df)[1]

print(training_data)
print(testing_data)


def test(data, tree_elm):
    print("test")
    # Create new query instances by simply removing the target feature column from the original dataset and
    # convert it to a dictionary
    queries = data.iloc[:, 1:].to_dict(orient="records")

    # Create a empty DataFrame in whose columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"])

    # Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i, "predicted"] = predict(queries[i], tree_elm, 1.0)
    print('The prediction accuracy is: ', (np.sum(predicted["predicted"] == data["wconfid"]) / len(data)) * 100, '%')

#print(training_data.columns[1:])
print("ID3 starts")
#tree = ID3(training_data, training_data, training_data.columns[1:])
print("Tree Print")
#pprint(tree)
#test(testing_data, tree)
