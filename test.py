import numpy as np

# Generate random data
set1 = np.random.rand(50, 10)
set2 = np.random.randint(0, 10, size=(50, 1))
third_array = np.random.rand(50, 10)

# Set threshold values
threshold1 = 2
threshold2 = 0.5

# Get indices that meet conditions
indices = np.where((np.max(set1, axis=1) / threshold1 > set1[np.arange(len(set2)), set2.flatten()]) & 
                   (np.max(set1, axis=1) > threshold2) & 
                   (np.argmax(set1, axis=1) != set2.flatten()))[0]

# Print number of indices that meet conditions
print(f"Number of indices that meet conditions: {len(indices)}")
for index in indices:
  print(set1[index])
  print(set2[index])

new_labels = np.argmax(set1[indices], axis=1)
new_labels = new_labels.reshape(-1, 1)

set2[indices] = new_labels

print(new_labels.shape)
print(third_array.shape)
print(set2.shape)

# update set2 with new labels

not_matching = third_array[indices] != set2[indices]
matching = third_array[indices] == new_labels[indices]
count_matching = np.sum(np.logical_and(not_matching, matching))

# Count changes from matching to not matching third_array
matching = third_array[indices] == set2[indices]
not_matching = third_array[indices] != new_labels[indices]
count_not_matching = np.sum(np.logical_and(matching, not_matching))

for index in indices:
  print(set1[index])
  print(set2[index])

print(count_matching)
print(count_not_matching)
