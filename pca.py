##  PCA
import numpy as np
import pandas as pd
import random as rd
from sklearn.decomposition import PCA 
from sklearn import preprocessing
import matplotlib.pyplot as plt

#####################
#
# Generate data
#
#####################
genes = ['gene' + str(i) for i in range(1, 101)]
wt = ['wt' + str(i) for i in range(1, 6)]
ko = ['ko' + str(i) for i in range(1, 6)]
data = pd.DataFrame(columns=[*wt, *ko], index=genes)
# the "stars" unpack the "wt" and "ko" arrays so that the column names are a single array [wt1, wt2, ..., wt5, ko1, ko2, ... ko5]
# without the stars, we would create an array of two arrays [[wt1, wt2, ..., wt5], [ko1, ko2, ..., ko5]]
for gene in data.index:
    data.loc[gene, 'wt1': 'wt5'] = np.random.poisson(lam=rd.randrange(10, 100), size=5)
    data.loc[gene, 'ko1': 'ko5'] = np.random.poisson(lam=rd.randrange(10, 100), size=5)
# for each gene in the index (i.e., gene1, gene2, ..., gene100), we create 5 values for the 'wt' samples and 5 values for the 'ko' samples
# for each gene, we select a new mean for the poisson distribution. The means can vary betweeen 10 and 1000.
print(data.head()) # the head() method returns the first 5 rows of data
print(data.shape) # the shape attribute returns the dimensions of our data matrix
# in this case we get (100, 10). 100 genes by 10 total samples

#####################
#
# Perform PCA on the data
#
#####################
# first scale and center the data
scaled_data = preprocessing.scale(data.T)
# After centering, the average value for each gene will be 0, 
# and after scaling, the standard deviation for the values for each gene will be 1
# Notice that we are passing in the transpose of our data. The scale function expects the samples to be rows instead of columns 
# we can also use: StandardScaler().fit_transform(data.T)
# In sklearn, variation is calculated as: (measurements - mean) ** 2 / the number of measurements
pca = PCA() # create a PCA object
pca.fit(scaled_data) # this is where we do all of the PCA math, i.e., calcualte loading scores and the variation each principal component accounts for
pca_data = pca.transform(scaled_data) # generate coordinates for a PCA graph based on the loading scores and the scaled data

#####################
#
# Plot a scree plot and a PCA plot
#
#####################
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1) 
# calculate the percentage of variation that each principal component account for
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
# create labels for the scree plot. These are "PC1", "PC2", etc (one label per principal component)
# create a bar plot
plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, index=[*wt, *ko], columns=labels)
# creat a scatter plot
plt.scatter(pca_df.PC1, pca_df.PC2)
plt.title('My PCA Graph')
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
# add sample names to the graph
for sample in pca_df.index:
    plt.annotate(sample, (pca_df.PC1.loc[sample], pca_df.PC2.loc[sample]))
plt.show()

#########################
#
# Determine which genes had the biggest influence on PC1
#
#########################
loading_scores = pd.Series(pca.components_[0], index=genes) # create a pandas 'Series' object with the loading scores in PC1
# Note: The PCs are zero-indexed, so PC1 = 0.
sorted_loading_scores = loading_scores.abs().sort_values(ascending=True) # sort the loading scores based on their magnitude (absolute value)
top_10_genes = sorted_loading_scores[:10].index.values # get the names of the top 10 indexes (which are the gene names)
print(loading_scores[top_10_genes]) # print the gene names and their scores (and +/- sign)