import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
# import matplotlib as plt
# from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import os as os

#%%
# get current working directory
wd = os.getcwd()

# read in entire survey
# for surface
# sl19 = pd.read_stata("~/Documents/code/csci5622/csci5622mod2/proj/data/SLHR7ADT/SLHR7AFL.DTA")
# for PC
# sl19 = pd.read_stata("C:/Users/ilyon/OneDrive - UCB-O365/Documents/code/csci5622/mod2/proj/data/SLHR7ADT/SLHR7AFL.DTA")

# generalized
sl19 = pd.read_stata(wd + "\data\SLHR7ADT\SLHR7AFL.DTA")

#%%
# keep just columns of interest
cols2keep = ["hv000","hv001","hv006","hv007","hv010","hv011","hv012","hv013","hv014","hv024","hv025","hv040","hv045c","hv201","hv204","hv205","hv206","hv207","hv208","hv209","hv210","hv211","hv212","hv213","hv214","hv215","hv216","hv217","hv219","hv220","hv221","hv226","hv227","hv230a","hv237","hv241","hv243a","hv243b","hv243c","hv243d","hv243e","hv244","hv245","hv246","hv246a","hv246b","hv246c","hv246d","hv246e","hv246f","hv247","hv270","hv271","hv270a","hv271a","hml1"]
sl19keep = sl19[cols2keep]

#%%
# keep just numeric variables of interest
intCols = ["hv010","hv011","hv012","hv014","hv216","hv271"]
sl19keepNum = sl19[intCols]

# copy to prepare for cleaning
df = sl19keepNum

#%%
# convert strings into ints
# TODO come back and pull out some of these categories into integers
# df.loc[df["hv204"] == "on premises", "hv204"] = 0   # water from house
# df.loc[df["hv245"] == "don't know", "hv245"] = 1
# df.loc[df["hv245"] == "unknown", "hv245"] = 1
# df.loc[df["hv245"] == "95 or over", "hv245"] = 95

#%%
# scale data with mean = 0, stddev = 1
sl19scaled = StandardScaler().fit_transform(df)

#%%
# reduce columns down to 3 for clustering
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(sl19scaled)
principalDf = pd.DataFrame(data = principalComponents, columns = ['pc1', 'pc2'])
finalDf = pd.concat([principalDf, sl19keep[["hv206"]]], axis = 1)

#%%
# finalDf.plot(x='pc1', y='pc2', kind='scatter', title="PCA 1 vs. 2", hue='hv206')
sns.set(rc={'figure.figsize':(8,10)})
sns.relplot(data=finalDf, x="pc1", y="pc2", hue="hv206", size=0.5)

#%% 
# set number of clusters
kmeans = KMeans(n_clusters=6)

# run kmeans
identified_clusters = kmeans.fit_predict(sl19scaled)

# merge clusters with original dataset
sl19clustered = sl19keep.copy()
sl19clustered["clusters"] = identified_clusters
finalDf["clusters"] = identified_clusters

# plot clusters
sns.relplot(data=finalDf, x="pc1", y="pc2", hue="clusters", size=0.5)

# evaluate silhouette score
score = silhouette_score(sl19scaled, kmeans.labels_, metric='euclidean')
print(score)