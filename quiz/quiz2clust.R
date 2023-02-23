##################
## Quiz 2 ML Code Support - Clustering Questions
## Gates
########################################

library(stats)  
#https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/dist
#install.packages("NbClust")
library(NbClust)
library(cluster)
library(mclust)
library(factoextra)

# setwd("....................Edit this......................")
AdmissionsData<-read.csv("StudentSummerProgramDataQuiz2.csv")

## Create a dataframe from this dataset that can be used
## with kmeans and distance metrics such as Eucli Dists
## HINT:
(Num_Only_Adms_Data <- AdmissionsData[-c(1, 2, 3, 5, 8)])

df_2d <- AdmissionsData[-c(1, 2, 3, 5, 6, 8)]

## Distance Metric Matrices using dist
(Eucl_Dist <- stats::dist(Num_Only_Adms_Data,method="minkowski", p=2))  
(Eucl_Dist2D <- stats::dist(df_2d,method="minkowski", p=2))  
(Manh_Dist <- stats::dist(Num_Only_Adms_Data,method="minkowski", p=1))
å
library(stylo)
(CosSim <- stylo::dist.cosine(as.matrix(Num_Only_Adms_Data)))

Hist1 <- stats::hclust(Eucl_Dist, method="ward.D2")
plot(Hist1)

Hist2 <- stats::hclust(Manh_Dist, method="ward.D2")
plot(Hist2)
## Ward  - a general agglomerative 

##################  k - means----------------

k <- 3 # number of clusters
(kmeansResult <- stats::kmeans(Num_Only_Adms_Data, k)) ## uses Euclidean
kmeansResult$centers
kmeansResult$betweenss
kmeansResult$totss
kmeansResult$cluster


k <- 5 # number of clusters
(kmeansResult <- stats::kmeans(df_2d, k)) ## uses Euclidean
kmeansResult$centers
kmeansResult$cluster




############# To use a different sim metric----------
## one option is akmeans
## https://cran.r-project.org/web/packages/akmeans/akmeans.pdf

library(akmeans)
(akmeans(Num_Only_Adms_Data, min.k=3, max.k=3, verbose = TRUE))




################ Cluster vis-------------------
(factoextra::fviz_cluster(kmeansResult, data = Num_Only_Adms_Data,
                          ellipse.type = "convex",
                          #ellipse.type = "concave",
                          palette = "jco",
                          #axes = c(1, 4), # num axes = num docs (num rows)
                          ggtheme = theme_minimal()))


## Silhouette........................
factoextra::fviz_nbclust(Num_Only_Adms_Data, method = "silhouette", 
                         FUN = hcut, k.max = 5)
