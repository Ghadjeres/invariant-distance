# Title     : TODO
# Objective : TODO
# Created by: gaetan
# Created on: 23/07/17

require(ggplot2)
require(grid)
require(reshape2)

df = read.csv('/home/gaetan/Projets/Python/workspace/DeepPermutations/deepPermutations/results/stats.csv',
header = TRUE)

ggplot(df, aes(x=distance, fill=label)) + geom_density(alpha=0.1) 

ggplot(df, aes(x=distance, fill=label)) + geom_histogram() 
+ facet_grid(.~label) 
+ xlim(0, 500)
