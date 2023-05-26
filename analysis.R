library(ggplot2)
library(reshape2)

#create data frame
df <- read.csv('accuracy.csv')
df$index <- 1:nrow(df) * 25

#melt data frame into long format
df <- melt(df ,  id.vars = 'index', variable.name = 'word')

#create line plot for each column in data frame
df %>%
  ggplot(aes(x = index, y = value)) + 
  geom_line(aes(color = word)) + 
  scale_x_continuous(n.breaks=13) +
  xlab('training data') + ylab('test accuracy') +
  theme_minimal()

ggsave('plots/accuracy.pdf', width = 6, height = 4)