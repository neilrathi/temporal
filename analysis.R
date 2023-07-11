library(tidyr)
library(ggplot2)
library(reshape2)

setwd('~/csboy/temporal/results')
files <- c('uniform-uniform', 'uniform-weighted', 'childes-uniform', 'childes-weighted')
for (i in files) {
  df <- read.csv(paste(i, '-f1.csv', sep = ''), sep = '\t')
  df %>%
    ggplot(aes(x = data, y = mean)) + 
    geom_line(aes(color = word), linewidth = 0.7) + 
    scale_color_brewer(palette = "Dark2") +
    scale_x_continuous(n.breaks=8) +
    xlab('Training Data Amount') + ylab('Test Data Accuracy (F1)') +
    ylim(0.5, 1) +
    theme_minimal() +
    theme(legend.position="top")
  ggsave(paste('../plots/nice/', i, '.pdf', sep = ''), width = 5, height = 4)
}

df <- read.csv('results-cuter.csv', sep = '\t')
df_long <- melt(data = df, id.vars = 'data', variable.name = 'word')
df_long %>%
  ggplot(aes(x = data, y = value)) + 
  geom_line(aes(color = word), linewidth = 0.7) + 
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(n.breaks=8) +
  xlab('Training Data Amount (epochs)') + ylab('Test Data Accuracy (F1)') +
  theme_minimal()
ggsave('../plots/nice/lm.pdf', width = 5, height = 2.5)

setwd('~/Downloads/')
df <- read.csv('bert-results.csv', sep = '\t')
df$logdata <- log(df$data, 10)
df %>%
  ggplot(aes(x = logdata, y = mean)) + 
  geom_line(aes(color = word), linewidth = 0.7) + 
  scale_color_brewer(palette = "Dark2") +
  scale_x_continuous(n.breaks=8) +
  scale_y_reverse() +
  xlab('Training Data Amount (epochs)') + ylab('Test Data Accuracy (F1)') +
  theme_minimal()