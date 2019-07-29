library(dplyr)

args <- commandArgs(trailingOnly=TRUE)

output_filename <- args[1]
df <- read.csv(args[2])

#df[is.infinite(as.matrix(df['res'])), 'res'] <- NA
df <- df[is.finite(as.matrix(df['res'])),]

print(head(df))

df <- df %>%
  group_by(fc_nf, N, K) %>%
  summarise(mean=mean(res), stderr=sqrt(var(res)/length(res)))#stdErr(value))

write.table(df, output_filename, quote=FALSE, row.names=FALSE)
