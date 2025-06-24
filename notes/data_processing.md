# Data subset command 
```
samtools view -h unmethylated_hifi_reads.bam | head -n 1000 | samtools view -b > unmethylated_subset.bam 
```


```
samtools view -h methylated_hifi_reads.bam | head -n 1000 | samtools view -b > methylated_subset.bam
```
