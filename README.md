# Task-2-Mainflow-Services-and-Technologies

import pandas as pd

# Load the CSV file
df = pd.read_csv('notable_ai_models.csv')

# Filter data
filtered_df = df[df['Organization categorization'] == 'industry']

# Handle missing values
df_filled = df.fillna(0)

# Calculate summary statistics
summary_stats = df.describe()

# Display results
print("Filtered DataFrame:")
print(filtered_df)
print("\nFilled DataFrame:")
print(df_filled)
print("\nSummary Statistics:")
print(summary_stats)

#OUTPUT

Filtered DataFrame:
Empty DataFrame
Columns: [System, Domain, Organization, Organization categorization, Country (from Organization), Authors, Publication date, Reference, Link, Citations, Notability criteria, Notability criteria notes, Parameters, Parameters notes, Training compute (FLOP), Training compute notes, Training dataset, Training dataset notes, Training dataset size (datapoints), Dataset size notes, Epochs, Training time (hours), Training time notes, Training hardware, Hardware quantity, Hardware utilization, Training compute cost (2023 USD), Compute cost notes, Confidence, Abstract, Model accessibility, Base model, Finetune compute (FLOP), Finetune compute notes, Batch size, Batch size notes, Frontier model, Training power draw (W)]
Index: []

[0 rows x 38 columns]

Filled DataFrame:
                                 System       Domain  \
0                           Qwen2.5-72B     Language   
1                    Table Tennis Agent     Robotics   
2                            AFM-server     Language   
3                         AFM-on-device     Language   
4                       Mistral Large 2     Language   
..                                  ...          ...   
862  Sequence-based pattern recognition       Vision   
863              Self Organizing System        Other   
864                   Genetic algorithm  Mathematics   
865                               SNARC     Robotics   
866                             Theseus     Robotics   

                                    Organization Organization categorization  \
0                                        Alibaba                    Industry   
1                                Google DeepMind                    Industry   
2                                          Apple                    Industry   
3                                          Apple                    Industry   
4                                     Mistral AI                    Industry   
..                                           ...                         ...   
862  Massachusetts Institute of Technology (MIT)                    Academia   
863  Massachusetts Institute of Technology (MIT)                    Academia   
864                 Institute for Advanced Study                    Academia   
865                           Harvard University                    Academia   
866                            Bell Laboratories                    Industry   

    Country (from Organization)  \
0                         China   
1                 Multinational   
2      United States of America   
3      United States of America   
4                        France   
..                          ...   
862    United States of America   
863    United States of America   
864    United States of America   
865    United States of America   
866    United States of America   

                                               Authors Publication date  \
0                                                    0       2024-09-19   
1    David B. D'Ambrosio, Saminda Abeyruwan, Laura ...       2024-08-07   
2    Andy Narayanan, Aonan Zhang, Bowen Zhang, Chen...       2024-07-29   
3    Andy Narayanan, Aonan Zhang, Bowen Zhang, Chen...       2024-07-29   
4    Albert Jiang, Alexandre Sablayrolles, Alexis T...       2024-07-24   
..                                                 ...              ...   
862                                    O. G. Selfridge       1955-03-01   
863                       W. A. Clark and B. G. Farley       1955-03-01   
864                                      NA Barricelli       1954-07-02   
865                                      Marvin Minsky       1952-01-08   
866                                     Claude Shannon       1950-07-02   

                                             Reference  \
0               Qwen2.5: A Party of Foundation Models!   
1    Achieving Human Level Competitive Robot Table ...   
2        Apple Intelligence Foundation Language Models   
3        Apple Intelligence Foundation Language Models   
4    Top-tier reasoning for high-complexity tasks, ...   
..                                                 ...   
862           Pattern recognition and modern computers   
863  Generalization of pattern recognition in a sel...   
864            Numerical testing of evolution theories   
865  A Neural-Analogue Calculator Based upon a Prob...   
866                                       Mighty Mouse   

                                                  Link  Citations  ...  \
0               https://qwenlm.github.io/blog/qwen2.5/        0.0  ...   
1    https://deepmind.google/research/publications/...        0.0  ...   
2    https://machinelearning.apple.com/research/app...        0.0  ...   
3    https://machinelearning.apple.com/research/app...        0.0  ...   
4          https://mistral.ai/news/mistral-large-2407/        0.0  ...   
..                                                 ...        ...  ...   
862     https://dl.acm.org/doi/10.1145/1455292.1455310      290.0  ...   
863     https://dl.acm.org/doi/10.1145/1455292.1455309       93.0  ...   
864  https://link.springer.com/article/10.1007/BF01...      266.0  ...   
865  https://en.wikipedia.org/wiki/Stochastic_neura...       33.0  ...   
866  https://www.technologyreview.com/2018/12/19/13...        0.0  ...   

    Confidence                                           Abstract  \
0    Confident  In the past three months since Qwen2’s release...   
1       Likely  Achieving human-level speed and performance on...   
2    Confident                                                  0   
3    Confident  We present foundation language models develope...   
4       Likely  Today, we are announcing Mistral Large 2, the ...   
..         ...                                                ...   
862    Unknown                                                  0   
863          0                                                  0   
864    Unknown                                                  0   
865          0                                                  0   
866          0                                                  0   

              Model accessibility Base model  Finetune compute (FLOP)  \
0      Open access (unrestricted)          0                      0.0   
1                      Unreleased          0                      0.0   
2          Hosted access (no API)          0                      0.0   
3          Hosted access (no API)          0                      0.0   
4    Open access (non-commercial)          0                      0.0   
..                            ...        ...                      ...   
862                             0          0                      0.0   
863                             0          0                      0.0   
864                             0          0                      0.0   
865                             0          0                      0.0   
866                             0          0                      0.0   

    Finetune compute notes    Batch size  \
0                        0  0.000000e+00   
1                        0  0.000000e+00   
2                        0  1.894975e+07   
3                        0  1.894975e+07   
4                        0  0.000000e+00   
..                     ...           ...   
862                      0  0.000000e+00   
863                      0  0.000000e+00   
864                      0  0.000000e+00   
865                      0  0.000000e+00   
866                      0  0.000000e+00   

                                      Batch size notes  Frontier model  \
0                                                    0               0   
1                                                    0               0   
2    Main pretraining uses sequence length of 4096 ...               0   
3    Main pretraining uses sequence length of 4096 ...               0   
4                                                    0               0   
..                                                 ...             ...   
862                                                  0               0   
863                                                  0               0   
864                                                  0               0   
865                                                  0               0   
866                                                  0            True   

    Training power draw (W)  
0              0.000000e+00  
1              0.000000e+00  
2              3.466814e+06  
3              0.000000e+00  
4              0.000000e+00  
..                      ...  
862            0.000000e+00  
863            0.000000e+00  
864            0.000000e+00  
865            0.000000e+00  
866            0.000000e+00  

[867 rows x 38 columns]

Summary Statistics:
           Citations    Parameters  Training compute (FLOP)  \
count     758.000000  5.800000e+02             4.220000e+02   
mean     5528.769129  3.285910e+10             6.046470e+23   
std     14878.359310  1.426448e+11             3.653663e+24   
min         0.000000  1.000000e+01             4.000000e+01   
25%       227.500000  1.775000e+07             1.413528e+17   
50%      1259.500000  1.825000e+08             1.035000e+20   
75%      4672.250000  3.277250e+09             2.596003e+22   
max    172714.000000  1.600000e+12             5.000000e+25   

       Training dataset size (datapoints)         Epochs  \
count                        4.290000e+02     210.000000   
mean                         4.110426e+11    1129.094214   
std                          1.771322e+12   13207.683793   
min                          0.000000e+00       0.170000   
25%                          1.150000e+05       3.542500   
50%                          5.000000e+07      30.000000   
75%                          1.200000e+10     150.000000   
max                          1.800000e+13  191400.000000   

       Training time (hours)  Hardware quantity  Hardware utilization  \
count             191.000000         156.000000             32.000000   
mean              494.261995        1562.115385              0.373699   
std               835.733900        5389.431347              0.108087   
min                 0.100000           1.000000              0.189200   
25%                36.500000          30.000000              0.299813   
50%               194.000000         256.000000              0.359000   
75%               600.000000         850.000000              0.464288   
max              7104.000000       55000.000000              0.565000   

       Training compute cost (2023 USD)  Finetune compute (FLOP)  \
count                      1.530000e+02             3.000000e+01   
mean                       8.365290e+05             9.807829e+21   
std                        4.281485e+06             3.892554e+22   
min                        8.542236e+00             0.000000e+00   
25%                        2.326664e+03             1.257522e+18   
50%                        1.588841e+04             7.850000e+19   
75%                        1.267858e+05             4.000000e+21   
max                        4.058659e+07             2.142900e+23   

         Batch size  Training power draw (W)  
count  6.500000e+01             1.220000e+02  
mean   4.438970e+06             1.477533e+06  
std    9.593190e+06             4.485470e+06  
min    2.700000e+01             5.113968e+02  
25%    6.553600e+04             1.793452e+04  
50%    2.000000e+06             2.225573e+05  
75%    4.000000e+06             5.579079e+05  
max    6.400000e+07             2.470979e+07  
