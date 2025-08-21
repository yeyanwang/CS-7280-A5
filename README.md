# Assignment 5 – Network Modeling, Statistics, and Machine Learning  

## Overview  
The goal of this assignment is to explore network modeling, statistical analysis, and machine learning methods in network science. You will implement methods covered in **Module 5**, focusing on structural properties, network generation and estimation, and graph-based ML techniques for NLP.  

This assignment contains both core and bonus exercises. Students may choose which parts to complete, but the **maximum score is capped at 100 points**.  

---

## Notes & Guidelines  

- **Graph-Based ML Bonus Exercises**:  
  Parts 4 and 5 are new additions, carried over from previous semesters. You may attempt them to hedge your score.  

- **Flexibility in Completion**:  
  - Parts 1–3 alone = **100 points possible** (equivalent to previous semesters).  
  - Completing Parts 4 and/or 5 = optional buffer, but total score will not exceed 100.  

- **Part Connections**:  
  - Part 2 builds on functions from Part 1.  
  - Parts 4 and 5 are conceptually linked. You can attempt either independently, but Part 4 introduces concepts that flow into Part 5.  

---

## Imports & Dependencies  

This assignment builds on previous modules and introduces new packages for graph-based ML.  

```python
# Previous 7280 modules
import copy
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
import ridgeplot                       
import scipy as sp
import seaborn as sns

# NLP data
import nlp_data                     

# Label propagation ML
import sklearn
from sklearn.semi_supervised import LabelPropagation
from sklearn.feature_extraction.text import TfidfVectorizer

# GAT ML
import torch                                              
import torch.nn.functional as F                           
from torch_geometric.data import Data, DataLoader          
from torch_geometric.nn import GATConv, global_mean_pool   
from sklearn.model_selection import train_test_split
