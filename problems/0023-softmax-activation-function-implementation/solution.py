import math
import numpy as np

def softmax(scores: list[float]) -> list[float]:
    # Your code here
    z= np.array(scores)
    e= np.exp(z-z.max())
    total= np.sum(e)
    return e/total