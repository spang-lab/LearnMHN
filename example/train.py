from mhn.optimizers import StateSpaceOptimizer
import mhn.ssr.state_space_restriction as ssr
import pandas as pd
import numpy as np
import os

dataset = "G12_Breast_Prim"
file_dir = os.path.dirname(os.path.abspath(__file__))

if ssr.cuda_available() in [ssr.CUDA_NOT_AVAILABLE, ssr.CUDA_NOT_FUNCTIONAL]:
    raise RuntimeError(f"{ssr.cuda_available()}")


opt = StateSpaceOptimizer()
events = pd.read_csv(f"{file_dir}/{dataset}/{dataset}_finalEvents.csv", index_col=0)

opt.load_data_matrix(np.array(events, dtype=np.int32))

opt.set_callback_func(lambda theta: print("Training step complete."))

options = {
    # "lam": 0.0001,
    "lam": 1/len(events.columns)
    # "reltol": 1e-4,d
    
}
opt.train(
    **options
).save(f"{file_dir}/{dataset}/{dataset}_final_lam-{options.get('lam', 0)}_reltol-{options.get('reltol', 1e-7)}")
