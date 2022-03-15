import pickle
import numpy as np
import torch
from models import SuperPoint, SuperGlue


fes = [SuperPoint(), SuperGlue()]
pth_paths = ["models/weights/superpoint_v1.pth", 
             "models/weights/superglue_outdoor.pth"]

for fe, pth_path in zip(fes, pth_paths):
    pp = fe.state_dict()
    tt = torch.load(pth_path)
    ntt = dict()
    # rm
    for k, v in tt.items():
        if 'num_batches_tracked' in k:
            continue
        if k.split('.')[-1] == 'running_mean':
            k = k.replace('running_mean', '_mean')
        if k.split('.')[-1] == 'running_var':
            k = k.replace('running_var', '_variance')
        ntt[k] = v
    print(len(pp.keys()), len(ntt.keys()))
    # print(pp.keys())
    # print(ntt.keys())
    dst = dict()
    for p, t in zip(sorted(pp.keys()), sorted(ntt.keys())):
        if p == t:
            dst[p] = np.array(ntt[t].detach().cpu())
        else:
            print(f"{p} cant write.")
    print("convert finished.")
    target_path = pth_path.replace(".pth", ".pdparams")
    pickle.dump(dst, open(target_path, 'wb'), protocol=2)
    print("save pdparams successfully.")