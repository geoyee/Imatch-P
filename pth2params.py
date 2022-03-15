import pickle
import numpy as np
import torch
from models import SuperPointFrontend


fe = SuperPointFrontend(weights_path="",
                        nms_dist=4,
                        conf_thresh=0.015,
                        nn_thresh=0.7)
pth_path = "models/weights/superpoint_v1.pth"

pp = fe.net.state_dict()
tt = torch.load(pth_path)

print(len(pp.keys()), len(tt.keys()))
dst = dict()
for p, t in zip(sorted(pp.keys()), sorted(tt.keys())):
    if p == t:
        dst[p] = np.array(tt[t].detach().cpu())
    else:
        print(f"{p} cant write.")
print("convert finished.")
target_path = pth_path.replace(".pth", ".pdparams")
pickle.dump(dst, open(target_path, 'wb'), protocol=2)
print("save pdparams successfully.")