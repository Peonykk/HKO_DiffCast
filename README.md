## HKO_DiffCast


- The folder consist of codes for operation of DiffCast with HKO trained dataset.
- Below will briefly point out the function of the codes, and give potential modification directions (codes marked with * is subjected to modifications).

### `diffcast.py`

This code is simply the model code of diffcast, but as we employ another deterministic model to make the inital guess, one may need to
run with the source code for inference.
Just make sure you got all packages installed.

### `functions.py`

This code only stores the essential functions that need for `diffcast.py`.

### `get_realtime_radar.py` *

This code is to get the realtime radar image from the hkodatabase. One need to take a look at the code and change the path to the radar images (and possibly the naming of them...).


### `multiple_infer_independent.py` *

The code is essentially the **inference code**, which is made for generating ensemble members for calculating ensemble mean for forecasting. But as one may need one member prediction only, you could put 1 in the member argument simply. Below explain some of the command line outputs:

```
1. --gpu: set 1 for single member prediction
2. --ddim_steps: 20 is workable, but default is 250, smaller would be faster but lower quality.
3. --member_id: please use default for single member inference.
4. --test_datetime: The datetime for inference (e.g. 202603021100), but one may ignore it if the code in searching for radar images is modified.
5. --ckpt_path: path of the checkpoint.
6. --version_name: This just for savepath usage, you may change the savepath and remove this argument
7. --freq: Timestep for inference (e.g '12mins')

```

For the arguments that is not mentioned, please use defalt setting.


### `utilsss.py`

This code just store the essential functions for inference.


### `vmlstm_B.py`

This code is the deterministic model script use for making initial guess (replacing the phydnet used by the original author).









### Contact 

2026 1 year placement intern (F32)  
Lai Ka Kiu (Peony)  
[26peonylai@hko.hksarg](link)  
[1155210153@link.cuhk.edu.hk](link)  