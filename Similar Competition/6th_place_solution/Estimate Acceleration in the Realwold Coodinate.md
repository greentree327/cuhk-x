I’d like to share a few tips that might be useful.

The sensor data is originally recorded in the device’s coordinate system (i.e., relative to the position and orientation of the wrist-worn device). However, I believe that converting the data to a world coordinate system (i.e., relative to the real world) can make it more interpretable and possibly better feature representation.

Here’s the code I used to convert sensor data into world coordinates:

from scipy.spatial.transform import Rotation as R

def compute_acc_world(acc, rot):
    # acc: [:, x, y, z]
    # rot: [:, x, y, z, w]
    r = R.from_quat(rot)  # shape: (M,)
    acc_world = r.apply(acc)
    return acc_world
Result
The plot below shows the acceleration in world coordinates. It looks reasonable — the estimated acceleration appears to include gravitational acceleration (approximately 9.8 m/s²). Note that in this context, gravity should be defined as pointing upward, because the sensor detects acceleration in the direction opposite to the external force (imagine being in a rocket accelerating upward — gravity feels like it pushes down).



Reference
I also shared EDA notebook for reproducibility.