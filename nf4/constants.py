import numpy as np

NF4_MAG = np.array(
    [     
        0.0,
        0.0911,
        0.1848,
        0.2844,
        0.3949,
        0.5251,
        0.6962,
        1.0,
        -0.0,
        -0.0911,
        -0.1848,
        -0.2844,
        -0.3949,
        -0.5251,
        -0.6962,
        -1.0,

    ],
    dtype=np.float32,
)

NF4_POS_MAG = NF4_MAG[0:8]  # Positive magnitudes only
print("NF4_POS_MAG:", NF4_POS_MAG)

