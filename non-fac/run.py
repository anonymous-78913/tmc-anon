from model import *
import sys
import os

output_filename = sys.argv[1]
basename = os.path.basename(output_filename)
#filename
#[fc|nf]_[K: 5 digit integer]_[seed: 2 digit integer]

fc_nf = basename[:2]
K = int(basename[3:8])
seed = int(basename[9:11])
t.manual_seed(seed)

mod = {"fc" : Fac, "nf" : NonFac}[fc_nf](100, 1.).cuda()

result = mod.reduce(K)

import pandas as pd
pd.DataFrame({
    'fc_nf' : [fc_nf],
    'N' : [100],
    'K' : [K],
    'seed' : [seed],
    'res' : [result.item()]
}).to_csv(output_filename, index=False)
