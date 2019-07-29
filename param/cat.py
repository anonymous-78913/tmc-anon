import sys
output_filename = sys.argv[1]
input_filenames = sys.argv[2:]

import pandas as pd
df = pd.concat(pd.read_csv(f) for f in input_filenames)
print(df)
df.to_csv(output_filename, sep=' ', index=False)
