import os

data_dir="data"
for f in sorted(os.listdir(data_dir)):
    size=os.path.getsize(os.path.join(data_dir,f))
    size_mb=round(size/1024/1024,2)
    print(f"  {f}({size_mb} MB)")





