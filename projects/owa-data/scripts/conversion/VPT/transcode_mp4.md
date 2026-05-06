## Steps

1. `find /path/to/vpt/ -name "*.mp4" > filelist.txt`
2. `sbatch transcode_mp4.sbatch`


On 4 DGX H100 (112 physical core per each node), transcode takes 150files/minutes, total 3 hours for 26322 files. It accounts to 750 seconds(video)/seconds(real) or 2193 hours(data)/3 hours(real)