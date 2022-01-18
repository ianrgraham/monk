# example showing showing how to launch jobs using snakemake

snakemake --profile slurm --configfile $1 --jobs 15 --forceall &
disown %1
