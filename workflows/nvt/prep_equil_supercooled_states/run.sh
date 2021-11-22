# example showing showing how to launch jobs using snakemake

snakemake --profile slurm --configfile $1 --jobs 14 --forceall &
disown %1