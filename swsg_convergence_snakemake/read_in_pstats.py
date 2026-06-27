import pstats

pstats.Stats('/home/jacob/SWSG_repo/swsg_convergence_snakemake/out.profile').sort_stats('time').print_stats()

import numpy