from steinerpy.library.pipeline import Process

# specify baseline and results file path
bl_path = "empty-48-48.map_10t_100i_baseline.pkl"
res_path = "empty-48-48.map_10t_100i_0h_results.pkl"

gen_proc = Process("")
gen_proc.specify(bl_path, res_path)
gen_proc.run()
