from steinerpy.library.pipeline import Process

# specify baseline and results file path
bl_path = "empty-48-48.map_10t_25i_baseline.pkl"
res_path = "empty-48-48.map_10t_25i_1h_results.pkl"

gen_proc = Process("")
gen_proc.specify_files(bl_path, res_path)

assert(len(gen_proc.baseline_data["solution"])==len(gen_proc.main_results_data["solution"]))

blah = gen_proc.run()

print(blah)
pass