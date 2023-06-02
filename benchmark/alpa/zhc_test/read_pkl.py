import pickle    

def run_app():
    path__pkl = "tmp_a100_gpu_real/gpt.grid_search_auto-2X1-actualA100-2023-03-01-02-57-12/Batchsize_1024-num_b_128-auto_layers_6/input_placement_specs.pkl"
    # tmp_a100_gpu_real/gpt.grid_search_auto-2X1-actualA100-2023-03-01-02-57-12/Batchsize_1024-num_b_128-auto_layers_6/input_placement_specs.pkl
    input_placement_specs = None
    with open(path__pkl, 'rb') as f:
        input_placement_specs = pickle.load(f)
    print(input_placement_specs)
    
    print('------------------------')

    with open("zhc_test/input_placement_specs_pkl_760M.txt",'w+') as txt_file:        
        for item in input_placement_specs:
            # print(item)
            str_line = ""
            if item is None:
                str_line= "None"
            else:
                str_line =str(item)
            txt_file.write(str_line+"\n")
            

    # print(input_placement_specs[0])    


if __name__ == "__main__":
    run_app()
    
    