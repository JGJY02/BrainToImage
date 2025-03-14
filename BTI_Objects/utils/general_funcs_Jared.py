import os

def use_or_make_dir(main_dir, save_dir):

    save_path = os.path.join(main_dir, save_dir)
    print(save_path)
    if not os.path.exists(save_path): #Jared Edition
        os.makedirs(save_path)
    return save_path