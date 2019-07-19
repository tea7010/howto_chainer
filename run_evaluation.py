from my_libs.load_data import load_data, load_new_dataset
from my_libs.reset_seed import reset_seed
from my_libs.ChainerPipeline import ChainerPipeline

def learn_network_model(preprocess_inst, network_inst, setting):
    reset_seed(0)
    train, valid, test = load_new_dataset('../new_dataset/dataset/data/')
       
    p = ChainerPipeline(preprocess_inst, network_inst, train, valid, setting)
    model, result = p.run()
    return model

if __name__ == '__main__':
    import my_libs.network as my_net
    import my_libs.preprocess as my_process

    setting = {
        'epoch': 2,
        'batch_size': 32,
        'use_gpu': True,
        'fixed_base_w': True 
    }

    model = learn_network_model(my_process.Processing_10(), my_net.VGG(), setting)