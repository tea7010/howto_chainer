from my_libs.load_data import load_data
from my_libs.reset_seed import reset_seed

from my_libs.ChainerPipeline import ChainerPipeline

def run_evaluation(preprocess_inst, network_inst, setting):
    reset_seed(0)
    train, test = load_data('../data')
       
    p = ChainerPipeline(preprocess_inst, network_inst, train, test, setting)
    model, resul = p.run()

if __name__ == '__main__':
    import my_libs.network as my_net
    import my_libs.preprocess as my_process

    setting = {
        'epoch': 100,
        'batch_size': 32,
        'use_gpu': True,
        'fixed_base_w': True
    }

    run_evaluation(my_process.Processing_3(), my_net.VGG(), setting)