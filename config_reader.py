from configobj import ConfigObj

def config_reader():
    config = ConfigObj('config')
    opt = config['opt']
    opt['batch_size'] = int(opt['batch_size'])
    opt['batch_size_test'] = int(opt['batch_size_test'])
    opt['cuda'] = bool(opt['cuda'])
    opt['dropout_prob'] = float(opt['dropout_prob'])
    opt['lr'] = float(opt['lr'])
    opt['lr_fc'] = float(opt['lr_fc'])
    opt['manual_seed'] = int(opt['manual_seed'])
    opt['market'] = bool(opt['market'])
    opt['nbins'] = int(opt['nbins'])
    opt['nepoch'] = int(opt['nepoch'])
    opt['nepoch_fc'] = int(opt['nepoch_fc'])
    opt['nworkers'] = int(opt['nworkers'])
    opt['visdom_port'] = int(opt['visdom_port'])

    return opt

if __name__ == "__main__":
    config_reader()

