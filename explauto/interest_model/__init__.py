import importlib


interest_models = {}

for mod_name in ['random', 'gmm_progress', 'discrete_progress', 'tree']:
    module = importlib.import_module('explauto.interest_model.{}'.format(mod_name))

    models = getattr(module, 'interest_models')

    for name, (im, conf) in list(models.items()):
        interest_models[name] = (im, conf)

def available_configurations(model):
    _, im_configs = interest_models[model]
    return im_configs
