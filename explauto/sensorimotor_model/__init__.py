import importlib


sensorimotor_models = {}

for mod_name in ['non_parametric']:
    module = importlib.import_module('explauto.sensorimotor_model.{}'.format(mod_name))

    models = getattr(module, 'sensorimotor_models')

    for name, (sm, conf) in list(models.items()):
        sensorimotor_models[name] = (sm, conf)

def available_configurations(model):
    _, sm_configs = sensorimotor_models[model]
    return sm_configs