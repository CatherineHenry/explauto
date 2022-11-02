import importlib
#from .environment import Environment


environments = {}
# https://github.com/flowersteam/explauto/commit/f7e51d4483846f982433b8eaa8b6f912ab9ceda3 -pypot was deprecated?
#for mod_name in ['simple_arm', 'pendulum', 'npendulum', 'pypot']:
for mod_name in ['simple_arm', 'pendulum', 'npendulum']:
    try:
        print(mod_name)
        module = importlib.import_module('explauto.environment.{}'.format(mod_name))
        env = getattr(module, 'environment')
        conf = getattr(module, 'configurations')
        testcases = getattr(module, 'testcases')
        environments[mod_name] = (env, conf, testcases)
    except ImportError as e:
        print(e)
        pass
def available_configurations(environment):
    _, env_configs, _ = environments[environment]
    return env_configs
