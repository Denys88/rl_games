from copy import deepcopy
import yaml
import json
import hashlib

# learning_rate: try [1e-4, 3e-4, 1e-3]
# gamma: try [0.95, 0.99]
# network.mlp.units: try [[256, 128], [512, 256]]

# Network.mlp.units

class Config():
    def __init__(self, data: dict):
        self._data = data
    
    def to_dict(self):
        return deepcopy(self._data)
            
    def clone(self):
        return Config(self.to_dict())

    def get(self, key: str, default=None):
        """Dot notation access: config.get('model.hidden_size')"""
        keys = key.split('.')
        data = self._data
        for k in keys:
            if isinstance(data, dict) and k in data:
                data = data[k]
            else:
                return default # Or raise ValueError(f"Key '{key}' not found in config")?

        return data

    def set(self, key: str, value: any):
        keys = key.split('.')
        data = self._data
        for k in keys[:-1]:
            if k not in data:
                data[k] = {}

            if not isinstance(data[k], dict):
                raise ValueError(f"Cannot set '{key}' because '{k}' is not a dict")

            data = data[k]

        data[keys[-1]] = value

    def __getitem__(self, key: str):
        result = self.get(key)
        if result is None:
            raise KeyError(f"Key {key} not found in config")
        return result
    
    def __setitem__(self, key: str, value: any):
        result = self.set(key, value)

    def flatten(self):
        """Flatten nested dict into dot notation: {'model': {'hidden_size': 128}} -> {'model.hidden_size': 128}"""
        flat = {}
        def _recurse(d, prefix='', sep='.'):
            for key, val in d.items():
                full_key = f"{prefix}{sep}{key}" if prefix else key
                if isinstance(val, dict):
                    _recurse(val, full_key, sep)
                else:
                    flat[full_key] = val

        _recurse(self._data)
        return flat
    
    def _deep_merge(self, base: dict, override: dict):
        # We mutate base
        for key, val in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(val, dict):
                self._deep_merge(base[key], val)
            else:
                # Should we check for existance and create new keys?
                # Should we check for dict/scala mismatch in base and override?
                base[key] = val
        return base

    def merge(self, other: 'Config'):
        # Should we merge in-place or return a new config?
        return Config(self._deep_merge(deepcopy(self._data), other._data))

    @classmethod
    def from_yaml_string(cls, yaml_config: str):
        data = yaml.safe_load(yaml_config)
        return cls(data)

    @classmethod
    def from_yaml_file(cls, path: str):
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(data)
    
    def to_yaml_string(self, sort_keys=False):
        yaml_str = yaml.dump(self._data)
        return yaml_str
    
    def to_yaml_file(self, path: str, sort_keys=False):
        with open(path, 'w') as f:
            f.write(self.to_yaml_string(sort_keys))

    def fingerprint(self):
        # yaml_str = self.to_yaml(sort_keys=True)
        # yaml_enc = yaml_str.encode('utf-8')
        # But can use yaml.dump() as well
        serialized = json.dumps(self._data, sort_keys=True)
        bytes_data = serialized.encode('utf-8')
        
        return hashlib.sha256(bytes_data).hexdigest()

    def __repr__(self):
        return f"Config({self._data})"


class SweepSpec():
    pass


class ConfigValidator():
    pass

