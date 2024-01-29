

class Model:

    def __init__(self, model_name: str):
        self.model_name = model_name

    def to_json(self):
        return {"model_name": self.model_name}

    @classmethod 
    def from_json(cls, json):
        return cls(json["model_name"]) 

    def build_model(self):
        """Build a model and return an executable function"""
        raise NotImplementedError


    def __call__(self, x, attempts=0):
        return self.build_model()(x)
