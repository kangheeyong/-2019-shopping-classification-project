

def get_logger(name=__file__): 
    import logging
    #import logging.handlers
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(levelname)-8s] %(asctime)s [%(filename)s] [%(funcName)s:%(lineno)d] %(message)s', '%Y-%m-%d %H:%M:%S')
    sh.setFormatter(formatter)

    logger.addHandler(sh)
    return logger

class Option(dict):
    def __init__(self,json_file):
        import json
        with open(json_file) as f :
            json_data  = json.load(f)
            for key, value in json_data.items() :
                self[key] = value
    def __getattr__(self, attr):
        return self[attr] 

