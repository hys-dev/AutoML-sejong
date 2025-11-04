class Experiment:
    def __init__(self, task='nas', status='running', config=None):
        self.task = task
        self.status = status
        self.config = config
        self.retiarii_experiment = None
        self.opt_experiment = None
        self.concurrent_future = None

    def set_task(self, task):
        self.task = task

    def get_task(self):
        return self.task

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def keys(self):
        return ['task', 'status', 'config']

    def __getitem__(self, item):
        return getattr(self, item)
