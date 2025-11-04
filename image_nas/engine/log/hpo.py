class HPOLog:
    def __init__(self, experiment):
        self.experiment = experiment

    def get_log(self):
        job_ids = self.get_trial_job_ids()
        jobs = []
        for job_id in job_ids:
            trial_job = self.get_trial_job_details(job_id)
            metric = self.get_job_metric(job_id)
            jobs.append({
                "job_id": job_id,
                "status": trial_job.status,
                "start_time": trial_job.startTime,
                "end_time": trial_job.endTime,
                "parameters": self._extract_parameters(trial_job.hyperParameters),
                "final_metric_data": metric[-1] if metric else 0,
                "metric": metric
            })
        return jobs

    def get_trial_job_ids(self):
        job_ids = []
        for trial_job in self.experiment.list_trial_jobs():
            job_id = trial_job.trialJobId
            if job_id:
                job_ids.append(job_id)
        return job_ids

    def get_job_metric(self, trial_id):
        job_metrics = self.experiment.get_job_metrics()

        metrics = []
        if trial_id in job_metrics:
            for job_metric in job_metrics[trial_id]:
                metric = job_metric.data
                if not metric >= 1.:
                    metrics.append(metric)
        return metrics

    def get_trial_job_details(self, job_id):
        for trial_job in self.experiment.list_trial_jobs():
            if trial_job.trialJobId == job_id:
                return trial_job
        return {}

    def get_status(self):
        return self.experiment.get_status()

    def get_job_statistic(self):
        statistics = self.experiment.get_job_statistics()
        return statistics

    def get_max_trial_num(self):
        profile = self.experiment.get_experiment_profile()
        return profile.get("params", {}).get("maxTrialNumber", 0)

    def get_start_time(self):
        profile = self.experiment.get_experiment_profile()
        return profile.get("startTime", 0)

    def _extract_parameters(self, hyperParameters):
        return hyperParameters[0].parameters
