class RobotContextCollector(object):
    def __init__(self, pose_adapter, people_adapter, social_adapter, sonar_adapter, logger):
        self.pose_adapter = pose_adapter
        self.people_adapter = people_adapter
        self.social_adapter = social_adapter
        self.sonar_adapter = sonar_adapter
        self.logger = logger

    def start(self):
        self.people_adapter.start()
        self.social_adapter.start()
        self.sonar_adapter.start()
        self.logger.info("Robot context collectors started")

    def stop(self):
        self.people_adapter.stop()
        self.social_adapter.stop()
        self.sonar_adapter.stop()
        self.logger.info("Robot context collectors stopped")

    def snapshot(self):
        pose = self.pose_adapter.snapshot()
        people = self.people_adapter.snapshot_people()
        social_people = self.social_adapter.snapshot_social_people(people)
        sonar = self.sonar_adapter.snapshot()
        snapshot = {
            "pose": pose,
            "people": people,
            "social_people": social_people,
            "sonar": sonar,
        }
        self.logger.info("Robot context snapshot ready, snapshot=%s", snapshot)
        return snapshot
