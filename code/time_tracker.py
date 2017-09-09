import time


class TimeTracker:

    def __init__(self, steps_amount):
        self.steps_amount = steps_amount
        self.start_time = time.time()
        self.timestamps = list()
        self.timestamps.append(("entry_point", 0, self.start_time))

    def reset(self):
        self.__init__(self.steps_amount)

    def log(self, operation_name="custom request"):
        timestamp = time.time()
        duration = timestamp - self.timestamps[-1][2]
        log_entry = (operation_name, duration, timestamp)

        self.timestamps.append(log_entry)
        print("{} operation has taken {:.2f} seconds".format(log_entry[0], log_entry[1]))

    def output_summary(self):
        elapsed = self.timestamps[-1][2] - self.timestamps[0][2]
        print("Total training time for {} batches: {:.2f} minutes".format(self.steps_amount, elapsed/60))