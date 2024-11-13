class LogsMixin:    

    def _log_info(self, action, path, extra=""):
        print(f"{self.__class__.__name__} {action} {path} {extra}")
