
class CustomException(Exception):

    def __init__(self,message="CustomException has occurred"):
        self.message = message
        super().__init__(self.message)
