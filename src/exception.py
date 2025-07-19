import sys

def error_message_detail(error, error_detail):
    _, _, exc_tb = error_detail.exc_info()
    if exc_tb is not None:
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = (
            f"Error occurred in python script name [{file_name}] "
            f"line number [{exc_tb.tb_lineno}] error message [{error}]"
        )
    else:
        error_message = f"Error message [{error}] (no traceback available)"
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)

    def __str__(self):
        return self.error_message

if __name__ == "__main__":
    try:
        a = 1 / 0
    except Exception as e:
        print(CustomException(e, sys))
